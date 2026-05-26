#!/usr/bin/env python3
"""
Collect prompt / tool-call statistics from local logs of:
- Claude Code (jsonl): ~/.claude/projects/**.jsonl
- Codex CLI (jsonl):   ~/.codex/sessions/**/rollout-*.jsonl (or CODEX_HOME)
- Gemini CLI (json):   ~/.gemini/tmp/**/session-*.json

This is intentionally format-tolerant: it uses heuristics rather than strict schemas.
"""

from __future__ import annotations

import argparse
import io
import csv
import dataclasses
import datetime as dt
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Utilities
# -----------------------------

def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Some tools occasionally write non-JSON diagnostic lines; skip.
                continue


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iso_from_mtime(path: Path) -> str:
    ts = path.stat().st_mtime
    return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _pseudonym(s: str) -> str:
    # Stable pseudonym for a student without storing an identifier in submissions.
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _maybe_str(x: Any) -> str:
    return x if isinstance(x, str) else ""


def _lower(x: Any) -> str:
    return _maybe_str(x).lower()


# -----------------------------
# Heuristics
# -----------------------------

def _is_user_message(obj: Dict[str, Any]) -> bool:
    # Claude Code format: {"type": "user", "message": {...}}
    if _lower(obj.get("type")) == "user":
        return True
    # Common patterns:
    # {"role":"user", "content":...}
    # {"type":"message", "role":"user", ...}
    role = _lower(obj.get("role"))
    if role == "user":
        return True
    # Nested role in message
    if isinstance(obj.get("message"), dict):
        role = _lower(obj["message"].get("role"))
        if role == "user":
            return True
    # Codex format: {"payload": {"role": "user", ...}}
    if isinstance(obj.get("payload"), dict):
        role = _lower(obj["payload"].get("role"))
        if role == "user":
            return True
    # Some logs use "author" / "sender"
    author = _lower(obj.get("author") or obj.get("sender"))
    if author == "user":
        return True
    return False


def _is_assistant_message(obj: Dict[str, Any]) -> bool:
    # Claude Code format: {"type": "assistant", "message": {...}}
    if _lower(obj.get("type")) == "assistant":
        return True
    role = _lower(obj.get("role"))
    if role in ("assistant", "model"):
        return True
    # Nested role in message
    if isinstance(obj.get("message"), dict):
        role = _lower(obj["message"].get("role"))
        if role in ("assistant", "model"):
            return True
    # Codex format: {"payload": {"role": "assistant", ...}}
    if isinstance(obj.get("payload"), dict):
        role = _lower(obj["payload"].get("role"))
        if role in ("assistant", "model"):
            return True
    author = _lower(obj.get("author") or obj.get("sender"))
    if author in ("assistant", "model"):
        return True
    return False

def _infer_message_type(obj: Dict[str, Any]) -> str:
    """
    Infers message type (user, assistant_with_tools, tool_result, assistant_response, unclassified)
    based on various heuristics, especially when 'role' field is missing or generic.
    """
    role = _lower(obj.get("role"))

    # Explicit roles
    if role == "user":
        return "user"
    if role in ("assistant", "model"):
        if isinstance(obj.get("tool_calls"), list) and obj.get("tool_calls"):
            return "assistant_with_tools"
        return "assistant_response"
    if role == "tool":
        return "tool_result"

    # Heuristics if role is missing or not specific enough
    if isinstance(obj.get("tool_calls"), list) and obj.get("tool_calls"):
        return "assistant_with_tools"
    if isinstance(obj.get("tool_call_id"), str) and isinstance(obj.get("name"), str):
        return "tool_result"
    
    # Check for user message content patterns
    if isinstance(obj.get("content"), str) and not obj.get("tool_code"):
        return "user"
    if isinstance(obj.get("parts"), list):
        for part in obj["parts"]:
            if isinstance(part, dict) and part.get("text"):
                return "user"

    # Default to assistant response if nothing else matches (most common remaining case)
    # or if we found any common indicator for assistant (e.g. from _is_assistant_message)
    if _is_assistant_message(obj):
        return "assistant_response"

    return "unclassified"


def _count_tool_calls_in_obj(obj: Any) -> int:
    """
    Best-effort tool call counter across logs:
    - OpenAI-style: {"tool_calls":[...]} or {"message":{"tool_calls":[...]}}
    - Claude-style: {"type":"tool_use", ...} or {"tool_name":...}
    - Gemini-style: may store tool invocations in nested fields.
    """
    if obj is None:
        return 0

    if isinstance(obj, list):
        return sum(_count_tool_calls_in_obj(x) for x in obj)

    if not isinstance(obj, dict):
        return 0

    # Direct list of tool calls
    if isinstance(obj.get("tool_calls"), list):
        return len(obj["tool_calls"])

    # Claude Code often has "type":"tool_use"
    if _lower(obj.get("type")) in ("tool_use", "tool-call", "tool_call", "tool"):
        return 1

    # Explicit tool naming fields
    for k in ("tool_name", "tool", "name", "function"):
        v = obj.get(k)
        if isinstance(v, str) and v and k in ("tool_name", "tool"):
            return 1
        # OpenAI tool calls: {"function":{"name":"...","arguments":"..."}}
        if k == "function" and isinstance(v, dict) and isinstance(v.get("name"), str):
            return 1

    # Recurse into common containers
    total = 0
    for k, v in obj.items():
        if isinstance(v, (dict, list)):
            total += _count_tool_calls_in_obj(v)
    return total


def _guess_tool(obj: Dict[str, Any]) -> str:
    """
    Roughly classify which tool produced this record based on file path patterns.
    """
    return obj.get("_source", "unknown")


# -----------------------------
# Data model
# -----------------------------

@dataclass
class ToolStats:
    """Statistics for a specific tool type (Read, Write, etc.)"""
    count: int = 0
    execution_times: List[float] = field(default_factory=list)

    def avg_time(self) -> float:
        if not self.execution_times:
            return 0.0
        return sum(self.execution_times) / len(self.execution_times)


@dataclass
class TraceEvent:
    """A single event in the time-based trace"""
    timestamp: str
    event_type: str  # "user_prompt", "assistant_response", "tool_call", "tool_result"
    coding_agent: str  # "claude_code", "codex_cli", "gemini_cli"
    tool_name: Optional[str] = None
    execution_time: Optional[float] = None
    working_dir: Optional[str] = None
    session_id: Optional[str] = None
    message_text: Optional[str] = None
    # Token fields — populated only when --tokens is used (Claude Code JSONL only)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    model: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "coding_agent": self.coding_agent,
            "tool_name": self.tool_name or "",
            "execution_time": self.execution_time or "",
            "working_dir": self.working_dir or "",
            "session_id": self.session_id or "",
            "message_text": self.message_text or "",
            "input_tokens": self.input_tokens if self.input_tokens is not None else "",
            "output_tokens": self.output_tokens if self.output_tokens is not None else "",
            "cache_creation_tokens": self.cache_creation_tokens if self.cache_creation_tokens is not None else "",
            "cache_read_tokens": self.cache_read_tokens if self.cache_read_tokens is not None else "",
            "model": self.model or "",
        }


@dataclass
class SessionStats:
    tool: str
    session_id: str
    file: str
    mtime_iso: str
    session_cwd: Optional[str] = None  # Session-level working directory (from cwd/workdir fields)
    prompts: int = 0
    assistant_msgs: int = 0
    tool_calls: int = 0
    tool_stats: Dict[str, ToolStats] = field(default_factory=dict)
    iterations_per_prompt: List[int] = field(default_factory=list)
    prompt_response_times: List[float] = field(default_factory=list)
    working_dirs: List[str] = field(default_factory=list)  # Tool-level working dirs
    trace_events: List[TraceEvent] = field(default_factory=list)

    def to_row(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "session_id": self.session_id,
            "file": self.file,
            "mtime_iso": self.mtime_iso,
            "prompts": self.prompts,
            "assistant_msgs": self.assistant_msgs,
            "tool_calls": self.tool_calls,
        }


# -----------------------------
# Parsers
# -----------------------------

def _extract_user_text(obj: Dict[str, Any]) -> Optional[str]:
    """Extract plain text from a user message object (not tool results)."""
    content = obj.get("message", {}).get("content")
    if isinstance(content, str):
        return content.strip() or None
    return None


def _extract_working_dir(obj: Dict[str, Any]) -> Optional[str]:
    """Extract working directory from tool parameters"""
    if not isinstance(obj, dict):
        return None

    # Look for file_path, path, directory, cwd in tool inputs
    for key in ["file_path", "path", "directory", "cwd", "notebook_path"]:
        val = obj.get(key) or obj.get("input", {}).get(key)
        if isinstance(val, str) and val:
            # Extract directory from file path
            if "/" in val or "\\" in val:
                return str(Path(val).parent) if Path(val).parent != Path(".") else val
            return val

    # Recursively search nested dicts
    for v in obj.values():
        if isinstance(v, dict):
            result = _extract_working_dir(v)
            if result:
                return result

    return None


def _extract_tool_name(obj: Dict[str, Any]) -> Optional[str]:
    """Extract tool name from a tool_use object"""
    if not isinstance(obj, dict):
        return None

    # Claude Code format: {"type": "tool_use", "name": "Read", ...}
    if obj.get("type") == "tool_use" and isinstance(obj.get("name"), str):
        return obj["name"]

    # Alternative formats
    for key in ["tool_name", "tool", "name"]:
        val = obj.get(key)
        if isinstance(val, str) and val:
            return val

    return None


def _parse_timestamp(ts: Any, path: Optional[Path] = None) -> Optional[dt.datetime]:
    """Parse timestamp from various formats, logging a warning if unparseable."""
    if isinstance(ts, (int, float)):
        return dt.datetime.fromtimestamp(ts)
    if isinstance(ts, str):
        try:
            return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            if path:
                print(f"Warning: Could not parse timestamp '{ts}' in file {path}", file=sys.stderr)
            else:
                print(f"Warning: Could not parse timestamp '{ts}'", file=sys.stderr)
    return None


def _extract_token_usage(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract token usage dict from an assistant message object."""
    usage = obj.get("message", {}).get("usage")
    if not isinstance(usage, dict):
        return None
    return usage


def parse_claude_jsonl(path: Path, capture_messages: bool = False, capture_tokens: bool = False) -> SessionStats:
    stats = SessionStats(
        tool="claude_code",
        session_id=path.stem,
        file=str(path),
        mtime_iso=_iso_from_mtime(path),
    )

    # Track state for calculating execution times
    pending_tools: Dict[str, Tuple[str, dt.datetime, Optional[str]]] = {}  # tool_use_id -> (tool_name, start_time, working_dir)
    current_prompt_iterations = 0
    last_user_prompt_time: Optional[dt.datetime] = None

    for obj in _read_jsonl(path):
        # Extract session-level cwd if available
        if not stats.session_cwd and obj.get("cwd"):
            stats.session_cwd = obj["cwd"]
        timestamp = _parse_timestamp(obj.get("timestamp"))

        # Check if this is a user message
        if _is_user_message(obj):
            # Check if it's a tool result (not a new prompt)
            is_tool_result = False
            if isinstance(obj.get("message", {}).get("content"), list):
                for item in obj["message"]["content"]:
                    if isinstance(item, dict) and item.get("tool_use_id"):
                        is_tool_result = True
                        # Process tool result
                        tool_use_id = item["tool_use_id"]
                        if tool_use_id in pending_tools:
                            tool_name, start_time, working_dir = pending_tools[tool_use_id]
                            if timestamp and start_time:
                                exec_time = (timestamp - start_time).total_seconds()
                                # Only record reasonable execution times (< 60 seconds)
                                if exec_time < 60:
                                    if tool_name not in stats.tool_stats:
                                        stats.tool_stats[tool_name] = ToolStats()
                                    stats.tool_stats[tool_name].execution_times.append(exec_time)

                                    stats.trace_events.append(TraceEvent(
                                        timestamp=timestamp.isoformat(),
                                        event_type="tool_result",
                                        coding_agent=stats.tool,
                                        tool_name=tool_name,
                                        execution_time=exec_time,
                                        working_dir=working_dir,
                                        session_id=stats.session_id,
                                    ))
                            del pending_tools[tool_use_id]

            if not is_tool_result:
                # This is a new user prompt
                stats.prompts += 1
                if current_prompt_iterations > 0:
                    stats.iterations_per_prompt.append(current_prompt_iterations)
                current_prompt_iterations = 0
                last_user_prompt_time = timestamp

                if timestamp:
                    stats.trace_events.append(TraceEvent(
                        timestamp=timestamp.isoformat(),
                        event_type="user_prompt",
                        coding_agent=stats.tool,
                        session_id=stats.session_id,
                        message_text=_extract_user_text(obj) if capture_messages else None,
                    ))

        # Check if this is an assistant message
        elif _is_assistant_message(obj):
            stats.assistant_msgs += 1
            current_prompt_iterations += 1

            # Check for tool calls
            has_tool_calls = False
            if isinstance(obj.get("message", {}).get("content"), list):
                for item in obj["message"]["content"]:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        has_tool_calls = True
                        stats.tool_calls += 1
                        tool_name = item.get("name", "unknown")
                        tool_use_id = item.get("id", "")
                        working_dir = _extract_working_dir(item.get("input", {}))

                        if tool_name not in stats.tool_stats:
                            stats.tool_stats[tool_name] = ToolStats()
                        stats.tool_stats[tool_name].count += 1

                        if working_dir:
                            stats.working_dirs.append(working_dir)

                        if tool_use_id and timestamp:
                            pending_tools[tool_use_id] = (tool_name, timestamp, working_dir)

                        if timestamp:
                            stats.trace_events.append(TraceEvent(
                                timestamp=timestamp.isoformat(),
                                event_type="tool_call",
                                coding_agent=stats.tool,
                                tool_name=tool_name,
                                working_dir=working_dir,
                                session_id=stats.session_id,
                            ))

            # Extract token usage (Claude Code JSONL only, when --tokens requested)
            token_event_kwargs: Dict[str, Any] = {}
            if capture_tokens:
                usage = _extract_token_usage(obj)
                if usage:
                    token_event_kwargs = {
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "cache_creation_tokens": usage.get("cache_creation_input_tokens"),
                        "cache_read_tokens": usage.get("cache_read_input_tokens"),
                        "model": obj.get("message", {}).get("model"),
                    }

            if not has_tool_calls and last_user_prompt_time and timestamp:
                response_time = (timestamp - last_user_prompt_time).total_seconds()
                # Only record reasonable response times (< 600 seconds / 10 minutes)
                if response_time < 600:
                    stats.prompt_response_times.append(response_time)

                if timestamp:
                    stats.trace_events.append(TraceEvent(
                        timestamp=timestamp.isoformat(),
                        event_type="assistant_response",
                        coding_agent=stats.tool,
                        session_id=stats.session_id,
                        **token_event_kwargs,
                    ))
            elif has_tool_calls and capture_tokens and token_event_kwargs and timestamp:
                # Emit a token-tracking assistant_response even when tool calls are present
                stats.trace_events.append(TraceEvent(
                    timestamp=timestamp.isoformat(),
                    event_type="assistant_response",
                    coding_agent=stats.tool,
                    session_id=stats.session_id,
                    **token_event_kwargs,
                ))

    # Finalize last prompt's iteration count
    if current_prompt_iterations > 0:
        stats.iterations_per_prompt.append(current_prompt_iterations)

    return stats


def parse_codex_jsonl(path: Path) -> SessionStats:
    """Parser for Codex CLI logs with tool call tracking"""
    stats = SessionStats(
        tool="codex_cli",
        session_id=path.stem,
        file=str(path),
        mtime_iso=_iso_from_mtime(path),
    )

    # Track pending function calls: call_id -> (tool_name, start_time)
    pending_calls: Dict[str, Tuple[str, dt.datetime]] = {}

    for obj in _read_jsonl(path):
        # Extract session-level cwd/workdir if available
        if not stats.session_cwd:
            # Check direct fields
            stats.session_cwd = obj.get("cwd") or obj.get("workdir") or obj.get("working_dir")

            # Check payload.cwd (for session_meta type)
            if not stats.session_cwd and isinstance(obj.get("payload"), dict):
                stats.session_cwd = obj["payload"].get("cwd")

            # Check environment_context in content (simple regex extraction)
            if not stats.session_cwd and isinstance(obj.get("payload"), dict):
                payload = obj["payload"]
                if isinstance(payload.get("content"), list):
                    for item in payload["content"]:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            match = re.search(r'<cwd>([^<]+)</cwd>', item["text"])
                            if match:
                                stats.session_cwd = match.group(1)
                                break

        timestamp = _parse_timestamp(obj.get("timestamp"))

        stats.prompts += 1 if _is_user_message(obj) else 0
        stats.assistant_msgs += 1 if _is_assistant_message(obj) else 0

        # Detect Codex function_call format
        if isinstance(obj.get("payload"), dict):
            payload = obj["payload"]
            payload_type = payload.get("type")

            # Function call start
            if payload_type == "function_call":
                tool_name = payload.get("name", "unknown")
                call_id = payload.get("call_id", "")
                stats.tool_calls += 1

                if tool_name not in stats.tool_stats:
                    stats.tool_stats[tool_name] = ToolStats()
                stats.tool_stats[tool_name].count += 1

                if call_id and timestamp:
                    pending_calls[call_id] = (tool_name, timestamp)

                if timestamp:
                    stats.trace_events.append(TraceEvent(
                        timestamp=timestamp.isoformat(),
                        event_type="tool_call",
                        coding_agent=stats.tool,
                        tool_name=tool_name,
                        working_dir=stats.session_cwd,
                        session_id=stats.session_id,
                    ))

            # Function call output (result)
            elif payload_type == "function_call_output":
                call_id = payload.get("call_id", "")
                if call_id in pending_calls:
                    tool_name, start_time = pending_calls[call_id]
                    if timestamp and start_time:
                        exec_time = (timestamp - start_time).total_seconds()
                        # Only record reasonable execution times (< 60 seconds)
                        if exec_time < 60:
                            if tool_name not in stats.tool_stats:
                                stats.tool_stats[tool_name] = ToolStats()
                            stats.tool_stats[tool_name].execution_times.append(exec_time)

                            stats.trace_events.append(TraceEvent(
                                timestamp=timestamp.isoformat(),
                                event_type="tool_result",
                                coding_agent=stats.tool,
                                tool_name=tool_name,
                                execution_time=exec_time,
                                working_dir=stats.session_cwd,
                                session_id=stats.session_id,
                            ))
                    del pending_calls[call_id]

    return stats


def _flatten_gemini_messages(doc: Any) -> List[Dict[str, Any]]:
    """
    Gemini session files vary; attempt to find message-like objects.
    """
    msgs: List[Dict[str, Any]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            # Common patterns: {"messages":[...]} or {"chat":[...]} etc.
            for key in ("messages", "turns", "chat", "history"):
                if isinstance(x.get(key), list):
                    for it in x[key]:
                        if isinstance(it, dict):
                            msgs.append(it)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(doc)
    return msgs


def parse_gemini_json(path: Path) -> SessionStats:
    """Enhanced parser for Gemini CLI logs with tool call and timing analysis."""
    stats = SessionStats(
        tool="gemini_cli",
        session_id=path.stem,
        file=str(path),
        mtime_iso=_iso_from_mtime(path),
    )
    doc = _read_json(path)

    stats.session_cwd = doc.get("cwd") or doc.get("workdir") or doc.get("working_dir")

    msgs = _flatten_gemini_messages(doc)
    if not msgs:
        print(f"DEBUG: No messages flattened for {path}", file=sys.stderr)
        stats.tool_calls += _count_tool_calls_in_obj(doc) # Fallback, if whole doc is a tool call
        return stats

    pending_tools: Dict[str, Tuple[str, dt.datetime, Optional[str]]] = {}  # tool_call_id -> (tool_name, start_time, working_dir)
    current_prompt_iterations = 0
    last_user_prompt_time: Optional[dt.datetime] = None

    for i, obj in enumerate(msgs):
        raw_timestamp = obj.get("timestamp")
        timestamp = _parse_timestamp(raw_timestamp, path)
        message_type = _infer_message_type(obj)
        
        # Fallback to file mtime if message timestamp is missing/unparseable
        event_timestamp_str = timestamp.isoformat() if timestamp else stats.mtime_iso
        
        print(f"DEBUG: Processing message {i} in {path.name}: inferred_type={message_type}, raw_role='{obj.get('role')}', timestamp={raw_timestamp}", file=sys.stderr)

        if message_type == "user":
            stats.prompts += 1
            if current_prompt_iterations > 0:
                stats.iterations_per_prompt.append(current_prompt_iterations)
            current_prompt_iterations = 0
            last_user_prompt_time = timestamp
            print(f"DEBUG: Found user prompt in {path.name}", file=sys.stderr)

            stats.trace_events.append(TraceEvent(
                timestamp=event_timestamp_str,
                event_type="user_prompt",
                coding_agent=stats.tool,
                session_id=stats.session_id,
            ))

        elif message_type == "assistant_with_tools":
            stats.assistant_msgs += 1
            current_prompt_iterations += 1
            
            tool_calls = obj["tool_calls"]
            print(f"DEBUG: Found assistant with tool_calls in {path.name}", file=sys.stderr)
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    stats.tool_calls += 1
                    tool_name = tool_call.get("name", "unknown")
                    # Gemini doesn't always have a unique ID per call, so we'll have to improvise one
                    tool_call_id = f"{tool_name}-{event_timestamp_str}-{i}-{hash(json.dumps(tool_call, sort_keys=True))}" 
                    working_dir = _extract_working_dir(tool_call.get("arguments", {}))

                    if tool_name not in stats.tool_stats:
                        stats.tool_stats[tool_name] = ToolStats()
                    stats.tool_stats[tool_name].count += 1

                    if working_dir:
                        stats.working_dirs.append(working_dir)

                    if timestamp: # Only add to pending if we have a valid start timestamp
                        pending_tools[tool_call_id] = (tool_name, timestamp, working_dir)
                    else:
                        print(f"DEBUG: Skipping pending tool tracking for {tool_name} due to missing timestamp in {path.name}", file=sys.stderr)

                    stats.trace_events.append(TraceEvent(
                        timestamp=event_timestamp_str,
                        event_type="tool_call",
                        coding_agent=stats.tool,
                        tool_name=tool_name,
                        working_dir=working_dir,
                        session_id=stats.session_id,
                    ))

        elif message_type == "tool_result":
            tool_name = obj.get("name", "unknown")
            print(f"DEBUG: Found tool result for {tool_name} in {path.name}", file=sys.stderr)

            matching_call_id = None
            for call_id, (name, _, _) in reversed(list(pending_tools.items())):
                if name == tool_name:
                    matching_call_id = call_id
                    break
            
            if matching_call_id and matching_call_id in pending_tools:
                _, start_time, working_dir = pending_tools[matching_call_id]
                if timestamp and start_time:
                    exec_time = (timestamp - start_time).total_seconds()
                    if exec_time < 60:
                        if tool_name not in stats.tool_stats:
                            stats.tool_stats[tool_name] = ToolStats()
                        stats.tool_stats[tool_name].execution_times.append(exec_time)
                        print(f"DEBUG: Calculated tool execution time {exec_time:.2f}s for {tool_name} in {path.name}", file=sys.stderr)
                        stats.trace_events.append(TraceEvent(
                            timestamp=event_timestamp_str,
                            event_type="tool_result",
                            coding_agent=stats.tool,
                            tool_name=tool_name,
                            execution_time=exec_time,
                            working_dir=working_dir,
                            session_id=stats.session_id,
                        ))
                    else:
                        print(f"DEBUG: Tool execution time {exec_time:.2f}s for {tool_name} exceeded 60s, not recorded in stats in {path.name}", file=sys.stderr)
                else:
                    print(f"DEBUG: Missing timestamp or start_time for tool result {tool_name} in {path.name}. Cannot calculate execution time.", file=sys.stderr)
                del pending_tools[matching_call_id]
            else:
                print(f"DEBUG: No matching pending tool call found for result {tool_name} in {path.name}", file=sys.stderr)
                stats.trace_events.append(TraceEvent(
                    timestamp=event_timestamp_str,
                    event_type="tool_result",
                    coding_agent=stats.tool,
                    tool_name=tool_name,
                    working_dir=stats.session_cwd, # Fallback to session cwd
                    session_id=stats.session_id,
                ))

        elif message_type == "assistant_response":
            stats.assistant_msgs += 1
            current_prompt_iterations += 1
            print(f"DEBUG: Found regular assistant response in {path.name}", file=sys.stderr)
            if last_user_prompt_time and timestamp:
                response_time = (timestamp - last_user_prompt_time).total_seconds()
                if response_time < 600:
                    stats.prompt_response_times.append(response_time)
                    print(f"DEBUG: Calculated prompt response time {response_time:.2f}s in {path.name}", file=sys.stderr)

            stats.trace_events.append(TraceEvent(
                timestamp=event_timestamp_str,
                event_type="assistant_response",
                coding_agent=stats.tool,
                session_id=stats.session_id,
            ))
        else: # Unclassified messages
            print(f"DEBUG: Unclassified message in {path.name}: {obj}", file=sys.stderr)


    if current_prompt_iterations > 0:
        stats.iterations_per_prompt.append(current_prompt_iterations)

    return stats


# -----------------------------
# Discovery
# -----------------------------

def discover_default_paths() -> List[Tuple[str, List[Path]]]:
    home = str(Path.home())

    claude_glob = os.path.join(home, ".claude", "projects", "**", "*.jsonl")
    codex_home = os.environ.get("CODEX_HOME", os.path.join(home, ".codex"))
    codex_glob = os.path.join(codex_home, "sessions", "**", "rollout-*.jsonl")
    gemini_glob = os.path.join(home, ".gemini", "tmp", "**", "session-*.json")

    claude = [Path(p) for p in glob.glob(claude_glob, recursive=True)]
    codex = [Path(p) for p in glob.glob(codex_glob, recursive=True)]
    gemini = [Path(p) for p in glob.glob(gemini_glob, recursive=True)]

    return [
        ("claude_code", claude),
        ("codex_cli", codex),
        ("gemini_cli", gemini),
    ]


def discover_from_roots(roots: List[str]) -> List[Path]:
    out: List[Path] = []
    for r in roots:
        out.extend([Path(p) for p in glob.glob(r, recursive=True)])
    return out


def classify_and_parse(path: Path, capture_messages: bool = False, capture_tokens: bool = False) -> Optional[SessionStats]:
    p = str(path)
    if p.endswith(".jsonl") and ("/.claude/projects/" in p.replace("\\", "/") or "\\.claude\\projects\\" in p):
        return parse_claude_jsonl(path, capture_messages=capture_messages, capture_tokens=capture_tokens)
    if p.endswith(".jsonl") and ("/.codex/" in p.replace("\\", "/") or "\\.codex\\" in p) and "rollout-" in path.name:
        return parse_codex_jsonl(path)
    if p.endswith(".json") and ("/.gemini/tmp/" in p.replace("\\", "/") or "\\.gemini\\tmp\\" in p) and path.name.startswith("session-"):
        return parse_gemini_json(path)

    # Fallback by extension
    if p.endswith(".jsonl"):
        # Try both jsonl parsers; choose the one that yields non-zero totals
        a = parse_claude_jsonl(path, capture_messages=capture_messages, capture_tokens=capture_tokens)
        b = parse_codex_jsonl(path)
        return a if (a.prompts + a.assistant_msgs + a.tool_calls) >= (b.prompts + b.assistant_msgs + b.tool_calls) else b
    if p.endswith(".json"):
        return parse_gemini_json(path)

    return None


# -----------------------------
# Main
# -----------------------------

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=None, help="Student identifier (email/ID). Used only to derive a pseudonym.")
    ap.add_argument("--csv", default="ai_usage_trace.csv", help="Output trace CSV path (time-based events)")
    ap.add_argument("--summary", default="summary.txt", help="Output summary text path")
    ap.add_argument("--json", help="Optional: Output detailed JSON for debugging")
    ap.add_argument("--filter", help="Optional: Regex pattern to filter by working directory/file path")
    ap.add_argument("--messages", action="store_true", help="Optional: Include user message text in CSV output")
    ap.add_argument("--tokens", action="store_true",
                    help="Optional: Capture per-event token counts from Claude Code JSONL files. "
                         "Adds input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens, model columns.")
    ap.add_argument(
        "--roots",
        nargs="*",
        default=[],
        help="Optional glob roots to search instead of defaults (use quotes). Example: '~/.claude/projects/**/*.jsonl'",
    )
    args = ap.parse_args(argv)

    student_hash = _pseudonym(args.student) if args.student else None

    # Compile filter regex if provided
    filter_pattern = None
    if args.filter:
        try:
            filter_pattern = re.compile(args.filter)
        except re.error as e:
            print(f"Error: Invalid regex pattern '{args.filter}': {e}", file=sys.stderr)
            return 1

    if args.roots:
        paths = discover_from_roots(args.roots)
    else:
        discovered = discover_default_paths()
        paths = [p for _, ps in discovered for p in ps]

    # Deduplicate + stable ordering
    uniq: Dict[str, Path] = {}
    for p in paths:
        uniq[str(p)] = p
    paths = sorted(uniq.values(), key=lambda x: str(x))

    sessions: List[SessionStats] = []
    for p in paths:
        try:
            s = classify_and_parse(p, capture_messages=args.messages, capture_tokens=args.tokens)
            if s is not None:
                sessions.append(s)
        except Exception as e:
            # If a file is corrupt/unreadable, skip rather than failing the whole submission.
            if args.json:  # Only show errors in debug mode
                print(f"Warning: Skipping {p}: {e}", file=sys.stderr)
            continue

    # Apply filter if specified (session-level filtering)
    if filter_pattern:
        filtered_sessions = []
        for s in sessions:
            # Check if session matches filter based on:
            # 1. Session-level cwd
            # 2. Any tool-level working directory
            session_matches = False

            if s.session_cwd and filter_pattern.search(s.session_cwd):
                session_matches = True
            elif any(filter_pattern.search(wd) for wd in s.working_dirs if wd):
                session_matches = True

            # If session matches, include the ENTIRE session with all its stats
            if session_matches:
                filtered_sessions.append(s)

        sessions = filtered_sessions

    # Aggregate statistics across all sessions
    total_prompts = sum(s.prompts for s in sessions)
    total_assistant_msgs = sum(s.assistant_msgs for s in sessions)
    total_tool_calls = sum(s.tool_calls for s in sessions)

    # Aggregate tool statistics
    all_tool_stats: Dict[str, ToolStats] = {}
    for s in sessions:
        for tool_name, tool_stat in s.tool_stats.items():
            if tool_name not in all_tool_stats:
                all_tool_stats[tool_name] = ToolStats()
            all_tool_stats[tool_name].count += tool_stat.count
            all_tool_stats[tool_name].execution_times.extend(tool_stat.execution_times)

    # Calculate overall average tool execution time
    all_exec_times = []
    for tool_stat in all_tool_stats.values():
        all_exec_times.extend(tool_stat.execution_times)

    avg_tool_time = sum(all_exec_times) / len(all_exec_times) if all_exec_times else 0.0

    # Calculate average iterations per prompt
    all_iterations = []
    for s in sessions:
        all_iterations.extend(s.iterations_per_prompt)
    avg_iterations = sum(all_iterations) / len(all_iterations) if all_iterations else 0.0

    # Calculate average prompt response time
    all_prompt_times = []
    for s in sessions:
        all_prompt_times.extend(s.prompt_response_times)
    avg_prompt_time = sum(all_prompt_times) / len(all_prompt_times) if all_prompt_times else 0.0

    # Print Ruby-style summary (and save to file)
    summary_buf = io.StringIO()

    def emit(line: str = "") -> None:
        print(line)
        summary_buf.write(line + "\n")

    emit(f"=== Claude Code Trace Analysis ({len(sessions)} files) ===")
    if filter_pattern:
        emit(f"🔍 Filter: {args.filter}")
    emit()
    emit("📊 AGGREGATE STATISTICS:")
    emit(f"  • Total files analyzed: {len(sessions)}")
    emit(f"  • Total tools called: {total_tool_calls}")

    if all_exec_times:
        emit(f"  • Tool use: {avg_tool_time:.3f} sec avg, {total_tool_calls} calls")

        # Sort tools by name for consistent output
        for tool_name in sorted(all_tool_stats.keys()):
            tool_stat = all_tool_stats[tool_name]
            avg_time = tool_stat.avg_time()
            emit(f"    + {tool_name} tool: {avg_time:.1f} sec avg, {tool_stat.count} calls")
    else:
        emit(f"  • Tool use: Unable to calculate (insufficient timing data)")

    emit(f"  • Total user prompts: {total_prompts}")

    if all_prompt_times:
        emit(f"  • User prompts: {avg_prompt_time:.3f} sec avg, {total_prompts} calls")

    if all_iterations:
        emit(f"  • Average Claude iterations per prompt: {avg_iterations:.2f}")

    # Per-agent breakdown
    agent_stats: Dict[str, Dict[str, Any]] = {}
    for s in sessions:
        agent = s.tool
        if agent not in agent_stats:
            agent_stats[agent] = {
                "sessions": 0,
                "prompts": 0,
                "tool_calls": 0,
                "tool_stats": {},
            }
        agent_stats[agent]["sessions"] += 1
        agent_stats[agent]["prompts"] += s.prompts
        agent_stats[agent]["tool_calls"] += s.tool_calls

        # Aggregate tool stats per agent
        for tool_name, tool_stat in s.tool_stats.items():
            if tool_name not in agent_stats[agent]["tool_stats"]:
                agent_stats[agent]["tool_stats"][tool_name] = ToolStats()
            agent_stats[agent]["tool_stats"][tool_name].count += tool_stat.count
            agent_stats[agent]["tool_stats"][tool_name].execution_times.extend(tool_stat.execution_times)

    # Show per-agent breakdown if:
    # - Filter is active (to show what matched)
    # - Multiple agents present
    # - Single non-Claude agent
    show_agent_breakdown = (
        filter_pattern is not None or
        len(agent_stats) > 1 or
        (len(agent_stats) == 1 and list(agent_stats.keys())[0] != "claude_code")
    )

    if show_agent_breakdown and agent_stats:
        emit()
        emit("📊 BY CODING AGENT:")
        for agent in sorted(agent_stats.keys()):
            stats = agent_stats[agent]
            emit(f"  • {agent}: {stats['sessions']} sessions, {stats['prompts']} prompts, {stats['tool_calls']} tool calls")

            # Show top 3 most used tools for this agent
            if stats["tool_stats"]:
                sorted_tools = sorted(stats["tool_stats"].items(), key=lambda x: x[1].count, reverse=True)[:3]
                for tool_name, tool_stat in sorted_tools:
                    avg_time = tool_stat.avg_time()
                    emit(f"    + {tool_name}: {avg_time:.1f} sec avg, {tool_stat.count} calls")

    # Collect unique working directories
    all_working_dirs = set()
    for s in sessions:
        all_working_dirs.update(s.working_dirs)

    if all_working_dirs:
        emit()
        emit(f"📂 WORKING DIRECTORIES ({len(all_working_dirs)} unique):")
        for wd in sorted(all_working_dirs)[:10]:  # Show top 10
            emit(f"  • {wd}")
        if len(all_working_dirs) > 10:
            emit(f"  ... and {len(all_working_dirs) - 10} more")

    with open(args.summary, "w", encoding="utf-8") as f:
        f.write(summary_buf.getvalue())
    print()
    print(f"✅ Wrote summary to: {args.summary}")

    # Write time-based CSV trace
    all_events: List[TraceEvent] = []
    for s in sessions:
        all_events.extend(s.trace_events)

    # Sort events by timestamp
    all_events.sort(key=lambda e: e.timestamp)

    TOKEN_FIELDS = ["input_tokens", "output_tokens", "cache_creation_tokens", "cache_read_tokens", "model"]

    with open(args.csv, "w", encoding="utf-8", newline="") as f:
        if all_events:
            fieldnames = list(all_events[0].to_row().keys())
            if not args.messages:
                fieldnames = [f for f in fieldnames if f != "message_text"]
            if not args.tokens:
                fieldnames = [f for f in fieldnames if f not in TOKEN_FIELDS]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for event in all_events:
                row = event.to_row()
                if not args.messages:
                    row.pop("message_text", None)
                if not args.tokens:
                    for tf in TOKEN_FIELDS:
                        row.pop(tf, None)
                w.writerow(row)

    print()
    print(f"✅ Wrote time-based trace to: {args.csv}")

    # Token summary (when --tokens was used)
    if args.tokens and all_events:
        total_input = sum(e.input_tokens or 0 for e in all_events)
        total_output = sum(e.output_tokens or 0 for e in all_events)
        total_cache_read = sum(e.cache_read_tokens or 0 for e in all_events)
        total_cache_creation = sum(e.cache_creation_tokens or 0 for e in all_events)
        cache_total = total_input + total_cache_read + total_cache_creation
        hit_rate = (total_cache_read / cache_total * 100) if cache_total > 0 else 0.0
        print(f"   Token summary: {total_input:,} input · {total_output:,} output · "
              f"cache hit rate {hit_rate:.1f}%")

    # Optionally write JSON for debugging
    if args.json:
        debug_data = {
            "student_pseudonym": student_hash,
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "summary": {
                "n_sessions": len(sessions),
                "prompts": total_prompts,
                "assistant_msgs": total_assistant_msgs,
                "tool_calls": total_tool_calls,
                "avg_iterations_per_prompt": avg_iterations,
                "avg_prompt_response_time": avg_prompt_time,
            },
            "tool_stats": {
                name: {
                    "count": stat.count,
                    "avg_time": stat.avg_time(),
                    "execution_times": stat.execution_times,
                }
                for name, stat in all_tool_stats.items()
            },
            "working_dirs": sorted(all_working_dirs),
            "sessions": [s.to_row() for s in sessions],
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(debug_data, f, indent=2, sort_keys=True)
        print(f"✅ Wrote debug JSON to: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
"""
Cursor JSONL log parser.

Reads Cursor agent-transcript logs (one JSON object per line) and yields
normalized rows compatible with the ai_usage_trace.csv format the dashboard
expects.

Actual log format (each line is one of):
  {"role": "user",      "message": {"content": [{"type": "text", "text": "..."}]}}
  {"role": "assistant", "message": {"content": [{"type": "tool_use", "name": "Shell",
                                                  "input": {"working_directory": "...", ...}}]}}
  {"type": "turn_ended", "status": "success"}

Fields that do NOT appear in this format and will always be empty:
  timestamp, execution_time, input_tokens, output_tokens,
  cache_creation_tokens, cache_read_tokens, model

session_id is derived from the filename stem (one file = one session).

Default log locations on macOS:
  ~/Library/Application Support/Cursor/User/workspaceStorage/*/agent-transcripts/*.jsonl
  ~/Library/Application Support/Cursor/logs/*.jsonl
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

CSV_HEADERS: List[str] = [
    "timestamp",
    "event_type",
    "coding_agent",
    "tool_name",
    "execution_time",
    "working_dir",
    "repo_root",
    "session_id",
    "message_text",
    "input_tokens",
    "output_tokens",
    "cache_creation_tokens",
    "cache_read_tokens",
    "model",
]

# Map snake_case Cursor tool names to dashboard canonical names.
# PascalCase names (used in newer Cursor versions) pass through unchanged.
_TOOL_ALIASES: Dict[str, str] = {
    "read_file": "Read",
    "edit_file": "Edit",
    "create_file": "Write",
    "run_terminal_cmd": "Shell",
    "grep_search": "Grep",
    "file_search": "Glob",
    "list_dir": "Glob",
    "codebase_search": "codebase_search",
    "web_search": "WebSearch",
    "delete_file": "Shell",
}

_DEFAULT_ROOTS_MAC: List[Path] = [
    Path.home()
    / "Library"
    / "Application Support"
    / "Cursor"
    / "User"
    / "workspaceStorage",
    Path.home() / "Library" / "Application Support" / "Cursor" / "logs",
]


def _get_repo_root(working_dir: str) -> str:
    """Return the git repo root for working_dir, or '' if not in a repo or path missing."""
    if not working_dir:
        return ""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, NotADirectoryError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


def _canonical_tool(raw_name: str) -> str:
    return _TOOL_ALIASES.get(raw_name, raw_name)


def _csv_field(value: str) -> str:
    """Wrap value in double-quotes if it contains commas, newlines, or quotes."""
    if "," in value or "\n" in value or '"' in value:
        return '"' + value.replace('"', '""') + '"'
    return value


def _parse_line(line: str) -> Optional[Dict]:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
        if not isinstance(obj, dict):
            return None
        return obj
    except json.JSONDecodeError:
        return None


def _make_row(
    event_type: str,
    session_id: str,
    tool_name: str = "",
    working_dir: str = "",
    repo_root: str = "",
    message_text: str = "",
) -> Dict[str, str]:
    return {
        "timestamp": "",
        "event_type": event_type,
        "coding_agent": "cursor",
        "tool_name": tool_name,
        "execution_time": "",
        "working_dir": working_dir,
        "repo_root": repo_root,
        "session_id": session_id,
        "message_text": message_text,
        "input_tokens": "",
        "output_tokens": "",
        "cache_creation_tokens": "",
        "cache_read_tokens": "",
        "model": "",
    }


def parse_cursor_jsonl(
    path: "os.PathLike[str]",
    *,
    verbose: bool = False,
) -> Iterator[Dict[str, str]]:
    """
    Yield one CSV row dict per relevant event in a Cursor JSONL transcript file.

    Skips blank lines, malformed JSON, and turn_ended lines silently.
    Each assistant message with tool_use blocks yields one row per tool call.
    Never raises on bad input.
    """
    path = Path(path)
    session_id = path.stem
    bad_lines = 0

    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            obj = _parse_line(raw_line)
            if obj is None:
                if raw_line.strip():
                    bad_lines += 1
                    if verbose:
                        print(
                            f"[cursor] skipping bad line {lineno} in {path.name}",
                            file=sys.stderr,
                        )
                continue

            role = obj.get("role")

            if role == "user":
                message = obj.get("message") or {}
                content = message.get("content") or []
                texts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                message_text = " ".join(t for t in texts if t).strip()
                yield _make_row("user_prompt", session_id, message_text=message_text)

            elif role == "assistant":
                message = obj.get("message") or {}
                content = message.get("content") or []
                if not isinstance(content, list):
                    content = []

                tool_uses = [
                    block
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "tool_use"
                ]

                if tool_uses:
                    for block in tool_uses:
                        tool_input = block.get("input") or {}
                        working_dir = str(tool_input.get("working_directory", "")).strip()
                        tool_name = _canonical_tool(str(block.get("name", "")).strip())
                        yield _make_row(
                            "tool_call",
                            session_id,
                            tool_name=tool_name,
                            working_dir=working_dir,
                            repo_root=_get_repo_root(working_dir),
                        )
                else:
                    yield _make_row("assistant_response", session_id)

            # role is None / type == "turn_ended" / anything else → skip

    if bad_lines and verbose:
        print(
            f"[cursor] {bad_lines} malformed line(s) skipped in {path.name}",
            file=sys.stderr,
        )


def find_cursor_logs(roots: Optional[List[Path]] = None) -> List[Path]:
    """
    Return all *.jsonl files under the given roots (default: macOS Cursor paths).

    Silently skips roots that do not exist.
    """
    search_roots = roots if roots is not None else _DEFAULT_ROOTS_MAC
    found: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        found.extend(root.rglob("*.jsonl"))
    return sorted(found)


def rows_to_csv(rows: List[Dict[str, str]]) -> str:
    """Serialize a list of row dicts to a RFC-4180-compatible CSV string."""
    lines = [",".join(CSV_HEADERS)]
    for row in rows:
        lines.append(",".join(_csv_field(row.get(h, "")) for h in CSV_HEADERS))
    return "\n".join(lines) + "\n"

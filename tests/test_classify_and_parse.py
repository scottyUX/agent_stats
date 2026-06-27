import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_usage_stats import classify_and_parse, _iso_from_mtime


def write_agent_transcript(tmp_path: Path, session_id: str = "abc", text: str = "hello") -> Path:
    p = tmp_path / ".cursor" / "projects" / "foo" / "agent-transcripts" / session_id / f"{session_id}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    line = {"role": "user", "message": {"content": [{"type": "text", "text": text}]}}
    p.write_text(json.dumps(line) + "\n", encoding="utf-8")
    return p


def write_legacy_cursor_path(tmp_path: Path, text: str = "hello") -> Path:
    p = tmp_path / "Library" / "Application Support" / "Cursor" / "User" / "workspaceStorage" / "hash" / "chat.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    line = {"role": "user", "message": {"content": [{"type": "text", "text": text}]}}
    p.write_text(json.dumps(line) + "\n", encoding="utf-8")
    return p


def write_claude_path(tmp_path: Path, text: str = "hello") -> Path:
    p = tmp_path / ".claude" / "projects" / "foo" / "session.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    line = {"type": "user", "message": {"role": "user", "content": text}}
    p.write_text(json.dumps(line) + "\n", encoding="utf-8")
    return p


def test_agent_transcript_routes_to_cursor(tmp_path):
    p = write_agent_transcript(tmp_path)
    stats = classify_and_parse(p)
    assert stats.tool == "cursor"
    assert stats.prompts == 1
    assert stats.trace_events[0].coding_agent == "cursor"


def test_agent_transcript_message_text_with_flag(tmp_path):
    p = write_agent_transcript(tmp_path, text="Find the auth bug")
    stats = classify_and_parse(p, capture_messages=True)
    assert stats.trace_events[0].message_text == "Find the auth bug"


def test_legacy_cursor_path_still_routes(tmp_path):
    p = write_legacy_cursor_path(tmp_path)
    stats = classify_and_parse(p)
    assert stats.tool == "cursor"
    assert stats.prompts == 1


def test_claude_path_not_misclassified(tmp_path):
    p = write_claude_path(tmp_path, text="claude prompt")
    stats = classify_and_parse(p, capture_messages=True)
    assert stats.tool == "claude_code"
    assert stats.trace_events[0].coding_agent == "claude_code"
    assert stats.trace_events[0].message_text == "claude prompt"


def test_mtime_timestamp_fallback(tmp_path):
    p = write_agent_transcript(tmp_path)
    expected = _iso_from_mtime(p)
    stats = classify_and_parse(p)
    assert stats.trace_events[0].timestamp == expected

import csv
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "research" / "fixtures" / "cursor" / "cursor_sample.jsonl"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parsers.cursor import (
    CSV_HEADERS,
    find_cursor_logs,
    parse_cursor_jsonl,
    rows_to_csv,
)


def write_jsonl(tmp_path: Path, lines: list) -> Path:
    p = tmp_path / "test-session.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for item in lines:
            if isinstance(item, str):
                fh.write(item + "\n")
            else:
                fh.write(json.dumps(item) + "\n")
    return p


def collect(path: Path) -> list[dict[str, str]]:
    return list(parse_cursor_jsonl(path))


# ---------------------------------------------------------------------------
# Empty / blank edge cases
# ---------------------------------------------------------------------------

def test_empty_file_produces_no_rows(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert collect(p) == []


def test_empty_file_csv_has_only_headers(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    csv_text = rows_to_csv(collect(p))
    lines = csv_text.strip().splitlines()
    assert len(lines) == 1
    assert lines[0] == ",".join(CSV_HEADERS)


def test_blank_lines_ignored(tmp_path):
    user_line = {"role": "user", "message": {"content": [{"type": "text", "text": "hi"}]}}
    p = tmp_path / "blanks.jsonl"
    p.write_text("\n\n" + json.dumps(user_line) + "\n\n\n", encoding="utf-8")
    rows = collect(p)
    assert len(rows) == 1
    assert rows[0]["event_type"] == "user_prompt"


# ---------------------------------------------------------------------------
# Bad lines
# ---------------------------------------------------------------------------

def test_bad_line_skipped_good_lines_parsed(tmp_path):
    lines = [
        {"role": "user", "message": {"content": [{"type": "text", "text": "Hello"}]}},
        "THIS IS NOT JSON",
        {"role": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read", "input": {}}]}},
        {"role": "assistant", "message": {"content": [{"type": "text", "text": "Done."}]}},
    ]
    p = write_jsonl(tmp_path, lines)
    rows = collect(p)
    event_types = [r["event_type"] for r in rows]
    assert event_types == ["user_prompt", "tool_call", "assistant_response"]


def test_all_bad_lines_yields_no_rows(tmp_path):
    p = write_jsonl(tmp_path, ["not json", "also not json", "!!!"])
    assert collect(p) == []


# ---------------------------------------------------------------------------
# Event type routing
# ---------------------------------------------------------------------------

def test_user_role_yields_user_prompt(tmp_path):
    lines = [{"role": "user", "message": {"content": [{"type": "text", "text": "go"}]}}]
    rows = collect(write_jsonl(tmp_path, lines))
    assert rows[0]["event_type"] == "user_prompt"


def test_assistant_with_tool_use_yields_tool_call(tmp_path):
    lines = [
        {
            "role": "assistant",
            "message": {"content": [{"type": "tool_use", "name": "Read", "input": {}}]},
        }
    ]
    rows = collect(write_jsonl(tmp_path, lines))
    assert rows[0]["event_type"] == "tool_call"


def test_assistant_without_tool_use_yields_assistant_response(tmp_path):
    lines = [
        {
            "role": "assistant",
            "message": {"content": [{"type": "text", "text": "Done."}]},
        }
    ]
    rows = collect(write_jsonl(tmp_path, lines))
    assert rows[0]["event_type"] == "assistant_response"


def test_turn_ended_line_skipped(tmp_path):
    lines = [
        {"role": "user", "message": {"content": [{"type": "text", "text": "go"}]}},
        {"type": "turn_ended", "status": "success"},
    ]
    rows = collect(write_jsonl(tmp_path, lines))
    assert len(rows) == 1
    assert rows[0]["event_type"] == "user_prompt"


def test_mixed_assistant_message_yields_tool_calls_only(tmp_path):
    """Assistant message with both text and tool_use → only tool_call rows, no assistant_response."""
    lines = [
        {
            "role": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Let me read that."},
                    {"type": "tool_use", "name": "Read", "input": {"path": "file.ts"}},
                ]
            },
        }
    ]
    rows = collect(write_jsonl(tmp_path, lines))
    assert len(rows) == 1
    assert rows[0]["event_type"] == "tool_call"
    assert rows[0]["tool_name"] == "Read"


def test_multiple_tool_uses_in_one_message_yield_multiple_rows(tmp_path):
    lines = [
        {
            "role": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Read", "input": {}},
                    {"type": "tool_use", "name": "Grep", "input": {}},
                ]
            },
        }
    ]
    rows = collect(write_jsonl(tmp_path, lines))
    assert len(rows) == 2
    assert rows[0]["tool_name"] == "Read"
    assert rows[1]["tool_name"] == "Grep"


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def test_message_text_extracted_from_user_content(tmp_path):
    msg = "Please fix the login redirect"
    lines = [{"role": "user", "message": {"content": [{"type": "text", "text": msg}]}}]
    rows = collect(write_jsonl(tmp_path, lines))
    assert rows[0]["message_text"] == msg


def test_message_text_joins_multiple_text_blocks(tmp_path):
    lines = [
        {
            "role": "user",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"},
                ]
            },
        }
    ]
    rows = collect(write_jsonl(tmp_path, lines))
    assert rows[0]["message_text"] == "Hello World"


def test_working_dir_extracted_from_tool_input(tmp_path):
    lines = [
        {
            "role": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Shell",
                        "input": {"command": "npm test", "working_directory": "/home/user/proj"},
                    }
                ]
            },
        }
    ]
    rows = collect(write_jsonl(tmp_path, lines))
    assert rows[0]["working_dir"] == "/home/user/proj"


def test_working_dir_empty_when_not_in_input(tmp_path):
    lines = [
        {
            "role": "assistant",
            "message": {"content": [{"type": "tool_use", "name": "Read", "input": {"path": "x.ts"}}]},
        }
    ]
    rows = collect(write_jsonl(tmp_path, lines))
    assert rows[0]["working_dir"] == ""


def test_session_id_derived_from_filename_stem(tmp_path):
    p = tmp_path / "my-session-abc.jsonl"
    p.write_text(
        json.dumps({"role": "user", "message": {"content": [{"type": "text", "text": "hi"}]}}) + "\n",
        encoding="utf-8",
    )
    rows = list(parse_cursor_jsonl(p))
    assert rows[0]["session_id"] == "my-session-abc"


def test_coding_agent_is_always_cursor(tmp_path):
    lines = [{"role": "user", "message": {"content": [{"type": "text", "text": "hi"}]}}]
    rows = collect(write_jsonl(tmp_path, lines))
    assert all(r["coding_agent"] == "cursor" for r in rows)


def test_timestamp_always_empty(tmp_path):
    lines = [{"role": "user", "message": {"content": [{"type": "text", "text": "hi"}]}}]
    rows = collect(write_jsonl(tmp_path, lines))
    assert rows[0]["timestamp"] == ""


def test_token_fields_always_empty(tmp_path):
    lines = [
        {"role": "assistant", "message": {"content": [{"type": "text", "text": "Done."}]}}
    ]
    rows = collect(write_jsonl(tmp_path, lines))
    r = rows[0]
    assert r["input_tokens"] == ""
    assert r["output_tokens"] == ""
    assert r["cache_creation_tokens"] == ""
    assert r["cache_read_tokens"] == ""


def test_model_always_empty(tmp_path):
    lines = [{"role": "user", "message": {"content": [{"type": "text", "text": "hi"}]}}]
    rows = collect(write_jsonl(tmp_path, lines))
    assert rows[0]["model"] == ""


# ---------------------------------------------------------------------------
# Tool name canonicalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cursor_name,expected", [
    ("read_file", "Read"),
    ("edit_file", "Edit"),
    ("create_file", "Write"),
    ("run_terminal_cmd", "Shell"),
    ("grep_search", "Grep"),
    ("codebase_search", "codebase_search"),
    ("list_dir", "Glob"),
    ("file_search", "Glob"),
    ("web_search", "WebSearch"),
    ("custom_unknown_tool", "custom_unknown_tool"),
    # PascalCase names from newer Cursor versions pass through unchanged
    ("Read", "Read"),
    ("Shell", "Shell"),
    ("StrReplace", "StrReplace"),
])
def test_tool_name_canonicalized(tmp_path, cursor_name, expected):
    lines = [
        {
            "role": "assistant",
            "message": {"content": [{"type": "tool_use", "name": cursor_name, "input": {}}]},
        }
    ]
    rows = collect(write_jsonl(tmp_path, lines))
    assert rows[0]["tool_name"] == expected


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def test_csv_headers_match_contract():
    csv_text = rows_to_csv([])
    reader = csv.DictReader(io.StringIO(csv_text))
    assert reader.fieldnames == CSV_HEADERS


def test_csv_row_has_all_columns(tmp_path):
    lines = [
        {"role": "user", "message": {"content": [{"type": "text", "text": "Fix it"}]}},
        {"role": "assistant", "message": {"content": [{"type": "tool_use", "name": "Edit", "input": {}}]}},
    ]
    rows = collect(write_jsonl(tmp_path, lines))
    csv_text = rows_to_csv(rows)
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        assert set(row.keys()) == set(CSV_HEADERS)


def test_repo_root_in_headers_after_working_dir():
    assert "repo_root" in CSV_HEADERS
    assert CSV_HEADERS.index("repo_root") == CSV_HEADERS.index("working_dir") + 1


# ---------------------------------------------------------------------------
# repo_root population
# ---------------------------------------------------------------------------

def test_repo_root_populated_for_tool_call_in_git_repo(tmp_path):
    lines = [
        {
            "role": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Shell",
                        "input": {"command": "npm test", "working_directory": "/home/user/project"},
                    }
                ]
            },
        }
    ]
    with patch("parsers.cursor._get_repo_root", return_value="/home/user/project"):
        rows = collect(write_jsonl(tmp_path, lines))
    assert rows[0]["repo_root"] == "/home/user/project"


def test_repo_root_empty_when_working_dir_not_a_git_repo(tmp_path):
    lines = [
        {
            "role": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Shell", "input": {"working_directory": "/tmp/no-git"}},
                ]
            },
        }
    ]
    with patch("parsers.cursor._get_repo_root", return_value=""):
        rows = collect(write_jsonl(tmp_path, lines))
    assert rows[0]["repo_root"] == ""


def test_repo_root_empty_when_no_working_dir(tmp_path):
    lines = [
        {
            "role": "assistant",
            "message": {"content": [{"type": "tool_use", "name": "Read", "input": {"path": "x.ts"}}]},
        }
    ]
    rows = collect(write_jsonl(tmp_path, lines))
    assert rows[0]["working_dir"] == ""
    assert rows[0]["repo_root"] == ""


def test_repo_root_maps_subdirectory_to_repo_root(tmp_path):
    lines = [
        {
            "role": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Shell",
                        "input": {"working_directory": "/home/user/project/src"},
                    }
                ]
            },
        }
    ]
    with patch("parsers.cursor._get_repo_root", return_value="/home/user/project"):
        rows = collect(write_jsonl(tmp_path, lines))
    assert rows[0]["working_dir"] == "/home/user/project/src"
    assert rows[0]["repo_root"] == "/home/user/project"


def test_repo_root_always_empty_for_user_prompt(tmp_path):
    lines = [{"role": "user", "message": {"content": [{"type": "text", "text": "go"}]}}]
    rows = collect(write_jsonl(tmp_path, lines))
    assert rows[0]["repo_root"] == ""


# ---------------------------------------------------------------------------
# find_cursor_logs
# ---------------------------------------------------------------------------

def test_find_cursor_logs_returns_jsonl_files(tmp_path):
    (tmp_path / "session-a.jsonl").write_text("{}\n")
    (tmp_path / "session-b.jsonl").write_text("{}\n")
    (tmp_path / "ignore.txt").write_text("x")
    found = find_cursor_logs([tmp_path])
    names = [p.name for p in found]
    assert "session-a.jsonl" in names
    assert "session-b.jsonl" in names
    assert "ignore.txt" not in names


def test_find_cursor_logs_skips_missing_roots(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert find_cursor_logs([missing]) == []


def test_csv_quotes_message_text_with_comma(tmp_path):
    msg = "Fix auth, please"
    lines = [{"role": "user", "message": {"content": [{"type": "text", "text": msg}]}}]
    rows = collect(write_jsonl(tmp_path, lines))
    csv_text = rows_to_csv(rows)
    reader = csv.DictReader(io.StringIO(csv_text))
    data_rows = list(reader)
    assert data_rows[0]["message_text"] == msg


# ---------------------------------------------------------------------------
# Integration: fixture file
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FIXTURE_PATH.exists(), reason="Fixture not found")
class TestSampleFixture:
    """
    Counts derived from cursor_sample.jsonl (actual Cursor nested schema).

    Prompt 1: read_file, grep_search, read_file, edit_file, run_terminal_cmd → 5 tool_calls
    Prompt 2: read_file, edit_file → 2 tool_calls
    Prompt 3: codebase_search, edit_file, list_dir, edit_file → 4 tool_calls
    Totals: 3 user_prompts, 11 tool_calls, 3 assistant_responses = 17 rows
    (The malformed line is skipped.)
    """

    def setup_method(self):
        self.rows = list(parse_cursor_jsonl(FIXTURE_PATH))
        self.by_type: dict[str, list] = {}
        for r in self.rows:
            self.by_type.setdefault(r["event_type"], []).append(r)

    def test_total_row_count(self):
        assert len(self.rows) == 17

    def test_user_prompt_count(self):
        assert len(self.by_type.get("user_prompt", [])) == 3

    def test_tool_call_count(self):
        assert len(self.by_type.get("tool_call", [])) == 11

    def test_assistant_response_count(self):
        assert len(self.by_type.get("assistant_response", [])) == 3

    def test_session_id_is_filename_stem(self):
        for r in self.rows:
            assert r["session_id"] == "cursor_sample"

    def test_working_dir_populated_for_shell_calls(self):
        shell_rows = [r for r in self.by_type.get("tool_call", []) if r["tool_name"] == "Shell"]
        assert len(shell_rows) == 1
        assert shell_rows[0]["working_dir"] == "/home/student/my-project"

    def test_first_tool_has_working_dir(self):
        read_rows = [r for r in self.by_type.get("tool_call", []) if r["tool_name"] == "Read"]
        assert read_rows[0]["working_dir"] == "/home/student/my-project"

    def test_no_token_data(self):
        for r in self.rows:
            assert r["input_tokens"] == ""
            assert r["output_tokens"] == ""

    def test_no_timestamp_data(self):
        for r in self.rows:
            assert r["timestamp"] == ""

    def test_message_text_on_first_user_prompt(self):
        user_rows = self.by_type.get("user_prompt", [])
        assert user_rows[0]["message_text"] == "Find and fix the authentication bug in the login flow"

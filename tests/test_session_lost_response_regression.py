"""Regression: lazy-retry run-journal recovery across multiple session reads.

The scenario this test pins down:

1. A WebUI process restarts mid-stream. On the first sidecar repair attempt
   the run-journal for the dead stream is NOT visible yet (page-cache loss,
   un-fsynced writes, slow network FS, etc.) so
   `_append_journaled_partial_output` returns False.
2. Pre-fix the repair path baked a permanent "no agent output was recovered"
   marker into the session and never looked at the journal again — even
   after the journaled tokens appeared on disk on a later read.
3. With the fix, the repair instead leaves a `_pending_journal_recovery`
   flag on the marker; the next `get_session()` call lazily re-runs the
   recovery, promotes the marker wording, and threads the journaled
   assistant text/tools into the transcript in the correct chronological
   position.
"""
import pytest

import api.models as models
import api.config as config
import api.profiles as profiles
import api.streaming as streaming  # noqa: F401  imported for fixture parity
from api.models import (
    Session,
    _apply_core_sync_or_error_marker,
)
from api.run_journal import append_run_event


# ── Fixtures (shape mirrors test_session_sidecar_repair.py) ────────────────


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    index_file = session_dir / "_index.json"
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", index_file)
    models.SESSIONS.clear()
    yield session_dir, index_file
    models.SESSIONS.clear()


@pytest.fixture(autouse=True)
def _isolate_stream_state():
    config.STREAMS.clear()
    config.CANCEL_FLAGS.clear()
    config.AGENT_INSTANCES.clear()
    config.STREAM_PARTIAL_TEXT.clear()
    yield
    config.STREAMS.clear()
    config.CANCEL_FLAGS.clear()
    config.AGENT_INSTANCES.clear()
    config.STREAM_PARTIAL_TEXT.clear()


@pytest.fixture(autouse=True)
def _isolate_agent_locks():
    config.SESSION_AGENT_LOCKS.clear()
    yield
    config.SESSION_AGENT_LOCKS.clear()


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / "sessions").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(profiles, "_DEFAULT_HERMES_HOME", home)
    return home


def _make_dead_stream_session(
    session_id: str,
    *,
    stream_id: str,
    existing_msgs_count: int = 96,
    pending_text: str = (
        "[IMPORTANT: Background process polling. "
        "Continue the user's prior request.]"
    ),
):
    """Build a session that mirrors the production bug: lots of prior history,
    pending_user_message set, an active_stream_id pointing at a dead stream,
    and pending_started_at populated."""
    messages = []
    for i in range(existing_msgs_count // 2):
        messages.append({"role": "user", "content": f"q{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
    s = Session(session_id=session_id, title="Lost-response repro", messages=messages)
    s.pending_user_message = pending_text
    s.pending_started_at = 1779237637  # production-shaped value
    s.active_stream_id = stream_id
    return s


# ── The regression test ────────────────────────────────────────────────────


def test_lost_response_recovered_on_second_read(hermes_home):
    sid = "9f14583f0e4e4444aaaa111122223333"
    stream_id = "7c8b4108d52b4aba9af362d3a54f47ac"

    # ── Stage 1: simulate page-cache loss — sidecar repair runs while the
    # run-journal for this stream is empty/absent on disk.
    s = _make_dead_stream_session(sid, stream_id=stream_id)
    s.save()
    core_path = hermes_home / "sessions" / f"session_{sid}.json"

    result = _apply_core_sync_or_error_marker(
        s, core_path, stream_id_for_recheck=stream_id,
    )
    assert result is True

    # Marker should carry the lazy-retry flag and *not* the permanent
    # "no agent output was recovered" wording yet.
    last = s.messages[-1]
    assert last.get("_error") is True
    assert last.get("type") == "interrupted"
    assert last.get("_pending_journal_recovery") is True, (
        "First repair pass should defer the recovery decision via a "
        "_pending_journal_recovery flag so a later read can self-heal."
    )
    assert last.get("_journal_retry_stream_id") == stream_id
    assert last.get("_journal_retry_attempts") == 0
    assert isinstance(last.get("_journal_retry_first_seen_ts"), int)
    assert "no agent output was recovered" not in last["content"]
    # pending fields cleared regardless of journal visibility
    assert s.pending_user_message is None
    assert s.active_stream_id is None
    assert s.pending_started_at is None

    # ── Stage 2: the journaled events become visible on disk.
    append_run_event(sid, stream_id, "token", {"text": "Checking GitHub first."})
    append_run_event(
        sid,
        stream_id,
        "tool",
        {
            "name": "terminal",
            "preview": "gh pr list --repo nesquena/hermes-webui",
            "args": {"command": "gh pr list --repo nesquena/hermes-webui"},
        },
    )
    append_run_event(
        sid,
        stream_id,
        "tool_complete",
        {"name": "terminal", "duration": 1.2, "is_error": False},
    )
    append_run_event(
        sid, stream_id, "token", {"text": " The first PR scan completed."},
    )

    # Pin session into the LRU cache and call get_session — this is the
    # production path that triggers lazy retry.
    models.SESSIONS[sid] = s
    reloaded = models.get_session(sid)
    assert reloaded is s

    contents = [m.get("content", "") for m in s.messages]
    # The marker self-healed:
    assert any("recovered from the run journal" in c for c in contents), (
        "After journaled tokens become readable, the marker must promote to "
        "the recovered-output wording."
    )
    assert not any("no agent output was recovered" in c for c in contents)

    # The journaled assistant text and tool card landed BEFORE the marker
    # so chronological order in the transcript is preserved.
    marker_idx = next(
        i for i, m in enumerate(s.messages)
        if m.get("type") == "interrupted" and m.get("_error")
    )
    recovered_msgs = [
        m for m in s.messages[:marker_idx]
        if m.get("_recovered_from_run_journal") is True
    ]
    assert recovered_msgs, "recovered assistant content must sit above the marker"
    recovered_text = " ".join(m.get("content", "") for m in recovered_msgs)
    assert "Checking GitHub first." in recovered_text
    assert "first PR scan completed" in recovered_text

    # Tool card lives in session.tool_calls and points at one of the
    # recovered assistant indices.
    assert s.tool_calls, "journaled tool should be materialized"
    assert s.tool_calls[-1]["name"] == "terminal"
    assert s.tool_calls[-1]["done"] is True

    # Flag and meta cleaned up after promotion.
    promoted = s.messages[marker_idx]
    assert "_pending_journal_recovery" not in promoted
    assert "_journal_retry_stream_id" not in promoted
    assert "_journal_retry_attempts" not in promoted
    assert "_journal_retry_first_seen_ts" not in promoted

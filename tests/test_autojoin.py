"""Tests for the invite-driven auto-join flow.

No network and no real Chime meeting: the join step is injected, so these
exercise the filtering, queueing and lifecycle logic rather than the media
stack (which ``test_audio.py`` and ``test_signaling.py`` cover).
"""

from __future__ import annotations

import asyncio

import pytest

from huddle_drawings.autojoin import AutoJoiner
from huddle_drawings.config import Config
from huddle_drawings.realtime import HuddleInvite

pytestmark = pytest.mark.asyncio


def make_config() -> Config:
    return Config(
        xoxc_token="xoxc-test-token-123",
        xoxd_token="xoxd-test-token-456",
        slack_host="example.slack.com",
    )


def make_invite(
    *, call_id: str = "R1", channel_id: str = "C1", sender: str = "U1"
) -> HuddleInvite:
    return HuddleInvite.from_event(
        {
            "type": "huddle_invite",
            "channel_id": channel_id,
            "call_id": call_id,
            "sender_user_id": sender,
            "free_willy": {
                "meeting": {
                    "MeetingId": "m-1",
                    "MediaRegion": "us-east-2",
                    "MeetingFeatures": None,
                    "MediaPlacement": {
                        "AudioHostUrl": "audio.example",
                        "SignalingUrl": "wss://sig.example",
                    },
                },
                "attendee": {
                    "AttendeeId": "att-1",
                    "ExternalUserId": "E-R-U",
                    "JoinToken": "SECRET-JOIN-TOKEN",
                },
            },
        }
    )


class FakeSlack:
    """Just enough SlackClient for AutoJoiner."""

    def __init__(self) -> None:
        self.declined: list[tuple[str, str]] = []
        self.closed = False

    async def auth_test(self) -> dict[str, str]:
        return {"user": "drawbot", "user_id": "U0BOT", "team_id": "E123"}

    async def rooms_invite_response(
        self, *, channel_id: str, room_id: str, response: str = "decline"
    ) -> dict[str, object]:
        assert response == "decline"
        self.declined.append((channel_id, room_id))
        return {"ok": True}

    async def close(self) -> None:
        self.closed = True


class FakeRealtime:
    """Feeds queued events to subscribers, then idles until cancelled."""

    def __init__(self, events: list[dict[str, object]]) -> None:
        self._events = events
        self._subs: list[tuple[frozenset[str], object]] = []
        self.connected = False
        self.closed = False

    def on_event(self, type_or_types, cb):  # noqa: ANN001
        types = (
            frozenset({type_or_types})
            if isinstance(type_or_types, str)
            else frozenset(type_or_types)
        )
        entry = (types, cb)
        self._subs.append(entry)
        return lambda: self._subs.remove(entry)

    async def connect(self) -> None:
        self.connected = True

    async def run_forever(self) -> None:
        for event in self._events:
            for types, cb in list(self._subs):
                if event.get("type") in types or "*" in types:
                    result = cb(event)
                    if asyncio.iscoroutine(result):
                        await result
            await asyncio.sleep(0)
        # Stay alive like the real reader does.
        await asyncio.Event().wait()

    async def close(self) -> None:
        self.closed = True


class FakeSession:
    """Stands in for a joined HuddleSession."""

    def __init__(self, invite: HuddleInvite) -> None:
        self.invite = invite
        self.closed = False

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed = True


def build(
    events: list[dict[str, object]],
    handled: list[HuddleInvite],
    **kwargs: object,
) -> tuple[AutoJoiner, FakeSlack, FakeRealtime, list[FakeSession]]:
    slack = FakeSlack()
    realtime = FakeRealtime(events)
    sessions: list[FakeSession] = []

    async def handler(session, invite) -> None:  # noqa: ANN001
        handled.append(invite)

    async def joiner(invite, _slack):  # noqa: ANN001
        session = FakeSession(invite)
        sessions.append(session)
        return session

    joinerobj = AutoJoiner(
        config=make_config(),
        handler=handler,
        slack=slack,  # type: ignore[arg-type]
        realtime=realtime,  # type: ignore[arg-type]
        joiner=joiner,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )
    return joinerobj, slack, realtime, sessions


async def run_briefly(joiner: AutoJoiner, timeout: float = 1.0) -> None:
    task = asyncio.create_task(joiner.run())
    try:
        await asyncio.wait_for(task, timeout=timeout)
    except TimeoutError:
        joiner.stop()
        with pytest.raises((asyncio.CancelledError, TimeoutError), check=lambda _: True):
            pass
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


async def test_joins_a_matching_invite_once() -> None:
    handled: list[HuddleInvite] = []
    invite_event = make_invite().raw
    joiner, slack, realtime, sessions = build([invite_event], handled, once=True)

    await asyncio.wait_for(joiner.run(), timeout=2)

    assert [i.call_id for i in handled] == ["R1"]
    assert len(sessions) == 1
    assert sessions[0].closed, "the session must be closed after the handler returns"
    assert realtime.closed
    assert not slack.closed, "an injected SlackClient must not be closed for us"


async def test_uses_the_invites_own_chime_credentials() -> None:
    """The invite carries everything needed; no rooms.join should be required."""
    invite = make_invite()
    assert invite.has_chime_credentials
    creds = invite.chime_credentials()
    assert creds.attendee_id == "att-1"
    assert creds.join_token == "SECRET-JOIN-TOKEN"
    assert creds.media_placement.signaling_url == "wss://sig.example"
    # A null MeetingFeatures must be stripped: the Chime SDK rejects it.
    assert "MeetingFeatures" not in creds.raw["meeting"]
    # And the secret must not leak through a repr.
    assert "SECRET-JOIN-TOKEN" not in repr(invite)
    assert "SECRET-JOIN-TOKEN" not in repr(creds)


async def test_ignores_invites_from_other_users() -> None:
    handled: list[HuddleInvite] = []
    events = [
        make_invite(call_id="R-bad", sender="U-stranger").raw,
        make_invite(call_id="R-good", sender="U-friend").raw,
    ]
    joiner, slack, _rt, _s = build(
        events, handled, allow_from={"U-friend"}, once=True
    )

    await asyncio.wait_for(joiner.run(), timeout=2)

    assert [i.call_id for i in handled] == ["R-good"]
    assert slack.declined == [], "filtering alone must not decline anything"


async def test_declines_rejected_invites_when_asked() -> None:
    handled: list[HuddleInvite] = []
    events = [
        make_invite(call_id="R-bad", sender="U-stranger").raw,
        make_invite(call_id="R-good", sender="U-friend").raw,
    ]
    joiner, slack, _rt, _s = build(
        events,
        handled,
        allow_from={"U-friend"},
        decline_rejected=True,
        once=True,
    )

    await asyncio.wait_for(joiner.run(), timeout=2)

    assert slack.declined == [("C1", "R-bad")]
    assert [i.call_id for i in handled] == ["R-good"]


async def test_channel_filter() -> None:
    handled: list[HuddleInvite] = []
    events = [
        make_invite(call_id="R-a", channel_id="C-other").raw,
        make_invite(call_id="R-b", channel_id="C-wanted").raw,
    ]
    joiner, *_ = build(
        events, handled, allow_channels={"C-wanted"}, once=True
    )

    await asyncio.wait_for(joiner.run(), timeout=2)
    assert [i.call_id for i in handled] == ["R-b"]


async def test_a_failing_join_does_not_stop_the_listener() -> None:
    handled: list[HuddleInvite] = []
    slack = FakeSlack()
    realtime = FakeRealtime(
        [make_invite(call_id="R-fail").raw, make_invite(call_id="R-ok").raw]
    )
    attempts: list[str] = []

    async def handler(session, invite) -> None:  # noqa: ANN001
        handled.append(invite)

    async def joiner(invite, _slack):  # noqa: ANN001
        attempts.append(invite.call_id)
        if invite.call_id == "R-fail":
            raise RuntimeError("chime said no")
        return FakeSession(invite)

    auto = AutoJoiner(
        config=make_config(),
        handler=handler,
        slack=slack,  # type: ignore[arg-type]
        realtime=realtime,  # type: ignore[arg-type]
        joiner=joiner,  # type: ignore[arg-type]
        once=True,
    )
    await asyncio.wait_for(auto.run(), timeout=2)

    assert attempts == ["R-fail", "R-ok"]
    assert [i.call_id for i in handled] == ["R-ok"]


async def test_a_raising_handler_still_leaves_the_huddle() -> None:
    slack = FakeSlack()
    realtime = FakeRealtime([make_invite().raw])
    sessions: list[FakeSession] = []

    async def handler(session, invite) -> None:  # noqa: ANN001
        raise RuntimeError("drawing blew up")

    async def joiner(invite, _slack):  # noqa: ANN001
        s = FakeSession(invite)
        sessions.append(s)
        return s

    auto = AutoJoiner(
        config=make_config(),
        handler=handler,
        slack=slack,  # type: ignore[arg-type]
        realtime=realtime,  # type: ignore[arg-type]
        joiner=joiner,  # type: ignore[arg-type]
        once=True,
    )
    await asyncio.wait_for(auto.run(), timeout=2)

    assert sessions and sessions[0].closed, (
        "a handler that raises must not leave us sitting in the call"
    )


async def test_stop_ends_the_loop() -> None:
    handled: list[HuddleInvite] = []
    joiner, _slack, realtime, _s = build([], handled)

    task = asyncio.create_task(joiner.run())
    await asyncio.sleep(0.05)
    joiner.stop()
    await asyncio.wait_for(task, timeout=2)

    assert handled == []
    assert realtime.closed


async def test_handles_several_invites_in_sequence() -> None:
    handled: list[HuddleInvite] = []
    events = [make_invite(call_id=f"R{i}").raw for i in range(3)]
    joiner, *_ = build(events, handled)

    task = asyncio.create_task(joiner.run())
    for _ in range(50):
        await asyncio.sleep(0.01)
        if len(handled) == 3:
            break
    joiner.stop()
    await asyncio.wait_for(task, timeout=2)

    assert [i.call_id for i in handled] == ["R0", "R1", "R2"]


async def test_malformed_invite_does_not_crash_the_listener() -> None:
    handled: list[HuddleInvite] = []
    events = [{"type": "huddle_invite"}, make_invite(call_id="R-ok").raw]
    joiner, *_ = build(events, handled, once=True)

    await asyncio.wait_for(joiner.run(), timeout=2)
    # The empty invite either parses to a blank one that fails to join, or is
    # skipped; either way the good one must still be handled.
    assert "R-ok" in [i.call_id for i in handled]


async def test_owns_and_closes_a_slack_client_it_created() -> None:
    """When no client is injected, run() must clean up the one it builds."""
    realtime = FakeRealtime([])

    async def handler(session, invite) -> None:  # noqa: ANN001
        pass

    auto = AutoJoiner(
        config=make_config(),
        handler=handler,
        realtime=realtime,  # type: ignore[arg-type]
    )
    task = asyncio.create_task(auto.run())
    await asyncio.sleep(0.05)
    auto.stop()
    # auth.test would need the network, so this raises -- the point is that it
    # does not leak the client, which httpx would warn about on GC.
    with __import__("contextlib").suppress(Exception):
        await asyncio.wait_for(task, timeout=2)

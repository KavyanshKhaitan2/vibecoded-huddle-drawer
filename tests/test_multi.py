"""Tests for multi-account colour drawing and account loading."""

from __future__ import annotations

import asyncio
import json

import pytest

from huddle_drawings.config import Config, ConfigError, load_accounts
from huddle_drawings.geometry import Point, Segment
from huddle_drawings.multi import Layer, MultiSession
from huddle_drawings.ratelimit import TokenBucket
from huddle_drawings.types import HuddleError

pytestmark = pytest.mark.asyncio


def seg(x: float = 0.1) -> Segment:
    p = Point(x, x)
    return Segment(p, p, p, p)


# --------------------------------------------------------------------------- #
# account loading
# --------------------------------------------------------------------------- #


async def test_load_accounts_numbered_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for n, (c, d) in enumerate(
        [("xoxc-aaa1", "xoxd-aaa1"), ("xoxc-bbb2", "xoxd-bbb2"), ("xoxc-ccc3", "xoxd-ccc3")]
    ):
        suffix = "" if n == 0 else f"_{n + 1}"
        monkeypatch.setenv(f"XOXC_TOKEN{suffix}", c)
        monkeypatch.setenv(f"XOXD_TOKEN{suffix}", d)
    monkeypatch.setenv("SLACK_HOST", "example.slack.com")

    accounts = load_accounts()
    assert [a.xoxc_token for a in accounts] == ["xoxc-aaa1", "xoxc-bbb2", "xoxc-ccc3"]
    assert all(a.slack_host == "example.slack.com" for a in accounts)


async def test_load_accounts_stops_at_a_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gap must stop the scan, not silently skip to later accounts."""
    monkeypatch.setenv("XOXC_TOKEN", "xoxc-aaa1")
    monkeypatch.setenv("XOXD_TOKEN", "xoxd-aaa1")
    # No _2, but a _3 exists: the _3 must not be picked up silently.
    monkeypatch.delenv("XOXC_TOKEN_2", raising=False)
    monkeypatch.delenv("XOXD_TOKEN_2", raising=False)
    monkeypatch.setenv("XOXC_TOKEN_3", "xoxc-ccc3")
    monkeypatch.setenv("XOXD_TOKEN_3", "xoxd-ccc3")

    accounts = load_accounts()
    assert len(accounts) == 1


async def test_load_accounts_rejects_half_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XOXC_TOKEN", "xoxc-aaa1")
    monkeypatch.setenv("XOXD_TOKEN", "xoxd-aaa1")
    monkeypatch.setenv("XOXC_TOKEN_2", "xoxc-bbb2")
    monkeypatch.delenv("XOXD_TOKEN_2", raising=False)

    with pytest.raises(ConfigError, match="half-configured"):
        load_accounts()


async def test_load_accounts_rejects_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two identical accounts are one attendee, so one line in one hue."""
    monkeypatch.setenv("XOXC_TOKEN", "xoxc-same")
    monkeypatch.setenv("XOXD_TOKEN", "xoxd-aaa1")
    monkeypatch.setenv("XOXC_TOKEN_2", "xoxc-same")
    monkeypatch.setenv("XOXD_TOKEN_2", "xoxd-bbb2")

    with pytest.raises(ConfigError, match="same XOXC token"):
        load_accounts()


async def test_load_accounts_from_json_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLACK_HOST", "base.slack.com")
    path = tmp_path / "accounts.json"
    path.write_text(
        json.dumps(
            [
                {"xoxc": "xoxc-one", "xoxd": "xoxd-one"},
                {"xoxc": "xoxc-two", "xoxd": "xoxd-two", "slack_host": "other.slack.com"},
            ]
        )
    )
    accounts = load_accounts(accounts_file=path)
    assert [a.xoxc_token for a in accounts] == ["xoxc-one", "xoxc-two"]
    assert accounts[0].slack_host == "base.slack.com"  # inherited
    assert accounts[1].slack_host == "other.slack.com"  # overridden


async def test_load_accounts_rejects_bad_json(tmp_path) -> None:
    path = tmp_path / "accounts.json"
    path.write_text("{not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_accounts(accounts_file=path)


async def test_load_accounts_rejects_entry_without_tokens(tmp_path) -> None:
    path = tmp_path / "accounts.json"
    path.write_text(json.dumps([{"xoxc": "xoxc-one"}]))
    with pytest.raises(ConfigError, match="needs both"):
        load_accounts(accounts_file=path)


# --------------------------------------------------------------------------- #
# MultiSession
# --------------------------------------------------------------------------- #


class FakeDrawer:
    def __init__(self, hue: int | None) -> None:
        self.hue = hue
        self.strokes: list[int] = []
        self.ended = 0
        self.keep_alive_started = 0
        self.keep_alive_stopped = 0
        # DrawClient exposes these counters and multi.py logs them.
        self.sent = 0
        self.dropped = 0

    async def stroke(self, segments, *, end: bool = True) -> None:  # noqa: ANN001
        self.strokes.append(len(segments))
        self.sent += len(segments)

    def start_keep_alive(self) -> None:
        self.keep_alive_started += 1

    async def stop_keep_alive(self) -> None:
        self.keep_alive_stopped += 1

    async def end_stroke(self, *, twice: bool = True) -> None:
        self.ended += 1


class FakeSession:
    def __init__(self, attendee: str) -> None:
        self.attendee_id = attendee
        self.closed = False
        self.drawers: list[FakeDrawer] = []

    def drawer(self, share, *, hue=None):  # noqa: ANN001
        d = FakeDrawer(hue)
        self.drawers.append(d)
        return d

    async def close(self) -> None:
        self.closed = True


def multi(n: int) -> MultiSession:
    return MultiSession(
        sessions=[FakeSession(f"att-{i}") for i in range(n)],  # type: ignore[arg-type]
        limiter=TokenBucket(),
    )


async def test_draw_layers_assigns_one_hue_per_account() -> None:
    m = multi(3)
    layers = [Layer(hue=h, segments=[seg()] * (i + 1)) for i, h in enumerate((0, 120, 240))]
    await m.draw_layers(layers, "SHARER", keep_alive_while_drawing=False)

    hues = [s.drawers[0].hue for s in m.sessions]  # type: ignore[attr-defined]
    assert hues == [0, 120, 240]
    totals = [sum(s.drawers[0].strokes) for s in m.sessions]  # type: ignore[attr-defined]
    assert totals == [1, 2, 3], "every layer's segments must all be sent"


async def test_draw_layers_ends_every_line() -> None:
    """All lines must be ENDed, or they fade at staggered times."""
    m = multi(2)
    await m.draw_layers(
        [Layer(0, [seg()]), Layer(180, [seg()])], "S", keep_alive_while_drawing=False
    )
    assert all(s.drawers[0].ended == 1 for s in m.sessions)  # type: ignore[attr-defined]


async def test_more_layers_than_attendees_raises_with_an_explanation() -> None:
    """Silently dropping a colour makes a picture subtly wrong."""
    m = multi(2)
    layers = [Layer(h, [seg()]) for h in (0, 120, 240)]
    with pytest.raises(HuddleError, match="need 3 attendees"):
        await m.draw_layers(layers, "S")


async def test_fewer_layers_than_accounts_is_fine() -> None:
    m = multi(4)
    await m.draw_layers([Layer(0, [seg()])], "S", keep_alive_while_drawing=False)
    assert m.sessions[0].drawers  # type: ignore[attr-defined]
    assert not m.sessions[3].drawers  # type: ignore[attr-defined]


async def test_keep_alive_holds_finished_layers_in_the_sequential_path() -> None:
    """Only the sequential path needs keep-alives; interleaving never idles."""
    m = multi(2)
    await m.draw_layers(
        [Layer(0, [seg()]), Layer(180, [seg()])],
        "S",
        interleave=False,
        keep_alive_while_drawing=True,
    )
    for s in m.sessions:
        d = s.drawers[0]  # type: ignore[attr-defined]
        assert d.keep_alive_started >= 1
        assert d.keep_alive_stopped >= 1


async def test_no_layers_is_a_noop() -> None:
    m = multi(2)
    await m.draw_layers([], "S")
    assert not m.sessions[0].drawers  # type: ignore[attr-defined]


async def test_close_closes_every_session_and_is_idempotent() -> None:
    m = multi(3)
    sessions = list(m.sessions)
    await m.close()
    await m.close()
    assert all(s.closed for s in sessions)  # type: ignore[attr-defined]
    assert len(m) == 0


async def test_join_all_requires_accounts() -> None:
    with pytest.raises(HuddleError, match="no accounts"):
        await MultiSession.join_all([], channel_id="C1")


async def test_join_all_shares_one_rate_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chime's limit is per *meeting*, so per-account buckets would overrun it.

    Each account getting its own bucket would let N accounts send N x the
    meeting's allowance, and the excess is dropped silently with no error.
    """
    seen: list[TokenBucket] = []

    async def fake_join(config, *, channel_id, muted, rate_limiter, multidevice=None):  # noqa: ANN001
        seen.append(rate_limiter)
        return FakeSession(f"att-{len(seen)}")

    from huddle_drawings import multi as multi_module

    monkeypatch.setattr(multi_module.HuddleSession, "join", staticmethod(fake_join))

    configs = [
        Config(xoxc_token=f"xoxc-{i}", xoxd_token=f"xoxd-{i}") for i in range(3)
    ]
    m = await MultiSession.join_all(configs, channel_id="C1")

    assert len(seen) == 3
    assert all(bucket is seen[0] for bucket in seen), "all accounts share one bucket"
    assert seen[0] is m.limiter


async def test_join_all_skips_a_failing_account_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    async def fake_join(config, *, channel_id, muted, rate_limiter, multidevice=None):  # noqa: ANN001
        calls["n"] += 1
        if config.xoxc_token == "xoxc-1":
            raise HuddleError("nope")
        return FakeSession(config.xoxc_token)

    from huddle_drawings import multi as multi_module

    monkeypatch.setattr(multi_module.HuddleSession, "join", staticmethod(fake_join))

    configs = [
        Config(xoxc_token=f"xoxc-{i}", xoxd_token=f"xoxd-{i}") for i in range(3)
    ]
    m = await MultiSession.join_all(configs, channel_id="C1")
    assert len(m) == 2, "losing one colour beats losing the whole picture"


async def test_join_all_require_all_is_fatal_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[FakeSession] = []

    async def fake_join(config, *, channel_id, muted, rate_limiter, multidevice=None):  # noqa: ANN001
        if config.xoxc_token == "xoxc-1":
            raise HuddleError("nope")
        s = FakeSession(config.xoxc_token)
        created.append(s)
        return s

    from huddle_drawings import multi as multi_module

    monkeypatch.setattr(multi_module.HuddleSession, "join", staticmethod(fake_join))

    configs = [
        Config(xoxc_token=f"xoxc-{i}", xoxd_token=f"xoxd-{i}") for i in range(3)
    ]
    with pytest.raises(HuddleError, match="could not join"):
        await MultiSession.join_all(configs, channel_id="C1", require_all=True)
    assert created and all(s.closed for s in created), "must not leak joined sessions"


async def test_join_all_fails_when_no_account_joins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_join(config, *, channel_id, muted, rate_limiter, multidevice=None):  # noqa: ANN001
        raise HuddleError("nope")

    from huddle_drawings import multi as multi_module

    monkeypatch.setattr(multi_module.HuddleSession, "join", staticmethod(fake_join))

    with pytest.raises(HuddleError):
        await MultiSession.join_all(
            [Config(xoxc_token="xoxc-1", xoxd_token="xoxd-1")], channel_id="C1"
        )


async def test_wait_for_share_races_the_sessions() -> None:
    class Slow(FakeSession):
        async def wait_for_share(self, timeout=None):  # noqa: ANN001
            await asyncio.sleep(5)
            raise AssertionError("should have been cancelled")

    class Fast(FakeSession):
        async def wait_for_share(self, timeout=None):  # noqa: ANN001
            return "THE-SHARE"

    m = MultiSession(
        sessions=[Slow("a"), Fast("b")],  # type: ignore[list-item]
        limiter=TokenBucket(),
    )
    assert await asyncio.wait_for(m.wait_for_share(), timeout=2) == "THE-SHARE"


# --------------------------------------------------------------------------- #
# multidevice: several attendees from ONE account
#
# Live-verified 2026-11-08: rooms.join with multidevice=true returns a fresh
# AttendeeId each call, and two such sessions drawing hue 0 and hue 210
# rendered as a red line and a blue line at the same time. This is what makes
# multi-colour drawing possible without extra accounts, and it contradicts
# docs/huddles-and-chime.md's claim that the id is stable "whatever the
# params".
# --------------------------------------------------------------------------- #


async def test_count_spawns_that_many_sessions_from_one_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[bool | None] = []
    counter = {"n": 0}

    async def fake_join(config, *, channel_id, muted, rate_limiter, multidevice=None):  # noqa: ANN001
        seen.append(multidevice)
        counter["n"] += 1
        return FakeSession(f"att-{counter['n']}")

    from huddle_drawings import multi as multi_module

    monkeypatch.setattr(multi_module.HuddleSession, "join", staticmethod(fake_join))

    one = [Config(xoxc_token="xoxc-only", xoxd_token="xoxd-only")]
    m = await MultiSession.join_all(one, channel_id="C1", count=4)

    assert len(m) == 4, "one account must be able to supply four attendees"
    assert seen == [True] * 4, "extra attendees require multidevice=true"


async def test_single_session_does_not_request_multidevice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lone session should reuse the account's stable attendee."""
    seen: list[bool | None] = []

    async def fake_join(config, *, channel_id, muted, rate_limiter, multidevice=None):  # noqa: ANN001
        seen.append(multidevice)
        return FakeSession("att-1")

    from huddle_drawings import multi as multi_module

    monkeypatch.setattr(multi_module.HuddleSession, "join", staticmethod(fake_join))

    await MultiSession.join_all(
        [Config(xoxc_token="xoxc-a", xoxd_token="xoxd-a")], channel_id="C1"
    )
    assert seen == [False] or seen == [None], seen


async def test_accounts_are_cycled_when_fewer_than_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    used: list[str] = []
    counter = {"n": 0}

    async def fake_join(config, *, channel_id, muted, rate_limiter, multidevice=None):  # noqa: ANN001
        used.append(config.xoxc_token)
        counter["n"] += 1
        return FakeSession(f"att-{counter['n']}")

    from huddle_drawings import multi as multi_module

    monkeypatch.setattr(multi_module.HuddleSession, "join", staticmethod(fake_join))

    accounts = [
        Config(xoxc_token="xoxc-a", xoxd_token="xoxd-a"),
        Config(xoxc_token="xoxc-b", xoxd_token="xoxd-b"),
    ]
    await MultiSession.join_all(accounts, channel_id="C1", count=5)
    assert used == ["xoxc-a", "xoxc-b", "xoxc-a", "xoxc-b", "xoxc-a"]


async def test_warns_when_attendees_collapse(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If a workspace ignores multidevice, the colours would silently merge."""

    async def fake_join(config, *, channel_id, muted, rate_limiter, multidevice=None):  # noqa: ANN001
        return FakeSession("same-attendee")

    from huddle_drawings import multi as multi_module

    monkeypatch.setattr(multi_module.HuddleSession, "join", staticmethod(fake_join))

    with caplog.at_level("WARNING"):
        await MultiSession.join_all(
            [Config(xoxc_token="xoxc-a", xoxd_token="xoxd-a")],
            channel_id="C1",
            count=3,
        )
    assert "distinct attendee" in caplog.text


async def test_count_must_be_positive() -> None:
    with pytest.raises(HuddleError, match="at least 1"):
        await MultiSession.join_all(
            [Config(xoxc_token="xoxc-a", xoxd_token="xoxd-a")],
            channel_id="C1",
            count=0,
        )


# --------------------------------------------------------------------------- #
# interleaving: why early layers used to fade first
#
# Drawing one layer to completion at a time leaves the short layers silent for
# the rest of the picture. The renderer starts fading a line 5s after its
# sender's last message, so on a big drawing the first colours vanish while
# the biggest is still going. Keep-alives can cover it only if every one gets
# through -- they share the meeting's budget with the drawing and Chime drops
# the excess silently.
# --------------------------------------------------------------------------- #


async def test_interleave_order_spreads_every_layer_across_the_whole_drawing() -> None:
    order = MultiSession._interleave_order([1200, 400, 200])
    total = len(order)
    assert total == 1800
    for index in range(3):
        at = [i for i, x in enumerate(order) if x == index]
        # Every layer must start near the beginning and end near the end.
        assert at[0] / total < 0.01, index
        assert at[-1] / total > 0.99, index


async def test_interleave_order_keeps_the_small_layer_from_going_quiet() -> None:
    """The gap between a small layer's sends must stay far below the fade."""
    sizes = [1200, 400, 200]
    order = MultiSession._interleave_order(sizes)
    smallest = sizes.index(min(sizes))
    gaps, run = [], 0
    for x in order:
        if x == smallest:
            gaps.append(run)
            run = 0
        else:
            run += 1
    # At ~120 messages/second, 5s of silence is 600 messages.
    assert max(gaps) < 100, max(gaps)


async def test_interleave_order_preserves_every_segment_exactly_once() -> None:
    sizes = [7, 3, 11]
    order = MultiSession._interleave_order(sizes)
    for index, size in enumerate(sizes):
        assert order.count(index) == size


async def test_interleave_order_handles_empty_and_single_layers() -> None:
    assert MultiSession._interleave_order([]) == []
    assert MultiSession._interleave_order([0, 0]) == []
    assert MultiSession._interleave_order([3]) == [0, 0, 0]
    # An empty layer must not appear at all.
    assert set(MultiSession._interleave_order([2, 0, 1])) == {0, 2}


class TimedDrawer(FakeDrawer):
    """Records the order in which segments were handed over."""

    def __init__(self, hue: int | None, tape: list[int], index: int) -> None:
        super().__init__(hue)
        self.tape = tape
        self.index = index

    async def stroke(self, segments, *, end: bool = True) -> None:  # noqa: ANN001
        for _ in segments:
            self.tape.append(self.index)
            # A real send awaits the shared rate limiter; without an await
            # point here asyncio.gather would run each layer to completion in
            # turn and hide the very scheduling this test is about.
            await asyncio.sleep(0)
        await super().stroke(segments, end=end)


async def test_interleaved_draw_finishes_all_layers_together() -> None:
    tape: list[int] = []

    class Session(FakeSession):
        def __init__(self, attendee: str, index: int) -> None:
            super().__init__(attendee)
            self.index = index

        def drawer(self, share, *, hue=None):  # noqa: ANN001
            d = TimedDrawer(hue, tape, self.index)
            self.drawers.append(d)
            return d

    m = MultiSession(
        sessions=[Session(f"att-{i}", i) for i in range(3)],  # type: ignore[arg-type]
        limiter=TokenBucket(),
    )
    layers = [
        Layer(hue=0, segments=[seg()] * 60),
        Layer(hue=120, segments=[seg()] * 20),
        Layer(hue=240, segments=[seg()] * 10),
    ]
    await m.draw_layers(layers, "S", interleave=True)

    assert len(tape) == 90
    # Each layer's last segment must land in the final tenth of the drawing.
    for index in range(3):
        last = max(i for i, x in enumerate(tape) if x == index)
        assert last >= len(tape) * 0.9, (index, last)


async def test_sequential_draw_finishes_small_layers_early() -> None:
    """The old behaviour, kept for comparison -- and this is the bug it had."""
    tape: list[int] = []

    class Session(FakeSession):
        def __init__(self, attendee: str, index: int) -> None:
            super().__init__(attendee)
            self.index = index

        def drawer(self, share, *, hue=None):  # noqa: ANN001
            d = TimedDrawer(hue, tape, self.index)
            self.drawers.append(d)
            return d

    m = MultiSession(
        sessions=[Session(f"att-{i}", i) for i in range(2)],  # type: ignore[arg-type]
        limiter=TokenBucket(),
    )
    layers = [Layer(hue=0, segments=[seg()] * 60), Layer(hue=120, segments=[seg()] * 5)]
    await m.draw_layers(
        layers, "S", interleave=False, keep_alive_while_drawing=False
    )

    small_last = max(i for i, x in enumerate(tape) if x == 1)
    assert small_last < len(tape) * 0.5, (
        "sequential drawing finishes the small layer early, which is why it "
        "faded first"
    )


async def test_interleaving_needs_no_keep_alives() -> None:
    m = multi(3)
    layers = [
        Layer(hue=0, segments=[seg()] * 30),
        Layer(hue=120, segments=[seg()] * 5),
        Layer(hue=240, segments=[seg()] * 5),
    ]
    await m.draw_layers(layers, "S", interleave=True)
    for session in m.sessions:
        drawer = session.drawers[0]  # type: ignore[attr-defined]
        assert drawer.keep_alive_started == 0, (
            "no layer goes quiet, so nothing needs keeping alive"
        )


async def test_hold_keeps_every_layer_alive_then_ends_them_together() -> None:
    m = multi(2)
    layers = [Layer(hue=0, segments=[seg()] * 4), Layer(hue=180, segments=[seg()] * 2)]
    await m.draw_layers(layers, "S", hold=0.01)
    for session in m.sessions:
        drawer = session.drawers[0]  # type: ignore[attr-defined]
        assert drawer.keep_alive_started >= 1, "hold must keep every layer up"
        assert drawer.keep_alive_stopped >= 1
        assert drawer.ended == 1, "every line must be ENDed so they fade together"


# --------------------------------------------------------------------------- #
# interruptible drawing and holding
# --------------------------------------------------------------------------- #


async def test_stop_event_cuts_the_hold_short_but_still_ends_the_lines() -> None:
    """Ctrl-C during a hold must stop holding, not leave the lines dangling."""
    m = multi(2)
    stop = asyncio.Event()
    stop.set()  # already interrupted
    await m.draw_layers(
        [Layer(0, [seg()]), Layer(180, [seg()])], "S", hold=30.0, stop=stop
    )
    for session in m.sessions:
        drawer = session.drawers[0]  # type: ignore[attr-defined]
        assert drawer.ended == 1, "an interrupted hold must still END the line"


async def test_stop_event_during_the_hold_returns_promptly() -> None:
    m = multi(1)
    stop = asyncio.Event()

    async def interrupt() -> None:
        await asyncio.sleep(0.05)
        stop.set()

    task = asyncio.create_task(interrupt())
    # hold=30s, but the interrupt should end it in well under a second.
    await asyncio.wait_for(
        m.draw_layers([Layer(0, [seg()])], "S", hold=30.0, stop=stop), timeout=3
    )
    await task
    assert m.sessions[0].drawers[0].ended == 1  # type: ignore[attr-defined]


async def test_stop_event_abandons_the_rest_of_the_drawing() -> None:
    """A big drawing must be abandonable mid-send, keeping what is drawn."""
    tape: list[int] = []
    stop = asyncio.Event()

    class Session(FakeSession):
        def drawer(self, share, *, hue=None):  # noqa: ANN001
            d = TimedDrawer(hue, tape, 0)
            self.drawers.append(d)
            return d

    m = MultiSession(
        sessions=[Session("att-0")],  # type: ignore[arg-type]
        limiter=TokenBucket(),
    )

    async def interrupt() -> None:
        while len(tape) < 5:
            await asyncio.sleep(0)
        stop.set()

    task = asyncio.create_task(interrupt())
    await asyncio.wait_for(
        m.draw_layers([Layer(0, [seg()] * 500)], "S", stop=stop), timeout=5
    )
    await task
    assert 5 <= len(tape) < 500, len(tape)
    assert m.sessions[0].drawers[0].ended == 1  # type: ignore[attr-defined]


async def test_no_stop_event_behaves_as_before() -> None:
    m = multi(1)
    await m.draw_layers([Layer(0, [seg()] * 3)], "S", hold=0.0)
    assert m.sessions[0].drawers[0].ended == 1  # type: ignore[attr-defined]

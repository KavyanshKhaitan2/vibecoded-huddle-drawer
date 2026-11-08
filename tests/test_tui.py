"""Tests for the interactive console.

The menu is driven through the ``Console`` protocol, so these script a
transcript of answers instead of using a terminal.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from huddle_drawings.config import LINE_CLEAR_WAIT
from huddle_drawings.geometry import Point, Rect, Segment
from huddle_drawings.multi import Layer, MultiSession
from huddle_drawings.ratelimit import TokenBucket
from huddle_drawings.tui import Tui, TuiState
from huddle_drawings.types import HuddleError, StreamSource

pytestmark = pytest.mark.asyncio


class ScriptedConsole:
    """Answers prompts from a list; records everything written."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.lines: list[str] = []
        self.prompts: list[str] = []

    def write(self, text: str = "") -> None:
        self.lines.append(text)

    async def ask(self, prompt: str, default: str = "") -> str:
        self.prompts.append(prompt)
        if not self.answers:
            return "q"  # always terminate rather than hang
        answer = self.answers.pop(0)
        return answer if answer != "" else default

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class FakeDrawer:
    def __init__(self, hue: int | None) -> None:
        self.hue = hue
        self.segment_count = 0
        # DrawClient exposes these and multi.py logs them after drawing.
        self.sent = 0
        self.dropped = 0

    async def stroke(self, segments, *, end: bool = True) -> None:  # noqa: ANN001
        self.segment_count += len(segments)
        self.sent += len(segments)

    def start_keep_alive(self) -> None: ...
    async def stop_keep_alive(self) -> None: ...
    async def end_stroke(self, *, twice: bool = True) -> None: ...


class FakeSignaling:
    def __init__(self) -> None:
        self.sources: dict[int, StreamSource] = {}
        self.sent: list[tuple[str, bytes]] = []
        self.callbacks: list[object] = []

    async def send_data_message(self, topic, data, *, lifetime_ms=0, wait=True):  # noqa: ANN001
        self.sent.append((topic, data))
        return True

    def on_data_message(self, cb):  # noqa: ANN001
        self.callbacks.append(cb)
        return lambda: self.callbacks.remove(cb)


class FakeSession:
    def __init__(self, attendee: str) -> None:
        self.attendee_id = attendee
        self.signaling = FakeSignaling()
        self.drawers: list[FakeDrawer] = []
        self.closed = False

    def drawer(self, share, *, hue=None):  # noqa: ANN001
        d = FakeDrawer(hue)
        self.drawers.append(d)
        return d

    async def close(self) -> None:
        self.closed = True


SHARE = StreamSource(
    stream_id=11, attendee_id="SHARER#content", media_type="VIDEO"
)


def build(answers: list[str], *, attendees: int = 1, share=SHARE):
    multi = MultiSession(
        sessions=[FakeSession(f"att-{i}") for i in range(attendees)],  # type: ignore[arg-type]
        limiter=TokenBucket(),
    )
    # hold=0 so draw_layers does not actually sleep in tests
    state = TuiState(multi=multi, share=share, hold=0.0)
    console = ScriptedConsole(answers)
    return Tui(state, console), state, console


def seg(x: float = 0.3) -> Segment:
    p = Point(x, x)
    return Segment(p, p, p, p)


# --------------------------------------------------------------------------- #
# loop basics
# --------------------------------------------------------------------------- #


async def test_quit_leaves_the_loop() -> None:
    tui, _state, console = build(["q"])
    await asyncio.wait_for(tui.run(), timeout=2)
    assert not tui.running
    assert "leaving the huddle" in console.text


async def test_unknown_option_is_reported_and_loop_continues() -> None:
    tui, _state, console = build(["zzz", "q"])
    await asyncio.wait_for(tui.run(), timeout=2)
    assert "no such option" in console.text


async def test_menu_shows_state() -> None:
    tui, state, console = build(["q"], attendees=3)
    state.hues = [0, 120, 240]
    await asyncio.wait_for(tui.run(), timeout=2)
    assert "3 attendee(s)" in console.text
    assert "hue(s) 0, 120, 240" in console.text


async def test_a_handler_that_raises_does_not_kill_the_session() -> None:
    """Losing the session because one action failed would mean re-joining."""
    tui, _state, console = build(["1", "q"])

    async def boom(_tui: Tui) -> None:
        raise RuntimeError("kaboom")

    tui.menu = [m for m in tui.menu if m.key != "1"] + [
        type(tui.menu[0])("1", "boom", boom, True)
    ]
    await asyncio.wait_for(tui.run(), timeout=2)
    assert "failed: RuntimeError: kaboom" in console.text
    assert "leaving the huddle" in console.text  # reached the quit afterwards


async def test_actions_needing_a_share_are_refused_without_one() -> None:
    tui, _state, console = build(["1", "q"], share=None)
    # Suppress the automatic share hunt by pre-setting it to None and making
    # the lookup fail fast.
    tui.state.multi.sessions[0].signaling.sources = {}  # type: ignore[attr-defined]

    async def no_share(_tui: Tui) -> None:
        raise TimeoutError

    tui.menu = [
        m if m.key != "8" else type(m)("8", m.label, no_share) for m in tui.menu
    ]
    await asyncio.wait_for(tui.run(), timeout=5)
    assert "no screen share yet" in console.text


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #


async def test_draw_shape_sends_segments() -> None:
    tui, state, console = build(["1", "heart", "q"])
    await asyncio.wait_for(tui.run(), timeout=5)
    drawers = state.multi.sessions[0].drawers  # type: ignore[attr-defined]
    assert drawers and drawers[0].segment_count > 0
    assert "drawn" in console.text


async def test_draw_shape_rejects_an_unknown_name() -> None:
    tui, state, console = build(["1", "dodecahedron", "q"])
    await asyncio.wait_for(tui.run(), timeout=5)
    assert "unknown shape" in console.text
    assert not state.multi.sessions[0].drawers  # type: ignore[attr-defined]


async def test_draw_reports_the_message_cost() -> None:
    """Every segment is one Chime message against a shared budget."""
    tui, _state, console = build(["1", "circle", "q"])
    await asyncio.wait_for(tui.run(), timeout=5)
    assert "of the meeting's shared" in console.text


async def test_draw_text_uses_the_stroke_font() -> None:
    tui, state, console = build(["3", "HI", "q"])
    await asyncio.wait_for(tui.run(), timeout=5)
    assert state.multi.sessions[0].drawers[0].segment_count > 0  # type: ignore[attr-defined]


async def test_draw_text_with_empty_input_does_nothing() -> None:
    tui, state, console = build(["3", " ", "q"])
    await asyncio.wait_for(tui.run(), timeout=5)
    assert "nothing to draw" in console.text
    assert not state.multi.sessions[0].drawers  # type: ignore[attr-defined]


async def test_draw_svg_rejects_arcs_without_dying() -> None:
    tui, _state, console = build(["4", "M 0 0 A 1 1 0 0 1 1 1", "q"])
    await asyncio.wait_for(tui.run(), timeout=5)
    assert "arc" in console.text.lower()
    assert "leaving the huddle" in console.text


async def test_layers_from_segments_splits_across_hues() -> None:
    tui, state, _console = build([], attendees=3)
    state.hues = [0, 120, 240]
    layers = tui.layers_from_segments([seg(i / 10) for i in range(9)])
    assert [layer.hue for layer in layers] == [0, 120, 240]
    assert [len(layer.segments) for layer in layers] == [3, 3, 3]


async def test_layers_from_segments_single_hue() -> None:
    tui, state, _console = build([])
    state.hues = [42]
    layers = tui.layers_from_segments([seg(), seg()])
    assert len(layers) == 1
    assert layers[0].hue == 42
    assert len(layers[0].segments) == 2


async def test_more_colours_than_attendees_is_refused() -> None:
    tui, state, _console = build([])
    state.hues = [0]
    with pytest.raises(HuddleError, match="need 2 attendees"):
        await tui.draw([Layer(0, [seg()]), Layer(180, [seg()])])


async def test_drawing_nothing_is_a_noop() -> None:
    tui, state, console = build([])
    await tui.draw([Layer(0, [])])
    assert "nothing to draw" in console.text
    assert not state.multi.sessions[0].drawers  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# the fade state machine -- the thing the TUI exists to manage
# --------------------------------------------------------------------------- #


async def test_seconds_until_clear_counts_down_from_the_last_end() -> None:
    _tui, state, _console = build([])
    assert state.seconds_until_clear() == 0.0
    state.last_end = time.monotonic()
    remaining = state.seconds_until_clear()
    assert 0 < remaining <= LINE_CLEAR_WAIT


async def test_wait_for_clear_returns_immediately_when_nothing_is_fading() -> None:
    tui, _state, _console = build([])
    assert await tui.wait_for_clear() is True


async def test_wait_for_clear_can_be_declined() -> None:
    """Drawing early is allowed; the old line comes back at full opacity."""
    tui, state, console = build(["n"])
    state.last_end = time.monotonic()
    assert await tui.wait_for_clear() is False
    assert "full opacity" in console.text


async def test_wait_for_clear_waits_then_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tui, state, console = build(["y"])
    # Pretend the fade is nearly over so the test does not sleep for 8.5s.
    state.last_end = time.monotonic() - (LINE_CLEAR_WAIT - 0.05)
    assert await asyncio.wait_for(tui.wait_for_clear(), timeout=3) is True
    assert "waiting" in console.text


async def test_draw_records_the_end_time_for_the_next_drawing() -> None:
    tui, state, _console = build(["1", "circle", "q"])
    assert state.last_end is None
    await asyncio.wait_for(tui.run(), timeout=5)
    assert state.last_end is not None


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #


async def test_set_hues_validates_and_clamps_to_attendees() -> None:
    tui, state, console = build(["5", "10,400,abc,20", "q"], attendees=1)
    await asyncio.wait_for(tui.run(), timeout=5)
    assert state.hues == [10], state.hues
    assert "outside 0-359" in console.text
    assert "not a whole number" in console.text


async def test_set_hues_keeps_current_when_all_invalid() -> None:
    tui, state, console = build(["5", "nope,999", "q"])
    before = list(state.hues)
    await asyncio.wait_for(tui.run(), timeout=5)
    assert state.hues == before
    assert "keeping the current ones" in console.text


async def test_set_hues_accepts_several_for_several_attendees() -> None:
    tui, state, _console = build(["5", "0,120,240", "q"], attendees=3)
    await asyncio.wait_for(tui.run(), timeout=5)
    assert state.hues == [0, 120, 240]


async def test_set_rect() -> None:
    tui, state, _console = build(["6", "0.1", "0.2", "0.3", "0.4", "q"])
    await asyncio.wait_for(tui.run(), timeout=5)
    assert state.rect == Rect(0.1, 0.2, 0.4, 0.6000000000000001) or (
        abs(state.rect.x0 - 0.1) < 1e-9 and abs(state.rect.width - 0.3) < 1e-9
    )


async def test_set_rect_rejects_zero_size() -> None:
    tui, state, console = build(["6", "0.5", "0.5", "0", "0", "q"])
    before = state.rect
    await asyncio.wait_for(tui.run(), timeout=5)
    assert state.rect == before
    assert "no size" in console.text


# --------------------------------------------------------------------------- #
# other actions
# --------------------------------------------------------------------------- #


async def test_send_reacji() -> None:
    tui, state, console = build(["7", "tada", "q"])
    await asyncio.wait_for(tui.run(), timeout=5)
    sent = state.multi.sessions[0].signaling.sent  # type: ignore[attr-defined]
    assert len(sent) == 1
    assert sent[0][0] == "data-channel"
    from huddle_drawings.peer_messages import parse_peer_message

    assert parse_peer_message(sent[0][1]).reacji.emoji == "tada"


async def test_watch_toggles_subscription() -> None:
    tui, state, console = build(["9", "9", "q"])
    await asyncio.wait_for(tui.run(), timeout=5)
    assert "watching peer messages" in console.text
    assert "stopped watching" in console.text
    assert not state.multi.sessions[0].signaling.callbacks  # type: ignore[attr-defined]


async def test_watch_ignores_our_own_echoed_messages() -> None:
    """Chime echoes our own data messages back; Slack clients ignore them."""
    from huddle_drawings.peer_messages import build_reacji
    from huddle_drawings.types import DataMessage

    tui, state, console = build(["9", "q"])
    task = asyncio.create_task(tui.run())
    await asyncio.sleep(0.05)
    signaling = state.multi.sessions[0].signaling  # type: ignore[attr-defined]
    cb = signaling.callbacks[0]
    cb(
        DataMessage(
            topic="data-channel",
            data=build_reacji(from_peer_id="att-0", emoji="mine"),
            sender_attendee_id="att-0",  # ourselves
        )
    )
    cb(
        DataMessage(
            topic="data-channel",
            data=build_reacji(from_peer_id="someone", emoji="theirs"),
            sender_attendee_id="someone-else",
        )
    )
    await asyncio.wait_for(task, timeout=5)
    assert "theirs" in console.text
    assert "mine" not in console.text


async def test_status_lists_sources() -> None:
    tui, state, console = build(["s", "q"])
    state.multi.sessions[0].signaling.sources = {  # type: ignore[attr-defined]
        11: SHARE
    }
    await asyncio.wait_for(tui.run(), timeout=5)
    assert "content share" in console.text
    assert "SHARER#content" in console.text


# --------------------------------------------------------------------------- #
# fill defaults
#
# Regression: the fill policy originally lived only in cli.py, so the console
# kept drawing hollow outlines after the command line was fixed. Both now go
# through image_trace.trace_options.
# --------------------------------------------------------------------------- #


async def test_trace_options_fill_by_default() -> None:
    tui, state, _console = build([])
    assert state.fill is None  # "use the per-source default"
    options = tui.trace_options(default_fill=True)
    assert options["mode"] == "fill"
    # A fill needs a small minimum run and a tight row height, or it comes out
    # striped and sparse.
    assert options["min_contour_length"] < 0.01
    assert options["fill_row_height"] <= 0.005


async def test_trace_options_honours_an_explicit_outline_choice() -> None:
    tui, state, _console = build([])
    state.fill = False
    assert tui.trace_options(default_fill=True)["mode"] != "fill"


async def test_trace_options_honours_an_explicit_fill_choice() -> None:
    tui, state, _console = build([])
    state.fill = True
    assert tui.trace_options(default_fill=False)["mode"] == "fill"


async def test_trace_options_carries_the_session_budget() -> None:
    tui, state, _console = build([])
    state.budget = 1234
    assert tui.trace_options(default_fill=True)["max_segments"] == 1234


async def test_fill_menu_sets_and_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    tui, state, console = build(["f", "outline", "f", "fill", "f", "auto", "q"])
    await asyncio.wait_for(tui.run(), timeout=5)
    assert state.fill is None  # ended on auto
    assert "now: outline" in console.text
    assert "now: fill" in console.text


async def test_fill_menu_rejects_nonsense() -> None:
    tui, state, console = build(["f", "banana", "q"])
    before = state.fill
    await asyncio.wait_for(tui.run(), timeout=5)
    assert state.fill is before
    assert "not one of fill/outline/auto" in console.text


async def test_status_line_shows_the_fill_mode() -> None:
    tui, state, console = build(["q"])
    state.fill = False
    await asyncio.wait_for(tui.run(), timeout=5)
    assert "| outline" in console.text

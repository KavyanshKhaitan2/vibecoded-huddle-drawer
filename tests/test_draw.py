"""Tests for :class:`draw.DrawClient`.

No network: a fake :class:`draw.DataMessageSender` records every payload, which
is then decoded with :func:`peer_messages.parse_peer_message` and asserted on as
a protobuf.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from huddle_drawings import config, geometry as g
from huddle_drawings.draw import (
    END_REPEAT_DELAY,
    HIDDEN_CURSOR,
    KEEP_ALIVE_INTERVAL,
    DataMessageSender,
    DrawClient,
)
from huddle_drawings.peer_messages import DrawType, parse_peer_message
from huddle_drawings.proto import peer_message_pb2 as pb
from huddle_drawings.types import StreamSource

ME = "att-me"
SHARER = "att-sharer"

pytestmark = pytest.mark.asyncio


class FakeSender:
    """Records what would have gone on the wire; can simulate rate limiting."""

    def __init__(self, *, accept: bool = True) -> None:
        self.calls: list[tuple[str, bytes, int, bool]] = []
        self.accept = accept
        #: Set to a count to drop exactly that many of the next sends.
        self.drop_next = 0

    async def send_data_message(
        self,
        topic: str,
        data: bytes,
        *,
        lifetime_ms: int = 0,
        wait: bool = True,
    ) -> bool:
        if self.drop_next > 0:
            self.drop_next -= 1
            return False
        if not self.accept:
            return False
        self.calls.append((topic, data, lifetime_ms, wait))
        return True

    # -- assertions helpers ---------------------------------------------

    @property
    def messages(self) -> list[pb.PeerMessage]:
        return [parse_peer_message(data) for _, data, _, _ in self.calls]

    @property
    def draws(self) -> list[pb.PeerMessage.Draw]:
        return [m.draw for m in self.messages if m.HasField("draw")]

    @property
    def types(self) -> list[str]:
        return [DrawType.Name(d.type) for d in self.draws]


def make_client(sender: FakeSender, *, hue: int | None = 200) -> DrawClient:
    c = DrawClient(sender, attendee_id=ME, item_id=SHARER, hue=hue)
    c.end_repeat_delay = 0.0  # keep the tests fast
    return c


SEGMENTS = [
    g.Segment.line((0.1, 0.1), (0.4, 0.2)),
    g.Segment.line((0.4, 0.2), (0.8, 0.5)),
]


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


async def test_fake_satisfies_the_protocol() -> None:
    assert isinstance(FakeSender(), DataMessageSender)


async def test_item_id_and_attendee_id_are_required() -> None:
    sender = FakeSender()
    with pytest.raises(ValueError, match="item_id"):
        DrawClient(sender, attendee_id=ME, item_id="")
    with pytest.raises(ValueError, match="attendee_id"):
        DrawClient(sender, attendee_id="", item_id=SHARER)


async def test_item_id_docstring_explains_the_silent_failure() -> None:
    doc = (DrawClient.__init__.__doc__ or "") + (DrawClient.__doc__ or "")
    assert "sharer's attendeeId" in doc
    assert "silently" in doc


async def test_for_share_strips_the_content_suffix() -> None:
    """INDEX lists a share as ``<sharerAttendeeId>#content``."""
    source = StreamSource(stream_id=3, attendee_id=f"{SHARER}#content")
    c = DrawClient.for_share(FakeSender(), attendee_id=ME, source=source)
    assert c.item_id == SHARER


async def test_set_item_id_resets_the_stroke_state() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.stroke(SEGMENTS, end=False)
    assert c.stroke_open
    c.set_item_id(f"{SHARER}-2#content")
    assert c.item_id == f"{SHARER}-2"
    assert not c.stroke_open
    assert c.last_segment is None


async def test_default_hue_is_a_valid_random_degree() -> None:
    c = DrawClient(FakeSender(), attendee_id=ME, item_id=SHARER)
    assert 0 <= c.hue <= 359


# --------------------------------------------------------------------------
# the stroke protocol: NEW degenerate, NEW append, EDIT replace, END
# --------------------------------------------------------------------------


async def test_stroke_starts_with_a_degenerate_new() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.stroke(SEGMENTS, end=False)

    assert sender.types == ["CHANGE_HUE", "NEW", "NEW", "NEW"]
    first = sender.draws[1].segment
    # The doc: "first point of a stroke: NEW with a degenerate segment (a=b=c=d)".
    assert (first.a.x, first.a.y) == (0.1, 0.1)
    for name in "abcd":
        point = getattr(first, name)
        assert (point.x, point.y) == (0.1, 0.1)

    # The appended segments are the real curves, in order.
    appended = sender.draws[2:]
    assert (appended[0].segment.a.x, appended[0].segment.d.x) == (0.1, 0.4)
    assert (appended[1].segment.a.x, appended[1].segment.d.x) == (0.4, 0.8)
    assert all(d.itemId == SHARER for d in sender.draws)
    assert all(m.fromPeerIdString == ME for m in sender.messages)


async def test_continuing_a_stroke_does_not_repeat_the_degenerate_dot() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.stroke(SEGMENTS, end=False)
    await c.stroke(SEGMENTS, end=False)
    assert sender.types == ["CHANGE_HUE"] + ["NEW"] * 5  # one degenerate, four curves


async def test_edit_replaces_the_last_segment() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.stroke(SEGMENTS, end=False)
    replacement = g.Segment.line((0.4, 0.2), (0.9, 0.9))
    await c.edit_last(replacement)

    assert sender.types[-1] == "EDIT"
    edited = sender.draws[-1].segment
    assert (edited.d.x, edited.d.y) == (0.9, 0.9)
    # The client now remembers the replacement, so keep-alives repeat it.
    assert (c.last_segment.d.x, c.last_segment.d.y) == (0.9, 0.9)


async def test_stroke_ends_with_two_ends() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.stroke(SEGMENTS)
    assert sender.types[-2:] == ["END", "END"]
    assert not c.stroke_open
    # END carries no segment, just the itemId.
    assert not sender.draws[-1].HasField("segment")


async def test_end_stroke_can_send_a_single_end() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.stroke(SEGMENTS, end=False)
    await c.end_stroke(twice=False)
    assert sender.types.count("END") == 1


async def test_the_two_ends_are_a_few_hundred_ms_apart() -> None:
    """A dropped END makes the line live ~8s instead of ~3.7s."""
    assert 0.1 <= END_REPEAT_DELAY <= 0.5
    sender = FakeSender()
    c = DrawClient(sender, attendee_id=ME, item_id=SHARER, hue=10)
    c.end_repeat_delay = 0.05
    started = time.monotonic()
    await c.end_stroke()
    assert time.monotonic() - started >= 0.05
    assert sender.types == ["END", "END"]


async def test_empty_stroke_sends_nothing() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.stroke([])
    assert sender.calls == []


async def test_draw_path_fits_a_curve_through_the_points() -> None:
    sender = FakeSender()
    c = make_client(sender)
    points = [g.Point(0.1, 0.1), g.Point(0.3, 0.5), g.Point(0.7, 0.2)]
    await c.draw_path(points, end=False)
    # CHANGE_HUE + degenerate + one segment per gap.
    assert sender.types == ["CHANGE_HUE", "NEW", "NEW", "NEW"]
    curves = [d.segment for d in sender.draws[2:]]
    assert (curves[0].a.x, curves[0].a.y) == (0.1, 0.1)
    assert (curves[-1].d.x, curves[-1].d.y) == (0.7, 0.2)


async def test_draw_path_with_one_point_draws_a_dot() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.draw_path([g.Point(0.5, 0.5)], end=False)
    assert sender.types == ["CHANGE_HUE", "NEW"]
    assert sender.draws[-1].segment.a.x == 0.5
    assert sender.draws[-1].segment.d.x == 0.5


async def test_draw_path_with_no_points_sends_nothing() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.draw_path([])
    assert sender.calls == []


async def test_protobuf_segments_are_accepted_too() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.stroke([g.to_proto_segment(SEGMENTS[0])], end=False)
    assert sender.types == ["CHANGE_HUE", "NEW", "NEW"]


async def test_everything_goes_on_the_data_channel_topic_with_lifetime_zero() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.stroke(SEGMENTS)
    await c.hide_cursor()
    for topic, _, lifetime_ms, wait in sender.calls:
        assert topic == config.TOPIC_DATA_CHANNEL
        assert lifetime_ms == 0
        assert wait is True


# --------------------------------------------------------------------------
# hue
# --------------------------------------------------------------------------


async def test_hue_is_sent_once_lazily_before_the_first_draw() -> None:
    sender = FakeSender()
    c = make_client(sender, hue=123)
    await c.stroke(SEGMENTS, end=False)
    await c.stroke(SEGMENTS, end=False)
    assert sender.types.count("CHANGE_HUE") == 1
    assert sender.types[0] == "CHANGE_HUE"
    assert sender.draws[0].hue == 123


async def test_set_hue_recolours_the_whole_line() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.stroke(SEGMENTS, end=False)
    await c.set_hue(300)
    assert c.hue == 300
    assert sender.types[-1] == "CHANGE_HUE"
    assert sender.draws[-1].hue == 300
    assert "whole current line" in (DrawClient.set_hue.__doc__ or "")


@pytest.mark.parametrize("hue", [-1, 360, 400, -360])
async def test_out_of_range_hue_is_rejected(hue: int) -> None:
    sender = FakeSender()
    with pytest.raises(ValueError, match="0..359"):
        DrawClient(sender, attendee_id=ME, item_id=SHARER, hue=hue)
    c = make_client(sender)
    with pytest.raises(ValueError, match="0..359"):
        await c.set_hue(hue)
    assert "CHANGE_HUE" not in sender.types


async def test_only_the_hue_is_controllable_is_documented() -> None:
    doc = (DrawClient.set_hue.__doc__ or "").lower()
    assert "hsla(hue, 100%, 75%)" in doc
    assert "dark, grey or white" in doc
    assert "accounts" in doc


# --------------------------------------------------------------------------
# cursor: POSITION
# --------------------------------------------------------------------------


async def test_move_cursor_sends_position_without_a_segment() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.move_cursor(0.25, 0.75)
    draw = sender.draws[-1]
    assert draw.type == DrawType.POSITION
    assert (draw.x, draw.y) == (0.25, 0.75)
    assert not draw.HasField("segment")
    assert draw.itemId == SHARER


async def test_hide_cursor_parks_it_far_off_screen() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.hide_cursor()
    draw = sender.draws[-1]
    assert draw.type == DrawType.POSITION
    assert (draw.x, draw.y) == HIDDEN_CURSOR == (-5.0, -5.0)


async def test_position_does_not_reset_the_fade_timer_but_edit_does() -> None:
    """The doc: POSITION "doesn't touch the line or its fade timer"."""
    sender = FakeSender()
    c = make_client(sender)
    await c.stroke(SEGMENTS, end=False)

    before = c.last_activity
    await asyncio.sleep(0.02)
    await c.move_cursor(0.5, 0.5)
    assert c.last_activity == before  # POSITION: no effect on the 5s timer
    assert c.seconds_until_keep_alive < c.keep_alive_interval

    await c.edit_last(SEGMENTS[-1])
    assert c.last_activity > before  # EDIT: timer reset
    assert c.seconds_until_keep_alive == pytest.approx(c.keep_alive_interval, abs=0.05)


async def test_position_does_not_open_a_stroke() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.move_cursor(0.5, 0.5)
    assert not c.stroke_open
    await c.stroke(SEGMENTS, end=False)
    assert sender.types == ["POSITION", "CHANGE_HUE", "NEW", "NEW", "NEW"]
    degenerate = sender.draws[2].segment  # the stroke still starts with a dot
    assert degenerate.a.x == degenerate.d.x == 0.1


# --------------------------------------------------------------------------
# keep-alive
# --------------------------------------------------------------------------


async def test_keep_alive_repeats_the_last_segment_as_an_edit() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.stroke(SEGMENTS, end=False)
    last = sender.draws[-1].segment

    await c.keep_alive()
    repeat = sender.draws[-1]
    assert repeat.type == DrawType.EDIT
    assert (repeat.segment.a.x, repeat.segment.a.y) == (last.a.x, last.a.y)
    assert (repeat.segment.d.x, repeat.segment.d.y) == (last.d.x, last.d.y)


async def test_keep_alive_with_nothing_drawn_is_a_no_op() -> None:
    sender = FakeSender()
    c = make_client(sender)
    await c.keep_alive()
    assert sender.calls == []


async def test_keep_alive_interval_stays_well_inside_five_seconds() -> None:
    """The fade starts 5s after the last non-POSITION message."""
    assert config.FADE_AFTER_IDLE == 5.0
    assert KEEP_ALIVE_INTERVAL < config.FADE_AFTER_IDLE
    # Headroom for one dropped keep-alive plus its retry backoff.
    assert 2 * KEEP_ALIVE_INTERVAL <= config.FADE_AFTER_IDLE
    assert DrawClient(FakeSender(), attendee_id=ME, item_id=SHARER).keep_alive_interval == (
        KEEP_ALIVE_INTERVAL
    )


async def test_keep_alive_task_sends_edits_on_cadence() -> None:
    sender = FakeSender()
    c = make_client(sender)
    c.keep_alive_interval = 0.05
    await c.stroke(SEGMENTS, end=False)
    baseline = len(sender.calls)

    c.start_keep_alive()
    c.start_keep_alive()  # idempotent
    await asyncio.sleep(0.28)
    await c.stop_keep_alive()

    edits = [d for d in sender.draws[baseline:] if d.type == DrawType.EDIT]
    assert 3 <= len(edits) <= 7  # ~0.28s / 0.05s, never faster than the cadence
    assert all(d.type == DrawType.EDIT for d in sender.draws[baseline:])

    # Nothing more arrives once it is stopped.
    stopped_at = len(sender.calls)
    await asyncio.sleep(0.12)
    assert len(sender.calls) == stopped_at


async def test_drawing_postpones_the_keep_alive() -> None:
    sender = FakeSender()
    c = make_client(sender)
    c.keep_alive_interval = 0.2
    await c.stroke(SEGMENTS, end=False)
    c.start_keep_alive()
    try:
        for _ in range(4):
            await asyncio.sleep(0.05)
            await c.edit_last(SEGMENTS[-1])
        # Every EDIT reset the timer, so the keep-alive never came due.
        assert all(
            d.type != DrawType.EDIT or d.segment.d.x == SEGMENTS[-1].d.x
            for d in sender.draws
        )
        assert c.seconds_until_keep_alive > 0
    finally:
        await c.stop_keep_alive()


async def test_keep_alive_retries_quickly_after_a_dropped_send() -> None:
    sender = FakeSender()
    c = make_client(sender)
    c.keep_alive_interval = 0.1
    await c.stroke(SEGMENTS, end=False)
    baseline = len(sender.calls)

    sender.drop_next = 1  # the first keep-alive is rate-limited away
    c.start_keep_alive()
    await asyncio.sleep(0.25)
    await c.stop_keep_alive()

    assert c.dropped == 1
    # The retry backoff is a fraction of the interval, so an EDIT still landed.
    assert len(sender.calls) > baseline


async def test_stop_keep_alive_without_starting_is_safe() -> None:
    c = make_client(FakeSender())
    await c.stop_keep_alive()


# --------------------------------------------------------------------------
# rate limiting
# --------------------------------------------------------------------------


async def test_rate_limited_sender_is_handled_not_raised() -> None:
    sender = FakeSender(accept=False)
    c = make_client(sender)
    await c.stroke(SEGMENTS)  # CHANGE_HUE + degenerate + 2 curves + 2 ENDs
    assert sender.calls == []
    assert c.sent == 0
    assert c.dropped == 6
    # Nothing reached the renderer, so no state is claimed.
    assert not c.stroke_open
    assert c.last_segment is None
    assert c.hue >= 0  # the hue is still remembered locally


async def test_counters_track_sends_and_drops() -> None:
    sender = FakeSender()
    c = make_client(sender)
    sender.drop_next = 2
    await c.stroke(SEGMENTS, end=False)
    assert c.dropped == 2
    assert c.sent == len(sender.calls)
    assert c.sent + c.dropped == 4


async def test_a_dropped_hue_is_retried_on_the_next_stroke() -> None:
    sender = FakeSender()
    c = make_client(sender, hue=42)
    sender.drop_next = 1
    await c.stroke(SEGMENTS, end=False)
    assert "CHANGE_HUE" not in sender.types  # it was dropped
    await c.stroke(SEGMENTS, end=False)
    assert sender.types.count("CHANGE_HUE") == 1
    assert sender.draws[-1].itemId == SHARER


# --------------------------------------------------------------------------
# fade / clear semantics
# --------------------------------------------------------------------------


async def test_wait_for_clear_defaults_to_the_documented_wait() -> None:
    c = make_client(FakeSender())
    assert c.line_clear_wait == config.LINE_CLEAR_WAIT == 8.5


async def test_wait_for_clear_waits_then_forgets_the_line() -> None:
    sender = FakeSender()
    c = make_client(sender)
    c.line_clear_wait = 0.05
    await c.stroke(SEGMENTS)
    started = time.monotonic()
    await c.wait_for_clear()
    assert time.monotonic() - started >= 0.05
    assert c.last_segment is None
    assert not c.stroke_open
    # A fresh picture re-announces the hue, since it is a new line.
    await c.stroke(SEGMENTS, end=False)
    assert sender.types.count("CHANGE_HUE") == 2


async def test_nothing_erases_a_line_early_is_documented() -> None:
    doc = (DrawClient.wait_for_clear.__doc__ or "").lower()
    assert "nothing erases a line early" in doc
    module_doc = (__import__("huddle_drawings.draw", fromlist=["x"]).__doc__ or "").lower()
    assert "desktop:screen" in module_doc

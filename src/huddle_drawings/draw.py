"""The drawing client: strokes, hue, cursor, keep-alives and fade bookkeeping.

:class:`DrawClient` turns geometry into ``PeerMessage.draw`` messages on topic
``data-channel``. It encodes the renderer rules from
``docs/huddles-peer-messages.md`` that are easy to get wrong:

* **``itemId`` must be the sharer's attendeeId.** Not yours, not the stream id.
  The renderer drops every draw whose ``itemId`` does not match the share it is
  drawing over, so a wrong value means the drawing silently never appears.
* **One line per attendee, one hue.** ``CHANGE_HUE`` recolours the sender's
  whole current line; the renderer always paints ``hsla(hue, 100%, 75%)``.
  Nothing can be drawn dark, grey or white, and N colours at once needs N
  accounts.
* **Nothing erases a line early.** Lines only fade: 0.7 s after ``END``, or 5 s
  after the last non-``POSITION`` message, then a 3 s fade. Any stroke sent
  before that fade finishes brings the *whole old line* back at full opacity,
  which is why :meth:`DrawClient.wait_for_clear` exists.
* **The sender must be a present attendee** with its audio leg up, or Slack
  clients drop its draws as coming from an unknown peer. That is ``audio.py``'s
  job, not this module's, but it is the first thing to check when nothing
  appears.
* **Drawing is only allowed on ``desktop:screen``** shares (and
  ``desktop:window`` on desktop builds >= 4.30.79). Browser and mobile shares
  cannot be drawn on at all -- see
  :data:`peer_messages.DRAWABLE_SOURCE_TYPES`.

Every message costs one token of the per-meeting budget (~150/s, ~200 burst,
shared by *all* senders), and over-budget messages are dropped silently. The
sender passed in is expected to be rate-limited; when it returns ``False`` the
message was dropped before it ever reached Chime, and this client counts that
in :attr:`DrawClient.dropped` rather than raising.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from . import config, geometry
from .geometry import Point, PointLike, Segment
from .peer_messages import (
    DrawType,
    ProtoSegment,
    SegmentLike,
    build_draw,
    coerce_segment,
    validate_hue,
)

logger = logging.getLogger(__name__)

__all__ = [
    "END_REPEAT_DELAY",
    "HIDDEN_CURSOR",
    "KEEP_ALIVE_INTERVAL",
    "DataMessageSender",
    "DrawClient",
]

#: How often :meth:`DrawClient.start_keep_alive` re-sends a no-op ``EDIT``.
#: The "vestigial" fade timer fires 5 s after the last non-``POSITION``
#: message, so this sits at half of that: well inside 5 s, with room for a
#: keep-alive to be dropped by the rate limiter and still make the next one in
#: time (the doc: keep-alives "count against the rate limit ... so leave
#: headroom").
KEEP_ALIVE_INTERVAL = config.FADE_AFTER_IDLE / 2.0

#: Gap between the two ``END`` messages. The doc: "Sending ``END`` twice (a few
#: hundred ms apart) and waiting ~8.5 s before reusing an account avoided every
#: 'old frame came back' glitch."
END_REPEAT_DELAY = 0.3

#: Where :meth:`DrawClient.hide_cursor` parks the pencil. The doc used -5,-5.
HIDDEN_CURSOR = (-5.0, -5.0)


@runtime_checkable
class DataMessageSender(Protocol):
    """Anything that can put a payload on a Chime data-message topic.

    ``signaling.ChimeSignaling`` satisfies this; tests use a fake. The return
    value is whether the message was actually sent: a rate-limited
    implementation with ``wait=False`` returns ``False`` instead of blocking.
    """

    async def send_data_message(
        self,
        topic: str,
        data: bytes,
        *,
        lifetime_ms: int = 0,
        wait: bool = True,
    ) -> bool: ...


class DrawClient:
    """Draws on one screen share, as one attendee, on one line, in one hue.

    Args:
        sender: the data-message transport (usually ``ChimeSignaling``).
        attendee_id: **our own** Chime attendeeId. Goes into
            ``fromPeerIdString`` on every message; Slack clients key the line
            by it and drop messages from attendees they have not seen in an
            ``AUDIO_STREAM_ID_INFO`` frame.
        item_id: **the sharer's attendeeId** -- the screen share appears in
            Chime's ``INDEX`` as ``<sharerAttendeeId>#content`` and this is that
            with ``#content`` stripped
            (:attr:`types.StreamSource.sharer_attendee_id`). It is required
            because there is no sensible default and no feedback: the renderer
            filters on it, so a wrong ``item_id`` means every drawing is
            silently discarded with no error anywhere. A share's attendeeId
            changes whenever the share restarts, so follow ``INDEX`` updates
            and call :meth:`set_item_id`.
        hue: HSL hue in degrees 0-359 for this attendee's line. Defaults to a
            random hue, like the real client (``drawHue =
            Math.floor(360*Math.random())``). It is sent as a ``CHANGE_HUE``
            lazily, just before the first draw.

    This class is not thread-safe, and a single instance assumes it owns the
    attendee's one line.
    """

    def __init__(
        self,
        sender: DataMessageSender,
        *,
        attendee_id: str,
        item_id: str,
        hue: int | None = None,
    ) -> None:
        if not attendee_id:
            raise ValueError("attendee_id is required (our own Chime attendeeId)")
        if not item_id:
            raise ValueError(
                "item_id is required and must be the sharer's attendeeId "
                "(INDEX's '<sharerAttendeeId>#content' without the suffix); "
                "a wrong itemId makes every drawing silently disappear"
            )
        self._sender = sender
        self._attendee_id = attendee_id
        self._item_id = item_id
        self._hue = validate_hue(hue) if hue is not None else random.randrange(360)
        self._hue_sent = False
        self._stroke_open = False
        self._last_segment: ProtoSegment | None = None
        self._last_activity = time.monotonic()
        self._keep_alive_task: asyncio.Task[None] | None = None

        #: Tunables, exposed so tests (and impatient callers) can shorten them.
        self.keep_alive_interval = KEEP_ALIVE_INTERVAL
        self.end_repeat_delay = END_REPEAT_DELAY
        self.line_clear_wait = config.LINE_CLEAR_WAIT

        #: Counters: messages handed to the sender, and messages the sender
        #: refused (rate limited) rather than sending.
        self.sent = 0
        self.dropped = 0

    # -- properties ------------------------------------------------------

    @property
    def attendee_id(self) -> str:
        return self._attendee_id

    @property
    def item_id(self) -> str:
        """The sharer's attendeeId these draws are addressed to."""
        return self._item_id

    @property
    def hue(self) -> int:
        """This line's hue, in degrees 0-359."""
        return self._hue

    @property
    def stroke_open(self) -> bool:
        """Whether a stroke has been started and not yet ``END``-ed."""
        return self._stroke_open

    @property
    def last_segment(self) -> ProtoSegment | None:
        """The last segment the renderer should hold, for ``EDIT`` no-ops."""
        return self._last_segment

    @property
    def last_activity(self) -> float:
        """``time.monotonic()`` of the last **non-POSITION** message sent.

        This is the clock the renderer's 5 s "vestigial" fade timer runs on, so
        ``POSITION`` deliberately does not update it.
        """
        return self._last_activity

    @property
    def seconds_until_keep_alive(self) -> float:
        """How long until a keep-alive is due. Can go negative (overdue)."""
        return self.keep_alive_interval - (time.monotonic() - self._last_activity)

    @classmethod
    def for_share(
        cls,
        sender: DataMessageSender,
        *,
        attendee_id: str,
        source: object,
        hue: int | None = None,
    ) -> DrawClient:
        """Build a client for a :class:`types.StreamSource` from ``INDEX``.

        Takes care of the ``#content`` strip, which is the single most common
        source of invisible drawings.
        """
        item_id = getattr(source, "sharer_attendee_id", None)
        if item_id is None:
            item_id = str(getattr(source, "attendee_id", source)).removesuffix(
                "#content"
            )
        return cls(sender, attendee_id=attendee_id, item_id=item_id, hue=hue)

    def set_item_id(self, item_id: str) -> None:
        """Point at a different share (its attendeeId changes on every restart)."""
        if not item_id:
            raise ValueError("item_id is required")
        self._item_id = item_id.removesuffix("#content")
        self._stroke_open = False
        self._last_segment = None

    # -- sending ---------------------------------------------------------

    async def _send(self, data: bytes, *, wait: bool = True) -> bool:
        """Hand one payload to the sender; log and count a rate-limited drop."""
        sent = await self._sender.send_data_message(
            config.TOPIC_DATA_CHANNEL, data, lifetime_ms=0, wait=wait
        )
        if sent:
            self.sent += 1
            return True
        self.dropped += 1
        logger.warning(
            "draw message dropped by the rate limiter (%d dropped so far); "
            "the per-meeting budget is ~150/s shared by every sender",
            self.dropped,
        )
        return False

    async def _send_draw(
        self,
        draw_type: DrawType.ValueType,
        *,
        segment: SegmentLike | None = None,
        hue: int | None = None,
        x: float | None = None,
        y: float | None = None,
        wait: bool = True,
    ) -> bool:
        data = build_draw(
            from_peer_id=self._attendee_id,
            draw_type=draw_type,
            item_id=self._item_id,
            segment=segment,
            hue=hue,
            x=x,
            y=y,
        )
        ok = await self._send(data, wait=wait)
        if not ok:
            # The renderer never saw it, so do not pretend it holds the segment
            # or that the fade timers were reset.
            return False
        if draw_type in (DrawType.NEW, DrawType.EDIT):
            self._stroke_open = True
            if segment is not None:
                self._last_segment = coerce_segment(segment)
        elif draw_type == DrawType.END:
            self._stroke_open = False
        if draw_type != DrawType.POSITION:
            # POSITION does not touch the line or either fade timer.
            self._last_activity = time.monotonic()
        return ok

    # -- hue -------------------------------------------------------------

    async def set_hue(self, hue: int) -> None:
        """Send ``CHANGE_HUE``, recolouring **the whole current line**.

        Hue is degrees 0-359; out-of-range values raise rather than being
        wrapped, because the renderer would clamp the resulting colour in a way
        that is hard to see. Only the hue is controllable: the renderer always
        draws ``hsla(hue, 100%, 75%)`` with an ``hsla(hue, 80%, 50%)`` shadow,
        so every colour is a pastel and nothing can be dark, grey or white.
        Changing hue mid-picture recolours what is already drawn too -- several
        colours on screen at once needs several *accounts*.
        """
        self._hue = validate_hue(hue)
        if await self._send_draw(DrawType.CHANGE_HUE, hue=self._hue):
            self._hue_sent = True

    async def _ensure_hue(self) -> None:
        if not self._hue_sent:
            await self.set_hue(self._hue)

    # -- strokes ---------------------------------------------------------

    async def stroke(self, segments: Sequence[Segment], *, end: bool = True) -> None:
        """Draw a stroke: one ``draw`` message per segment.

        Follows the renderer's stroke protocol:

        1. If no stroke is open, a ``NEW`` with a **degenerate** segment
           (``a=b=c=d``) at the first segment's start point -- that is what the
           real client sends for the first point of a stroke.
        2. One ``NEW`` per segment, each appending to the sender's line. The
           renderer does its own ``moveTo(a)`` per segment, so the segments do
           not have to connect.
        3. ``END`` (twice, see :meth:`end_stroke`) when ``end`` is true.

        Calling this again with ``end=False`` continues the same stroke and
        skips the degenerate dot. Accepts :class:`geometry.Segment` or protobuf
        ``Segment`` values.
        """
        segs = list(segments)
        if not segs:
            logger.debug("stroke() called with no segments")
            return
        await self._ensure_hue()
        if not self._stroke_open:
            first = _first_point(segs[0])
            await self._send_draw(DrawType.NEW, segment=Segment.degenerate(first))
        for segment in segs:
            await self._send_draw(DrawType.NEW, segment=segment)
        if end:
            await self.end_stroke()

    async def draw_path(self, points: Sequence[Point], *, end: bool = True) -> None:
        """Draw a smooth path through ``points`` (Catmull-Rom -> Béziers).

        The curve passes through every point given, costing one message per gap
        plus the degenerate first point. For long paths, simplify first
        (:func:`geometry.rdp`, or :func:`geometry.smooth_path`) -- a dropped
        point is a message saved against the ~150/s meeting budget.

        A single point draws one dot.
        """
        pts = [Point.of(p) for p in points]
        if not pts:
            return
        if len(pts) == 1:
            await self._ensure_hue()
            await self._send_draw(DrawType.NEW, segment=Segment.degenerate(pts[0]))
            if end:
                await self.end_stroke()
            return
        await self.stroke(geometry.catmull_rom_to_beziers(pts), end=end)

    async def edit_last(self, segment: Segment | ProtoSegment) -> None:
        """Send ``EDIT``, **replacing** the last segment of our line.

        Used to refine a stroke in flight (the real client emits ``EDIT`` while
        the smoothed segment's start point has not moved yet) and, with the
        same segment repeated, as a keep-alive no-op.
        """
        await self._send_draw(DrawType.EDIT, segment=segment)

    async def end_stroke(self, *, twice: bool = True) -> None:
        """Finish the stroke with ``END`` -- twice, by default.

        ``END`` starts a 700 ms timer, after which the line fades over 3 s and
        clears (~3.7 s in total). **If the ``END`` is dropped** -- and the rate
        limiter drops messages silently -- the line instead sits on the 5 s
        vestigial timer and lives ~8 s, which is how an "old frame" comes back
        on top of the next picture. So it is sent twice, ``end_repeat_delay``
        apart, exactly as the doc recommends. The second ``END`` is harmless if
        the first arrived.
        """
        await self._send_draw(DrawType.END)
        if twice:
            await asyncio.sleep(self.end_repeat_delay)
            await self._send_draw(DrawType.END)

    # -- cursor ----------------------------------------------------------

    async def move_cursor(self, x: float, y: float) -> None:
        """Send ``POSITION``: move our pencil cursor and name tag only.

        It does not add to the line and **does not reset either fade timer**, so
        a stream of ``POSITION`` messages will not keep a drawing alive -- use
        :meth:`keep_alive` for that. Coordinates are normalized 0..1; a NaN
        point parks the name tag in the top-left corner.
        """
        await self._send_draw(DrawType.POSITION, x=float(x), y=float(y))

    async def hide_cursor(self) -> None:
        """Hide the pencil cursor and name tag by parking them off-screen.

        Sends ``POSITION`` at ``-5,-5`` (the value the doc used). The cursor
        stays hidden until the next ``NEW``/``EDIT`` moves it back.
        """
        await self.move_cursor(*HIDDEN_CURSOR)

    # -- keep-alive / fade bookkeeping -----------------------------------

    async def keep_alive(self) -> None:
        """Re-send the last segment as a no-op ``EDIT`` to defer the fade.

        ``EDIT`` replaces the last segment with itself: nothing changes in the
        picture, but the renderer's 5 s vestigial timer is reset. This is the
        only way to hold a drawing on screen -- there is no "keep" message.
        Does nothing if no segment has been sent yet.

        This holds up a stroke that has **not** been ``END``-ed. ``END`` starts
        a separate 700 ms timer, and the doc does not say whether a later
        ``EDIT`` cancels that one (only ``NEW`` is documented as cancelling a
        fade), so do not count on keeping an ended line alive -- draw with
        ``end=False`` if you intend to keep it up.
        """
        if self._last_segment is None:
            logger.debug("keep_alive() with nothing drawn yet; nothing to repeat")
            return
        await self._send_draw(DrawType.EDIT, segment=self._last_segment)

    def start_keep_alive(self) -> None:
        """Start a background task that keeps the line from fading.

        Fires a :meth:`keep_alive` whenever ``keep_alive_interval`` (2.5 s --
        half of the 5 s vestigial timer) has passed since the last
        non-``POSITION`` message, so ordinary drawing postpones it and
        ``POSITION`` does not. The margin is deliberate: a keep-alive can be
        dropped by the rate limiter and the next one still lands inside 5 s.

        Pair this with strokes drawn ``end=False``: once ``END`` has gone out
        the 700 ms timer takes over and the line is on its way out. And note
        this keeps the line alive, it does not stop a fade already running --
        nothing does, and the next stroke brings the whole old line back at
        full opacity.
        """
        if self._keep_alive_task is not None and not self._keep_alive_task.done():
            return
        self._keep_alive_task = asyncio.create_task(
            self._keep_alive_loop(), name=f"draw-keepalive-{self._attendee_id}"
        )

    async def stop_keep_alive(self) -> None:
        """Cancel the keep-alive task and wait for it to finish."""
        task = self._keep_alive_task
        self._keep_alive_task = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _keep_alive_loop(self) -> None:
        while True:
            delay = self.seconds_until_keep_alive
            if delay > 0:
                await asyncio.sleep(delay)
                continue
            if self._last_segment is None:
                await asyncio.sleep(self.keep_alive_interval)
                continue
            await self.keep_alive()
            if self.seconds_until_keep_alive <= 0:
                # The send was dropped, so the timer did not move. Back off a
                # little rather than spinning on a full rate limiter.
                await asyncio.sleep(min(0.5, self.keep_alive_interval / 4))

    async def wait_for_clear(self) -> None:
        """Wait until our line is certainly gone (~8.5 s).

        ``config.LINE_CLEAR_WAIT``. The doc: a dropped ``END`` leaves a line
        alive ~8 s instead of ~3.7 s, and "sending ``END`` twice (a few hundred
        ms apart) and waiting ~8.5 s before reusing an account avoided every
        'old frame came back' glitch". **Nothing erases a line early**, so this
        wait is the only way to start a fresh picture from the same account --
        which is why a flipbook needs about one account per frame per 8 s.
        """
        await asyncio.sleep(self.line_clear_wait)
        self._stroke_open = False
        self._last_segment = None
        self._hue_sent = False


def _first_point(segment: Segment | ProtoSegment | PointLike) -> Point:
    start = getattr(segment, "a", None)
    return Point.of(start if start is not None else segment)

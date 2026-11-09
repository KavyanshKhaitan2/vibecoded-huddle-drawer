"""Play a video onto a huddle screen share as a **flipbook**.

One frame = one traced line drawing = one attendee's line. Decode the video at
a low target fps, trace each frame to cubic Béziers
(:mod:`huddle_drawings.image_trace`), and draw each frame from a *different*
attendee, rotating through them.

Everything about the design is forced by three facts from
``docs/huddles-peer-messages.md``, none of which this module gets to choose:

* **A line cannot be erased, only waited out.** After ``END`` it starts fading
  0.7 s later and fades for 3 s; a dropped ``END`` leaves it alive ~8 s. And
  "any stroke before the fade has finished brings the whole old line back at
  full opacity" -- the doc found that "the hard way while playing video as a
  flipbook". So a frame can never be drawn from an attendee that drew a recent
  frame: it needs a *fresh* one, and an attendee is only reusable
  :data:`config.LINE_CLEAR_WAIT` (8.5 s, the doc's measured-safe interval)
  after its ``END``. :func:`play` enforces that with a per-attendee
  ``free_at``, and when nothing is free it **skips the frame** rather than
  resurrecting the previous picture.
* **Attendees needed = frames inside one cooldown window, per colour layer**
  (:func:`attendees_needed`): the doc's "frames per second × ~8 s = accounts
  needed (plus one per extra colour layer per frame)". They do not have to be
  separate Slack accounts -- ``rooms.join`` with ``multidevice=true`` returns a
  fresh AttendeeId per call, so ``MultiSession.join_all(accounts, count=N)``
  supplies N distinct attendees from one login (9 in 3.1 s, live-verified
  2026-11-08).
* **The ~150 messages/s budget is per *meeting*,** shared by every attendee,
  and excess is dropped **silently**. So a frame's segment budget is
  ``rate / fps`` with headroom (:func:`segments_per_frame`), and the bucket
  that enforces it is ``MultiSession.limiter`` -- the one every session was
  joined with. This module never creates a second one; a second bucket would
  let the meeting be overrun by exactly the factor nobody can see.

Chime also drops an attendee whose audio leg stops, so :func:`play` never
closes or re-joins a session: the whole pool stays joined for the entire
playback.

What this actually looks like, honestly
---------------------------------------

* **A frame is emoji-level detail.** At 1 fps a frame gets about 120 segments
  (``150 × 0.8 / 1``), which is a logo, an icon, a silhouette or a couple of
  letters -- not a picture of a room. At 2 fps it is ~60. :mod:`image_trace`
  degrades gracefully into that budget (coarser curves, then fewer contours),
  so what you get is a rough whole frame rather than a crisp corner of one,
  but it is rough.
* **It reads as a slideshow, not as motion.** At 1-2 fps nothing in the frame
  appears to move; successive stills just replace each other, and each one
  arrives by being *drawn* over ~0.5-1 s rather than appearing at once. Video
  with camera motion or cuts turns into unrelated scribbles. What works is
  something that is already a few solid shapes moving slowly.
* **Every attendee is a visible participant.** A 2 fps flipbook needs ~18
  attendees, so the huddle gains ~18 entries in its participant list, each
  with an audio leg, a pencil cursor and a name tag parked wherever its line
  ended. That is the real cost of this feature, and it is not subtle to anyone
  else in the huddle. 1 fps halves it to 9. Nobody should run this in a huddle
  whose participants have not agreed to it.
* **Pacing is hostage to the budget.** 1 fps × 120 segments is 120 messages/s
  against a meeting limit of ~150 shared with everyone else, so a flipbook
  leaves almost no room for another drawer, and anything over the limit
  disappears with no error. The per-frame cost is also slightly more than its
  segments: each layer's stroke opens with a degenerate ``NEW`` and closes
  with two ``END``s.
* **Frames get skipped when the pool is short.** With fewer attendees than
  ``fps × 8.5``, :func:`play` draws what it can and skips the rest -- playback
  stays in time with the wall clock and the drawn frames are clean, but the
  effective frame rate is ``attendees / 8.5``, whatever ``fps`` says.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import av
from PIL import Image

from . import config, image_trace
from .config import (
    CHIME_DATA_MESSAGE_RATE,
    CLIENT_DATA_MESSAGE_RATE,
    LINE_CLEAR_WAIT,
)
from .draw import END_REPEAT_DELAY
from .geometry import Rect
from .multi import Layer, MultiSession
from .types import HuddleError, StreamSource

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_HEADROOM",
    "DEFAULT_HUE",
    "DEFAULT_SIZE",
    "Frame",
    "PlaybackStats",
    "attendees_needed",
    "decode_frames",
    "play",
    "segments_per_frame",
    "trace_frames",
]

#: Longest side the decoder scales frames down to before tracing. The tracer
#: downscales again to :data:`image_trace.DEFAULT_MAX_DIMENSION` (512), so this
#: only has to be big enough not to lose detail the 7 px pen could have shown;
#: smaller frames make the whole decode -> trace pipeline cheaper, which
#: matters when it has to keep up with a frame deadline.
DEFAULT_SIZE = 384

#: Fraction of the meeting's message budget a flipbook aims to use. The rest is
#: for everyone else in the huddle -- and for this module's own per-frame
#: overhead (a degenerate ``NEW`` and two ``END``s per layer per frame). It is
#: deliberately the *same* knob ``MultiSession.join_all`` sizes its shared
#: TokenBucket from, so :func:`segments_per_frame` predicts what the limiter
#: will actually pass; forking the two would make the budget a frame is given
#: differ from the budget it can spend, and the difference would vanish
#: silently mid-frame.
DEFAULT_HEADROOM = config.DATA_MESSAGE_HEADROOM

#: Hue for a single-colour flipbook. Arbitrary, like every hue here -- the
#: renderer forces ``hsla(hue, 100%, 75%)``, so nothing can be drawn dark, grey
#: or white -- but a cool blue is the least assertive pastel available.
DEFAULT_HUE = image_trace.DEFAULT_GREY_HUE

#: Timestamp slop, in seconds, when matching decoded frames to wanted ones.
_EPS = 1e-9

# Patched by tests to run playback on a virtual clock. Everything in :func:`play`
# reads the wall clock through these two names.
_monotonic = time.monotonic


async def _wait(delay: float) -> None:
    await asyncio.sleep(delay)


# --------------------------------------------------------------------------
# the arithmetic
# --------------------------------------------------------------------------


def attendees_needed(
    fps: float, *, colors: int = 1, clear_wait: float = LINE_CLEAR_WAIT
) -> int:
    """How many attendees a flipbook at ``fps`` needs, with ``colors`` layers.

    The doc's rule -- "frames per second × ~8 s = accounts needed (plus one per
    extra colour layer per frame)" -- counted exactly: an attendee that ``END``s
    a frame at ``t`` is not reusable until ``t + clear_wait``, so every frame
    whose deadline falls in that closed window needs its own attendee. At
    ``fps`` that is ``floor(fps · clear_wait) + 1`` of them, times one attendee
    per colour layer (Slack keys lines by attendee, so N colours in one frame
    are N lines from N attendees).

    The ``+ 1`` is the frame that lands *exactly* on the boundary: at 2 fps and
    8.5 s the frames at 0.0 … 8.5 inclusive are 18, and giving the boundary
    frame a fresh attendee is what leaves the first one a half-second of slack
    instead of none. So 1 fps -> 9, 2 fps -> 18, 4 fps -> 35, and a 2-colour
    2 fps flipbook -> 36.

    ``MultiSession.join_all(accounts, count=attendees_needed(fps))`` is the
    intended way to get them; they need not be distinct logins.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    if colors < 1:
        raise ValueError(f"colors must be at least 1, got {colors!r}")
    if clear_wait <= 0:
        raise ValueError(f"clear_wait must be positive, got {clear_wait!r}")
    per_frame_slots = math.floor(float(fps) * float(clear_wait)) + 1
    return per_frame_slots * int(colors)


def segments_per_frame(
    fps: float,
    *,
    rate: float = CHIME_DATA_MESSAGE_RATE,
    headroom: float = DEFAULT_HEADROOM,
) -> int:
    """The segment budget for one frame: ``rate · headroom / fps``.

    One segment is one Chime data message and the whole *meeting* shares about
    ``rate`` of them per second, so a flipbook's fidelity is bought entirely out
    of its frame rate: 1 fps gets ~135 segments, 2 fps ~67, 5 fps ~27. The
    budget covers **all** of a frame's colour layers together, for the same
    reason ``image_trace.trace_image_colors`` treats ``max_segments`` as a total:
    more attendees buy colours, not throughput.

    ``rate`` is the *meeting's* whole limit and ``headroom`` is the share of it
    this client takes (90% by default), leaving room for everyone else in the
    huddle and for this module's per-frame overhead; over-budget messages are
    dropped silently, so aiming at the full 150 would quietly lose segments out
    of the middle of frames. Never returns less than 1 -- a frame with no
    segments is not a frame -- but at that point the fps is far past what this
    can honestly draw.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate!r}")
    if not 0.0 < headroom <= 1.0:
        raise ValueError(f"headroom must be in (0, 1], got {headroom!r}")
    return max(1, int(float(rate) * float(headroom) / float(fps)))


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------


def decode_frames(
    path: str | Path,
    *,
    fps: float,
    max_frames: int | None = None,
    start: float = 0.0,
    duration: float | None = None,
    size: int = DEFAULT_SIZE,
) -> Iterator[Image.Image]:
    """Decode ``path`` into RGB :class:`PIL.Image.Image` frames at ``fps``.

    Seeks to ``start``, then **resamples** by picking, for each wanted timestamp
    ``start + k/fps``, the decoded frame whose own timestamp is nearest to it.
    That is the only resampling that is right for a variable-frame-rate source
    (where "every n-th frame" means nothing) and it is also what makes the
    output count predictable: ``fps`` frames per second of source, regardless of
    the source's own rate. Asking for more than the source has duplicates the
    nearest frame rather than inventing one.

    Decoding is a single forward pass -- a video codec cannot skip to an
    arbitrary frame without decoding the ones before it -- but only the
    *selected* frames are converted to RGB and scaled, which is where the cost
    actually is. The long side is scaled down to ``size`` (never up).

    ``duration`` is measured from ``start``, and a wanted timestamp at or past
    ``start + duration`` ends the iteration; a source shorter than ``duration``
    simply ends early. ``max_frames`` caps the count.

    Decoding uses **av** (already a dependency, via aiortc). Raises
    :class:`HuddleError`, naming the path, if the file cannot be opened or has
    no video stream. Note that ffmpeg's idea of "a video" is broader than
    yours: a plain-text file demuxes as an ANSI-art video and will decode to
    frames of rendered text. A file that decodes to **zero** frames (``start``
    past the end, say) is not an error -- the iterator is simply empty, with a
    warning logged.

    The returned iterator owns an open container and closes it when exhausted,
    so consume it (or ``close()`` it) rather than dropping it on the floor.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start!r}")
    if duration is not None and duration <= 0:
        raise ValueError(f"duration must be positive, got {duration!r}")
    if max_frames is not None and max_frames < 0:
        raise ValueError(f"max_frames must be >= 0, got {max_frames!r}")
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size!r}")

    source = Path(path)
    try:
        container = av.open(str(source))
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise HuddleError(f"cannot open {source} as a video: {exc}") from exc

    try:
        streams = container.streams.video
    except Exception as exc:  # noqa: BLE001 - defensive: malformed container
        with contextlib.suppress(Exception):
            container.close()
        raise HuddleError(f"cannot read video streams from {source}: {exc}") from exc

    if not streams:
        kinds = sorted({s.type for s in container.streams}) or ["none"]
        with contextlib.suppress(Exception):
            container.close()
        raise HuddleError(
            f"{source} has no video stream (found: {', '.join(kinds)})"
        )

    stream = streams[0]
    with contextlib.suppress(Exception):
        # Frame-threaded decoding: this pass has to keep up with a frame
        # deadline while the event loop also runs every attendee's audio leg.
        stream.thread_type = "AUTO"

    return _iter_frames(
        container,
        stream,
        source=source,
        fps=float(fps),
        max_frames=max_frames,
        start=float(start),
        duration=duration,
        size=int(size),
    )


def _iter_frames(
    container: Any,
    stream: Any,
    *,
    source: Path,
    fps: float,
    max_frames: int | None,
    start: float,
    duration: float | None,
    size: int,
) -> Iterator[Image.Image]:
    step = 1.0 / fps
    limit = None if duration is None else start + duration
    emitted = 0
    try:
        if max_frames is not None and max_frames <= 0:
            return
        if start > 0:
            _seek(container, stream, start, source)
        base: float | None = None
        wanted = 0
        previous: tuple[float, Any] | None = None
        decoded = 0
        for raw in container.decode(stream):
            timestamp = _frame_time(raw, decoded, stream, fps)
            decoded += 1
            if timestamp < start - _EPS:
                # Seeking lands on the keyframe at or before `start`.
                continue
            if base is None:
                base = max(start, timestamp)
            while True:
                target = base + wanted * step
                if limit is not None and target >= limit - _EPS:
                    return
                if target > timestamp + _EPS:
                    break
                chosen = raw
                if previous is not None and abs(target - previous[0]) <= abs(
                    timestamp - target
                ):
                    chosen = previous[1]
                yield _to_image(chosen, size)
                emitted += 1
                wanted += 1
                if max_frames is not None and emitted >= max_frames:
                    return
            previous = (timestamp, raw)
    except (av.FFmpegError, OSError) as exc:
        raise HuddleError(f"failed decoding {source}: {exc}") from exc
    finally:
        if not emitted and max_frames != 0:
            log.warning(
                "decoded no frames from %s (start=%.3fs%s): nothing to play",
                source,
                start,
                "" if duration is None else f", duration={duration:.3f}s",
            )
        with contextlib.suppress(Exception):
            container.close()


def _seek(container: Any, stream: Any, start: float, source: Path) -> None:
    """Seek to ``start`` seconds, tolerating a container that cannot."""
    try:
        offset = int(start / stream.time_base)
        container.seek(offset, stream=stream)
    except Exception as exc:  # noqa: BLE001 - seeking is best-effort
        log.warning(
            "cannot seek %s to %.3fs (%s); decoding from the beginning instead",
            source,
            start,
            exc,
        )


def _frame_time(raw: Any, index: int, stream: Any, fps: float) -> float:
    """A frame's presentation time in seconds, with a fallback for no pts.

    Some containers hand over frames with no timestamp at all. Counting them
    against the stream's average rate is a guess, but it is a better one than
    treating every frame as time zero.
    """
    seconds = getattr(raw, "time", None)
    if seconds is not None:
        return float(seconds)
    rate = getattr(stream, "average_rate", None)
    return index / float(rate or fps)


def _to_image(raw: Any, size: int) -> Image.Image:
    """One decoded frame as an RGB PIL image, long side at most ``size``."""
    width, height = int(raw.width), int(raw.height)
    longest = max(width, height)
    if longest > size:
        scale = size / float(longest)
        return raw.to_image(
            width=max(1, round(width * scale)),
            height=max(1, round(height * scale)),
        )
    return raw.to_image()


# --------------------------------------------------------------------------
# tracing
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Frame:
    """One traced frame, ready to send: ``layers`` is what goes on the wire.

    One :class:`multi.Layer` per colour, which is one attendee's line; the
    single-colour case is one layer. ``segments`` is the total message cost of
    the frame across all its layers (one segment = one Chime data message), so
    it is directly comparable with :func:`segments_per_frame`.

    A frame can legitimately have **no** layers: a uniform or near-blank image
    traces to nothing at all. :func:`play` counts those separately rather than
    burning an attendee's cooldown on an empty line.
    """

    index: int
    layers: list[Layer]
    segments: int

    @classmethod
    def of(cls, index: int, layers: Sequence[Layer]) -> Frame:
        kept = [layer for layer in layers if len(layer.segments)]
        return cls(
            index=index,
            layers=kept,
            segments=sum(len(layer.segments) for layer in kept),
        )

    def __len__(self) -> int:
        return self.segments

    def __bool__(self) -> bool:
        return bool(self.layers)


def trace_frames(
    frames: Iterable[Image.Image],
    rect: Rect | None = None,
    *,
    colors: int = 1,
    max_segments: int | None = None,
    fill: bool = True,
    hue: int = DEFAULT_HUE,
    mode: str = "auto",
    invert: bool = False,
    **kw: Any,
) -> Iterator[Frame]:
    """Trace decoded frames into :class:`Frame` values, lazily.

    ``max_segments`` is the **per-frame** budget and should be
    ``segments_per_frame(fps)``; it defaults to
    :data:`image_trace.DEFAULT_MAX_SEGMENTS`, which is a fine budget for a
    still and a hopeless one for a flipbook (4000 segments is ~27 s of
    messages, i.e. 27 frames' worth of budget spent on one frame).

    ``fill``, ``mode``, ``invert`` and any extra keywords go to
    :func:`image_trace.trace_options` and then to the tracer, so a frame is
    traced exactly as a still image would be. ``colors > 1`` runs
    :func:`image_trace.trace_image_colors` instead and yields one layer per
    surviving colour -- which multiplies the attendees needed per frame, hence
    ``colors`` in :func:`attendees_needed`.

    Generating is lazy and stays lazy: :func:`play` pulls one frame at a time
    so that decoding and tracing overlap with the previous frame's sending
    (tracing is tens to hundreds of milliseconds, well inside a 1 fps frame).
    """
    if colors < 1:
        raise ValueError(f"colors must be at least 1, got {colors!r}")
    budget = (
        image_trace.DEFAULT_MAX_SEGMENTS if max_segments is None else int(max_segments)
    )
    if budget < 0:
        raise ValueError(f"max_segments must be >= 0, got {max_segments!r}")
    options = image_trace.trace_options(
        fill=fill, mode=mode, invert=invert, max_segments=budget
    )
    options.update(kw)
    return _iter_traced(frames, rect, colors=colors, hue=hue, options=options)


def _iter_traced(
    frames: Iterable[Image.Image],
    rect: Rect | None,
    *,
    colors: int,
    hue: int,
    options: dict[str, Any],
) -> Iterator[Frame]:
    for index, image in enumerate(frames):
        if colors == 1:
            result = image_trace.trace_image(image, rect, **options)
            layers = [Layer(hue=hue, segments=result.segments)]
        else:
            layers = [
                Layer(hue=layer.hue, segments=layer.segments)
                for layer in image_trace.trace_image_colors(
                    image, rect, layers=colors, **options
                )
            ]
        frame = Frame.of(index, layers)
        log.debug(
            "frame %d traced: %d segment(s) in %d layer(s)",
            frame.index,
            frame.segments,
            len(frame.layers),
        )
        yield frame


# --------------------------------------------------------------------------
# playback
# --------------------------------------------------------------------------


@dataclass
class PlaybackStats:
    """What a :func:`play` run actually did.

    ``frames_skipped`` is the number that matters: a skipped frame is one that
    came up while every attendee was still inside its
    :data:`config.LINE_CLEAR_WAIT` cooldown. Skipping is deliberate -- drawing
    early would bring the *previous* frame back at full opacity on top of the
    new one -- so a large ``frames_skipped`` means the attendee pool is too
    small for the frame rate, not that anything failed.

    ``segments_dropped`` is the other silent failure mode: segments the shared
    rate limiter refused because the meeting's ~150 messages/s was already
    spent. Those never reach Chime at all.
    """

    fps: float = 0.0
    attendees: int = 0
    attendees_needed: int = 0
    frames_drawn: int = 0
    frames_skipped: int = 0
    frames_empty: int = 0
    segments_sent: int = 0
    segments_dropped: int = 0
    seconds: float = 0.0
    loops: int = 0
    stopped: bool = False

    @property
    def frames_total(self) -> int:
        """Every frame that came up, drawn, skipped or empty."""
        return self.frames_drawn + self.frames_skipped + self.frames_empty

    @property
    def effective_fps(self) -> float:
        """Frames actually drawn per second of playback.

        Over the **scheduled** length of what was played (``frames_total /
        fps``) or the wall clock, whichever is longer. Dividing by the wall
        clock alone would flatter a short clip -- three frames at 1 fps are
        drawn at t=0, 1 and 2, so playback "ends" after 2 s and the arithmetic
        would claim 1.5 fps -- while dividing by the schedule alone would
        ignore a run that fell behind.
        """
        if not self.frames_total or self.fps <= 0:
            return 0.0
        span = max(self.seconds, self.frames_total / self.fps)
        return self.frames_drawn / span if span > 0 else 0.0

    def describe(self) -> str:
        """One line for logs."""
        parts = [
            f"{self.frames_drawn} frame(s) drawn",
            f"{self.segments_sent} segments",
            f"{self.seconds:.1f}s",
            f"{self.effective_fps:.2f} fps effective (asked {self.fps:g})",
            f"{self.attendees} attendee(s), {self.attendees_needed} needed",
        ]
        if self.frames_skipped:
            parts.append(
                f"{self.frames_skipped} skipped (no attendee out of cooldown)"
            )
        if self.frames_empty:
            parts.append(f"{self.frames_empty} traced to nothing")
        if self.segments_dropped:
            parts.append(f"{self.segments_dropped} segments dropped by the limiter")
        if self.loops > 1:
            parts.append(f"{self.loops} loop(s)")
        if self.stopped:
            parts.append("STOPPED")
        return ", ".join(parts)


@dataclass(slots=True)
class _Pool:
    """The attendees, and when each one's line is certainly gone.

    ``free_at`` is ``END`` time + ``clear_wait``: nothing erases a line early,
    so this timestamp is the only thing standing between a flipbook and the old
    frame coming back at full opacity on top of the new one.
    """

    sessions: list[Any]
    clear_wait: float
    free_at: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.free_at = [float("-inf")] * len(self.sessions)

    def take(self, count: int, now: float) -> list[int] | None:
        """Indices of ``count`` attendees free since longest, or ``None``.

        ``None`` means *skip this frame*. Returning a still-cooling attendee
        instead would redraw the previous frame at full opacity, which is the
        exact glitch the doc warns about.
        """
        free = sorted(
            (index for index, free_at in enumerate(self.free_at) if free_at <= now),
            key=lambda index: (self.free_at[index], index),
        )
        if len(free) < count:
            return None
        return free[:count]

    def used(self, indices: Iterable[int], ended_at: float) -> None:
        for index in indices:
            # The second END lands END_REPEAT_DELAY later, and the cooldown
            # runs from the last message on that line.
            self.free_at[index] = ended_at + END_REPEAT_DELAY + self.clear_wait


async def play(
    multi: MultiSession,
    frames: Iterable[Frame],
    share: StreamSource | str,
    *,
    fps: float,
    colors: int = 1,
    loop: bool = False,
    stop: asyncio.Event | None = None,
    on_frame: Callable[[Frame, float], None] | None = None,
    clear_wait: float = LINE_CLEAR_WAIT,
) -> PlaybackStats:
    """Draw ``frames`` onto ``share`` as a flipbook, one attendee per frame.

    ``multi`` supplies the attendee pool and -- importantly -- the single
    shared :class:`ratelimit.TokenBucket` every one of its sessions was joined
    with. Chime's ~150 messages/s is per *meeting*, so that one bucket is what
    keeps a flipbook inside it; nothing here creates another. The sessions are
    left joined for the whole playback and are never closed between frames:
    Chime drops an attendee whose audio leg stops.

    For every frame, the attendees that have been out of cooldown longest are
    taken, the frame's layers are interleaved onto their lines
    (``MultiSession._interleave_order``, so a multi-colour frame's colours
    arrive together), and every line is ``END``-ed immediately -- there is no
    hold, because the next frame is due and the cooldown cannot start until the
    ``END`` goes out. ``END`` is sent twice, the repeat in the background, so
    the doc's "send END twice" advice does not cost 0.3 s of every frame's
    deadline.

    **If no attendee is free the frame is skipped** and counted in
    :attr:`PlaybackStats.frames_skipped`. That is the whole point: drawing from
    an attendee whose previous line has not finished fading brings that old
    frame back at full opacity.

    Pacing accumulates the next deadline (``t0 + n/fps``) instead of sleeping
    ``1/fps`` per frame, so the time spent decoding, tracing and sending does
    not make playback drift ever further behind. If it falls more than a whole
    frame behind it resyncs to now and says so, rather than trying to repay a
    debt it cannot.

    ``stop`` abandons playback between frames and between segments within a
    frame; what is already drawn stays (nothing erases it) and the lines are
    still ``END``-ed, so the last frame fades normally. ``loop=True`` repeats
    until ``stop`` is set, which requires holding every traced frame in memory
    (segments only, so a few hundred KB per minute of flipbook).

    ``on_frame(frame, elapsed)`` is called after each frame is drawn, with the
    seconds since playback started -- for a progress line. Exceptions from it
    are logged and ignored.

    ``clear_wait`` is the per-attendee cooldown and defaults to
    :data:`config.LINE_CLEAR_WAIT` (8.5 s), the interval the doc measured as
    avoiding "every 'old frame came back' glitch". Shortening it is how to
    trade flicker risk for frame rate; there is no value at which the renderer
    forgives you, only one that was never observed to fail.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    if colors < 1:
        raise ValueError(f"colors must be at least 1, got {colors!r}")
    sessions = list(multi.sessions)
    if not sessions:
        raise HuddleError(
            "no sessions joined, so there is nobody to draw a frame from; "
            f"a {fps:g} fps flipbook needs "
            f"{attendees_needed(fps, colors=colors, clear_wait=clear_wait)} "
            "attendees (MultiSession.join_all(..., count=N))"
        )

    needed = attendees_needed(fps, colors=colors, clear_wait=clear_wait)
    stats = PlaybackStats(
        fps=float(fps), attendees=len(sessions), attendees_needed=needed
    )
    limiter = getattr(multi, "limiter", None)
    log.info(
        "flipbook at %g fps: %d attendee(s), %d needed (%g fps x %.1fs cooldown "
        "x %d colour(s)), ~%d segments per frame against our %.0f/s share of "
        "the meeting's budget",
        fps,
        len(sessions),
        needed,
        fps,
        clear_wait,
        colors,
        segments_per_frame(fps),
        getattr(limiter, "rate", CLIENT_DATA_MESSAGE_RATE),
    )
    if len(sessions) < needed:
        log.warning(
            "only %d attendee(s) for %g fps: expect about %d of every %d "
            "frames to be skipped (an attendee cannot draw again for %.1fs "
            "after its line, and nothing erases a line early). The effective "
            "rate will be about %.2f fps.",
            len(sessions),
            fps,
            max(0, needed - len(sessions)),
            needed,
            clear_wait,
            len(sessions) / clear_wait,
        )

    queue: list[Frame] | None = None
    if loop:
        queue = await asyncio.to_thread(list, frames)
        if not queue:
            log.warning("nothing to play: no frames")
            return stats

    pool = _Pool(sessions=sessions, clear_wait=float(clear_wait))
    interval = 1.0 / float(fps)
    started = _monotonic()
    deadline = started
    pending_ends: set[asyncio.Task[None]] = set()
    warned_skip = False
    warned_late = False

    try:
        while True:
            stats.loops += 1
            source: Iterator[Frame] = iter(queue if queue is not None else frames)
            while True:
                if stop is not None and stop.is_set():
                    stats.stopped = True
                    break
                # Pull (decode + trace) *before* waiting for the deadline, so
                # that work happens inside the previous frame's slack. It runs
                # in a thread because it is CPU-bound and the event loop also
                # has every attendee's audio leg to service -- and Chime drops
                # an attendee whose audio stops.
                frame = (
                    next(source, None)
                    if queue is not None
                    else await asyncio.to_thread(next, source, None)
                )
                if frame is None:
                    break

                if not await _wait_until(deadline, stop):
                    stats.stopped = True
                    break
                deadline += interval

                now = _monotonic()
                if not frame.layers:
                    stats.frames_empty += 1
                    log.debug("frame %d traced to nothing; skipping", frame.index)
                else:
                    need = len(frame.layers)
                    if need > len(sessions):
                        raise HuddleError(
                            f"frame {frame.index} has {need} colour layer(s) but "
                            f"only {len(sessions)} attendee(s) joined. Slack keys "
                            "lines by attendee, so each colour needs its own; "
                            f"join with count={attendees_needed(fps, colors=need, clear_wait=clear_wait)}."
                        )
                    chosen = pool.take(need, now)
                    if chosen is None:
                        stats.frames_skipped += 1
                        if not warned_skip:
                            warned_skip = True
                            log.warning(
                                "skipping frame %d: none of the %d attendee(s) is "
                                "out of its %.1fs cooldown. %g fps x %.1fs x %d "
                                "colour(s) needs %d attendees; drawing early "
                                "would bring the previous frame back at full "
                                "opacity, so the frame is dropped instead.",
                                frame.index,
                                len(sessions),
                                clear_wait,
                                fps,
                                clear_wait,
                                colors,
                                needed,
                            )
                        else:
                            log.debug("skipping frame %d: no attendee free", frame.index)
                    else:
                        sent, dropped = await _draw_frame(
                            [sessions[i] for i in chosen],
                            frame,
                            share,
                            stop=stop,
                            pending_ends=pending_ends,
                        )
                        pool.used(chosen, _monotonic())
                        stats.frames_drawn += 1
                        stats.segments_sent += sent
                        stats.segments_dropped += dropped
                        if on_frame is not None:
                            try:
                                on_frame(frame, _monotonic() - started)
                            except Exception:  # noqa: BLE001 - caller's callback
                                log.warning(
                                    "on_frame callback failed for frame %d",
                                    frame.index,
                                    exc_info=True,
                                )

                lag = _monotonic() - deadline
                if lag > interval:
                    if not warned_late:
                        warned_late = True
                        log.warning(
                            "falling behind by %.2fs at %g fps (a frame takes "
                            "longer to trace and send than its slot); resyncing "
                            "to the wall clock instead of accumulating the debt",
                            lag,
                            fps,
                        )
                    deadline = _monotonic()

            if stats.stopped or not loop:
                break
    finally:
        stats.seconds = _monotonic() - started
        if pending_ends:
            # Let the repeat ENDs land: an END that never arrives leaves the
            # line alive ~8s instead of ~3.7s.
            with contextlib.suppress(Exception):
                await asyncio.wait(pending_ends, timeout=END_REPEAT_DELAY * 4)
            for task in pending_ends:
                if not task.done():
                    task.cancel()
        log.info("flipbook finished: %s", stats.describe())
    return stats


async def _draw_frame(
    sessions: Sequence[Any],
    frame: Frame,
    share: StreamSource | str,
    *,
    stop: asyncio.Event | None,
    pending_ends: set[asyncio.Task[None]],
) -> tuple[int, int]:
    """Draw one frame from ``sessions``, one layer each, and ``END`` the lines.

    Returns ``(segments sent, segments the limiter dropped)``.

    Deliberately not ``MultiSession.draw_layers``: that one picks sessions
    positionally from ``self.sessions`` (it has no notion of a cooldown), holds
    and keeps lines alive, and ``END``s with a blocking 0.3 s gap between the
    two ``END``s -- which at 2 fps would be 60% of the frame's slot. What is
    worth reusing is the interleave order, and that is reused directly.
    """
    layers = frame.layers
    drawers = [
        session.drawer(share, hue=layer.hue)
        for session, layer in zip(sessions, layers, strict=True)
    ]
    cursors = [0] * len(layers)
    sent = 0
    for index in MultiSession._interleave_order([len(x.segments) for x in layers]):
        if stop is not None and stop.is_set():
            log.info(
                "frame %d interrupted after %d of %d segment(s)",
                frame.index,
                sent,
                frame.segments,
            )
            break
        await drawers[index].stroke(
            [layers[index].segments[cursors[index]]], end=False
        )
        cursors[index] += 1
        sent += 1

    # END now, even when interrupted: the line has to be ENDed for its 3.7s
    # fade (and so this attendee's cooldown) to start at all.
    await asyncio.gather(
        *(_end(drawer) for drawer in drawers), return_exceptions=True
    )
    for drawer in drawers:
        task = asyncio.create_task(_repeat_end(drawer))
        pending_ends.add(task)
        task.add_done_callback(pending_ends.discard)

    dropped = sum(int(getattr(drawer, "dropped", 0) or 0) for drawer in drawers)
    return sent, dropped


async def _end(drawer: Any) -> None:
    await drawer.end_stroke(twice=False)


async def _repeat_end(drawer: Any) -> None:
    """The doc's second ``END``, in the background rather than in the frame.

    "If an ``END`` is dropped, the line lives ~8 s instead of ~3.7 s. Sending
    ``END`` twice (a few hundred ms apart) ... avoided every 'old frame came
    back' glitch." The gap is real time, so waiting for it inline would spend
    0.3 s of every frame's slot; the cooldown already allows for it.
    """
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.sleep(END_REPEAT_DELAY)
        with contextlib.suppress(Exception):
            await drawer.end_stroke(twice=False)


async def _wait_until(deadline: float, stop: asyncio.Event | None) -> bool:
    """Wait for ``deadline``; ``False`` if ``stop`` was set instead.

    The deadline is absolute (``t0 + n/fps``), which is what keeps playback on
    the wall clock: sleeping ``1/fps`` per frame would add every frame's
    decode, trace and send time to the total, and a flipbook would slide
    further behind its soundtrack with every frame.
    """
    if stop is not None and stop.is_set():
        return False
    while True:
        remaining = deadline - _monotonic()
        if remaining <= 0:
            await asyncio.sleep(0)  # always give the loop a turn
            return not (stop is not None and stop.is_set())
        if stop is None:
            await _wait(remaining)
            continue
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=remaining)
        if stop.is_set():
            return False

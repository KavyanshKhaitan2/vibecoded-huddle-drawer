"""Tests for playing a video onto a share as a flipbook.

No network and no live Slack: videos are generated with ``av`` into ``tmp_path``
and playback runs against fake sessions/drawers (the shape used in
``tests/test_multi.py``) on a virtual clock.

The two things worth testing hardest are the two that cannot be seen from
outside:

* an attendee must never be reused before its line is certainly gone
  (``config.LINE_CLEAR_WAIT``), because drawing early brings the whole previous
  frame back at full opacity instead of erasing anything;
* pacing must not drift, because every frame's decode/trace/send time would
  otherwise be added to the playback length.
"""

from __future__ import annotations

import asyncio
import os
from fractions import Fraction

import av
import numpy as np
import pytest

from huddle_drawings import video
from huddle_drawings.config import (
    CHIME_DATA_MESSAGE_RATE,
    CLIENT_DATA_MESSAGE_RATE,
    DATA_MESSAGE_HEADROOM,
    LINE_CLEAR_WAIT,
)
from huddle_drawings.geometry import Point, Rect, Segment
from huddle_drawings.multi import Layer, MultiSession
from huddle_drawings.ratelimit import TokenBucket
from huddle_drawings.types import HuddleError

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------- #
# fixtures: a tiny generated video
# --------------------------------------------------------------------------- #

WIDTH, HEIGHT = 160, 120
SOURCE_FPS = 8
SOURCE_FRAMES = 16  # 2.0 seconds
#: Frame ``i``'s background grey, from mid-grey upwards: widely spaced so a
#: lossy codec cannot make two frames ambiguous (which is what lets a test say
#: *which* frame it got), and never black, so the black square on it is always
#: a real feature to trace.
GREY_BASE = 45
GREY_STEP = 14


def grey_of(index: int) -> int:
    return GREY_BASE + GREY_STEP * index


def write_video(
    path: str,
    *,
    frames: int = SOURCE_FRAMES,
    fps: int = SOURCE_FPS,
    pts: list[int] | None = None,
) -> str:
    """A video whose frame ``i`` is a distinct grey with a black square on it.

    The grey identifies the frame; the square gives the tracer something real
    to trace (a uniform image traces to nothing at all). ``pts`` gives explicit
    presentation stamps in ``1/fps`` units, for a variable-frame-rate source.
    """
    with av.open(path, "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = WIDTH, HEIGHT
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            array[:, :] = grey_of(index)
            left = 10 + 5 * index
            array[40:80, left : left + 40] = 0
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            if pts is not None:
                frame.pts = pts[index]
                frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> str:
    """16 frames at 8 fps = 2.0 s, encoded once for the whole module."""
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    return write_video(str(path))


def first_grey(image) -> int:  # noqa: ANN001
    """Which source frame an image is, read off its background."""
    return int(image.getpixel((0, 0))[0])


def nearest_index(grey: int) -> int:
    return int(round((grey - GREY_BASE) / GREY_STEP))


# --------------------------------------------------------------------------- #
# decode_frames
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("fps", "expected"),
    [(8, 16), (4, 8), (2, 4), (1, 2)],
)
async def test_decode_resamples_to_the_target_fps(
    clip: str, fps: int, expected: int
) -> None:
    images = list(video.decode_frames(clip, fps=fps))
    assert len(images) == expected
    # Every frame is a distinct source frame, evenly spaced.
    picked = [nearest_index(first_grey(i)) for i in images]
    assert picked == sorted(picked)
    step = SOURCE_FPS // fps
    assert picked == [n * step for n in range(expected)]


async def test_decode_above_the_source_rate_duplicates_rather_than_inventing(
    clip: str,
) -> None:
    images = list(video.decode_frames(clip, fps=16))
    # Wanted timestamps run 0 .. 1.875 (the last frame's own time) at 1/16 s.
    assert len(images) == 31
    # Twice as many frames as the source has, so every one is a duplicate of a
    # real frame rather than an interpolation.
    assert len({nearest_index(first_grey(i)) for i in images}) == SOURCE_FRAMES


async def test_decode_honours_max_frames(clip: str) -> None:
    assert len(list(video.decode_frames(clip, fps=8, max_frames=3))) == 3
    assert list(video.decode_frames(clip, fps=8, max_frames=0)) == []


async def test_decode_honours_start(clip: str) -> None:
    images = list(video.decode_frames(clip, fps=8, start=1.0))
    assert len(images) == 8
    # 1.0 s into an 8 fps source is frame 8.
    assert nearest_index(first_grey(images[0])) == 8


async def test_decode_honours_duration(clip: str) -> None:
    assert len(list(video.decode_frames(clip, fps=8, duration=1.0))) == 8
    assert len(list(video.decode_frames(clip, fps=4, duration=1.0))) == 4
    images = list(video.decode_frames(clip, fps=8, start=0.5, duration=0.5))
    assert len(images) == 4
    assert nearest_index(first_grey(images[0])) == 4


async def test_decode_tolerates_a_video_shorter_than_the_duration(clip: str) -> None:
    assert len(list(video.decode_frames(clip, fps=8, duration=30.0))) == 16


async def test_decode_downscales_the_long_side(clip: str) -> None:
    assert list(video.decode_frames(clip, fps=1, size=64))[0].size == (64, 48)
    # Never upscales: the source is 160x120.
    assert list(video.decode_frames(clip, fps=1, size=4096))[0].size == (160, 120)


async def test_decode_returns_rgb_images(clip: str) -> None:
    assert list(video.decode_frames(clip, fps=1))[0].mode == "RGB"


async def test_decode_of_an_empty_range_is_empty_not_an_error(clip: str) -> None:
    """``start`` past the end is not a failure, just nothing to play."""
    assert list(video.decode_frames(clip, fps=8, start=50.0)) == []


async def test_decode_rejects_a_file_that_is_not_a_video(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "random.bin"
    path.write_bytes(os.urandom(8192))
    with pytest.raises(HuddleError, match="random.bin"):
        video.decode_frames(path, fps=1)

    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(HuddleError, match="empty.mp4"):
        video.decode_frames(empty, fps=1)


async def test_decode_rejects_a_file_with_no_video_stream(tmp_path) -> None:  # noqa: ANN001
    """An audio-only file opens fine and has nothing to draw."""
    path = tmp_path / "audio.wav"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("pcm_s16le", rate=8000)
        stream.layout = "mono"  # type: ignore[assignment]
        samples = np.zeros((1, 800), dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = 8000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with pytest.raises(HuddleError, match="no video stream"):
        video.decode_frames(path, fps=1)


async def test_decode_rejects_bad_arguments(clip: str) -> None:
    for kwargs in (
        {"fps": 0},
        {"fps": -1},
        {"fps": 1, "start": -1.0},
        {"fps": 1, "duration": 0.0},
        {"fps": 1, "max_frames": -1},
        {"fps": 1, "size": 0},
    ):
        with pytest.raises(ValueError):
            video.decode_frames(clip, **kwargs)  # type: ignore[arg-type]


async def test_decode_a_variable_frame_rate_source_picks_the_nearest_frame(
    tmp_path,  # noqa: ANN001
) -> None:
    """VFR is why resampling is by timestamp and not by counting frames."""
    # Frames at 0.0, 0.1, 0.15, 0.6, 1.0, 1.05 s, in 1/20 s units.
    stamps = [0, 2, 3, 12, 20, 21]
    path = write_video(
        str(tmp_path / "vfr.mp4"), frames=len(stamps), fps=20, pts=stamps
    )
    images = list(video.decode_frames(path, fps=2, duration=1.0))
    # Wanted 0.0 and 0.5: nearest are frame 0 (0.0) and frame 3 (0.6). Taking
    # every n-th frame would have given frame 1 (0.1), a tenth of a second in.
    assert [nearest_index(first_grey(i)) for i in images] == [0, 3]


# --------------------------------------------------------------------------- #
# the arithmetic
# --------------------------------------------------------------------------- #


async def test_attendees_needed_matches_the_doc() -> None:
    """"frames per second x ~8 s = accounts needed."""
    assert video.attendees_needed(1) == 9
    assert video.attendees_needed(2) == 18
    assert LINE_CLEAR_WAIT == 8.5


async def test_attendees_needed_multiplies_by_the_colour_layers() -> None:
    """"plus one per extra colour layer per frame" -- one line per attendee."""
    assert video.attendees_needed(1, colors=1) == 9
    assert video.attendees_needed(1, colors=2) == 18
    assert video.attendees_needed(2, colors=3) == 54


async def test_attendees_needed_grows_with_fps_and_the_cooldown() -> None:
    counts = [video.attendees_needed(f) for f in (0.5, 1, 2, 4, 8)]
    assert counts == sorted(counts)
    assert counts == [5, 9, 18, 35, 69]
    # A shorter cooldown would need fewer; the cooldown is the whole cost.
    assert video.attendees_needed(2, clear_wait=1.0) == 3


async def test_attendees_needed_rejects_nonsense() -> None:
    for kwargs in ({"fps": 0}, {"fps": -1}, {"fps": 1, "colors": 0}, {"fps": 1, "clear_wait": 0}):
        with pytest.raises(ValueError):
            video.attendees_needed(**kwargs)  # type: ignore[arg-type]


async def test_segments_per_frame_falls_as_fps_rises() -> None:
    budgets = [video.segments_per_frame(f) for f in (1, 2, 4, 8, 16)]
    assert budgets == sorted(budgets, reverse=True)
    assert budgets == [135, 67, 33, 16, 8]
    # 1 fps spends a whole second of our share of the meeting's budget.
    assert video.segments_per_frame(1) == int(CLIENT_DATA_MESSAGE_RATE)
    # And that share is the one knob, not a second copy of it.
    assert video.DEFAULT_HEADROOM == DATA_MESSAGE_HEADROOM
    assert video.segments_per_frame(1) == int(
        CHIME_DATA_MESSAGE_RATE * DATA_MESSAGE_HEADROOM
    )
    # Never zero, however absurd the frame rate.
    assert video.segments_per_frame(10_000) == 1


async def test_segments_per_frame_rejects_nonsense() -> None:
    for kwargs in (
        {"fps": 0},
        {"fps": 1, "rate": 0},
        {"fps": 1, "headroom": 0},
        {"fps": 1, "headroom": 1.5},
    ):
        with pytest.raises(ValueError):
            video.segments_per_frame(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# trace_frames
# --------------------------------------------------------------------------- #


async def test_trace_frames_yields_one_frame_per_image_within_budget(
    clip: str,
) -> None:
    budget = video.segments_per_frame(1)
    frames = list(
        video.trace_frames(
            video.decode_frames(clip, fps=2), max_segments=budget, fill=False
        )
    )
    assert len(frames) == 4
    assert [f.index for f in frames] == [0, 1, 2, 3]
    for frame in frames:
        assert frame.segments == sum(len(layer.segments) for layer in frame.layers)
        assert frame.segments <= budget, frame
        assert frame.segments > 0, "a black square on grey must trace to something"
        assert len(frame.layers) == 1, "one hue means one attendee"
        assert frame.layers[0].hue == video.DEFAULT_HUE


async def test_trace_frames_a_tighter_budget_costs_fewer_segments(clip: str) -> None:
    images = list(video.decode_frames(clip, fps=1, max_frames=1))
    big = next(iter(video.trace_frames(list(images), max_segments=120, fill=True)))
    small = next(iter(video.trace_frames(list(images), max_segments=20, fill=True)))
    assert small.segments <= 20 < big.segments


async def test_trace_frames_colors_gives_one_layer_per_attendee(tmp_path) -> None:  # noqa: ANN001
    """Each colour layer is another attendee, hence another line."""
    path = str(tmp_path / "colour.mp4")
    with av.open(path, "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height = WIDTH, HEIGHT
        stream.pix_fmt = "yuv420p"
        for index in range(4):
            array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            array[:, :] = (240, 240, 240)
            array[20:60, 10 + 5 * index : 60 + 5 * index] = (220, 20, 20)
            array[70:110, 90 - 5 * index : 140 - 5 * index] = (20, 60, 220)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(array, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    frames = list(
        video.trace_frames(
            video.decode_frames(path, fps=2),
            colors=2,
            max_segments=120,
            fill=True,
        )
    )
    assert frames
    for frame in frames:
        assert 1 <= len(frame.layers) <= 2
        assert all(0 <= layer.hue < 360 for layer in frame.layers)
        assert frame.segments <= 120


async def test_trace_frames_an_empty_frame_has_no_layers() -> None:
    """A uniform image traces to nothing; that must not look like a frame."""
    from PIL import Image

    blank = Image.new("RGB", (64, 48), (128, 128, 128))
    frame = next(iter(video.trace_frames([blank], max_segments=60)))
    assert frame.layers == []
    assert frame.segments == 0
    assert not frame


async def test_trace_frames_accepts_a_rect_and_extra_tracer_options(
    clip: str,
) -> None:
    rect = Rect(0.25, 0.25, 0.75, 0.75)
    frame = next(
        iter(
            video.trace_frames(
                video.decode_frames(clip, fps=1, max_frames=1),
                rect,
                max_segments=120,
                fill=False,
                simplify=0.02,
            )
        )
    )
    for layer in frame.layers:
        for segment in layer.segments:
            for point in segment.points:
                assert rect.contains(point), point


async def test_trace_frames_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        video.trace_frames([], colors=0)
    with pytest.raises(ValueError):
        video.trace_frames([], max_segments=-1)


# --------------------------------------------------------------------------- #
# fakes for playback
# --------------------------------------------------------------------------- #


def seg(x: float = 0.1) -> Segment:
    p = Point(x, x)
    return Segment(p, p, p, p)


def frame(index: int, *, segments: int = 3, layers: int = 1) -> video.Frame:
    return video.Frame.of(
        index,
        [Layer(hue=30 * (n + 1), segments=[seg()] * segments) for n in range(layers)],
    )


class Clock:
    """A virtual monotonic clock: sleeping advances it instantly."""

    def __init__(self, now: float = 1000.0) -> None:
        self.t = now

    def now(self) -> float:
        return self.t

    def advance(self, delay: float) -> None:
        self.t += max(0.0, delay)

    async def wait(self, delay: float) -> None:
        self.advance(delay)
        await asyncio.sleep(0)


class FakeDrawer:
    def __init__(self, session: FakeSession, hue: int | None) -> None:
        self.session = session
        self.hue = hue
        self.segments = 0
        self.ended = 0
        self.keep_alive_started = 0
        self.sent = 0
        self.dropped = 0

    async def stroke(self, segments, *, end: bool = True) -> None:  # noqa: ANN001
        self.segments += len(segments)
        self.sent += len(segments)
        self.session.on_stroke(self)
        await asyncio.sleep(0)

    async def end_stroke(self, *, twice: bool = True) -> None:
        self.ended += 1

    def start_keep_alive(self) -> None:  # pragma: no cover - must never run
        self.keep_alive_started += 1

    async def stop_keep_alive(self) -> None:  # pragma: no cover
        pass


class FakeSession:
    def __init__(self, attendee: str, rig: Rig) -> None:
        self.attendee_id = attendee
        self.rig = rig
        self.drawers: list[FakeDrawer] = []
        self.closed = False

    def drawer(self, share, *, hue=None):  # noqa: ANN001
        self.rig.usage.append((self.rig.clock.now(), self.attendee_id))
        drawer = FakeDrawer(self, hue)
        self.drawers.append(drawer)
        return drawer

    def on_stroke(self, drawer: FakeDrawer) -> None:
        self.rig.on_stroke(drawer)

    async def close(self) -> None:  # pragma: no cover - play must not close
        self.closed = True


class Rig:
    """A fake :class:`MultiSession` plus the tape of what happened."""

    def __init__(
        self,
        attendees: int,
        *,
        clock: Clock | None = None,
        cost: float = 0.0,
        on_segment=None,  # noqa: ANN001
    ) -> None:
        self.clock = clock or Clock()
        self.cost = cost
        self.usage: list[tuple[float, str]] = []
        self.segments = 0
        self._on_segment = on_segment
        self.sessions = [FakeSession(f"att-{i}", self) for i in range(attendees)]
        self.multi = MultiSession(
            sessions=self.sessions,  # type: ignore[arg-type]
            limiter=TokenBucket(),
        )

    def on_stroke(self, drawer: FakeDrawer) -> None:
        self.segments += 1
        if self.cost:
            self.clock.advance(self.cost)
        if self._on_segment is not None:
            self._on_segment(self.segments)

    @property
    def drawers(self) -> list[FakeDrawer]:
        return [d for s in self.sessions for d in s.drawers]

    def times_for(self, attendee: str) -> list[float]:
        return [t for t, who in self.usage if who == attendee]


@pytest.fixture
def virtual(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """Run :func:`play` on a virtual clock: deterministic and instant."""
    clock = Clock()
    monkeypatch.setattr(video, "_monotonic", clock.now)
    monkeypatch.setattr(video, "_wait", clock.wait)
    return clock


@pytest.fixture
def fast_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Don't spend a real 0.3 s per frame waiting for the repeat ``END``."""
    monkeypatch.setattr(video, "END_REPEAT_DELAY", 0.0)


# --------------------------------------------------------------------------- #
# play: attendee rotation (the heart of it)
# --------------------------------------------------------------------------- #


async def test_play_never_reuses_an_attendee_before_its_line_is_gone(
    virtual: Clock,
) -> None:
    """The glitch this avoids: an early stroke redraws the whole old frame."""
    rig = Rig(3, clock=virtual)
    frames = [frame(i) for i in range(120)]  # 12 s at 10 fps
    stats = await video.play(rig.multi, frames, "SHARER", fps=10)

    assert stats.frames_drawn == len(rig.usage) > 3, "the pool must be reused"
    for session in rig.sessions:
        times = rig.times_for(session.attendee_id)
        gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
        assert all(gap >= LINE_CLEAR_WAIT for gap in gaps), (session.attendee_id, gaps)
    # The cooldown also covers the repeat END, which lands later still.
    for session in rig.sessions:
        times = rig.times_for(session.attendee_id)
        for a, b in zip(times, times[1:], strict=False):
            assert b - a >= LINE_CLEAR_WAIT + video.END_REPEAT_DELAY


async def test_play_uses_a_fresh_attendee_for_every_frame_in_a_burst(
    virtual: Clock, fast_end: None
) -> None:
    rig = Rig(9, clock=virtual)
    stats = await video.play(rig.multi, [frame(i) for i in range(9)], "S", fps=1)
    assert stats.frames_drawn == 9
    assert stats.frames_skipped == 0
    assert len({who for _, who in rig.usage}) == 9, "nine frames, nine attendees"


async def test_play_skips_rather_than_drawing_early(
    virtual: Clock, fast_end: None, caplog: pytest.LogCaptureFixture
) -> None:
    rig = Rig(3, clock=virtual)
    frames = [frame(i) for i in range(20)]  # 2 s at 10 fps
    with caplog.at_level("WARNING"):
        stats = await video.play(rig.multi, frames, "S", fps=10)

    assert stats.frames_drawn == 3, "only three attendees are free"
    assert stats.frames_skipped == 17
    assert stats.frames_total == 20
    assert len(rig.usage) == 3
    # 3 attendees / 8.5 s of cooldown, whatever fps said.
    assert stats.effective_fps == pytest.approx(1.5, abs=1e-6)
    # The first skip must explain itself with the arithmetic.
    assert "cooldown" in caplog.text
    assert "needs 86 attendees" in caplog.text
    assert "3 attendee(s)" in caplog.text


async def test_play_warns_up_front_when_the_pool_is_too_small(
    virtual: Clock, fast_end: None, caplog: pytest.LogCaptureFixture
) -> None:
    rig = Rig(2, clock=virtual)
    with caplog.at_level("WARNING"):
        await video.play(rig.multi, [frame(0)], "S", fps=2)
    assert "only 2 attendee(s) for 2 fps" in caplog.text
    assert "18" in caplog.text


async def test_play_ends_every_line_it_draws(virtual: Clock, fast_end: None) -> None:
    """Without an END the line lives ~8 s and the cooldown never starts."""
    rig = Rig(9, clock=virtual)
    await video.play(rig.multi, [frame(i) for i in range(5)], "S", fps=1)
    drawers = rig.drawers
    assert len(drawers) == 5
    assert all(d.ended >= 1 for d in drawers)


async def test_play_never_holds_a_frame(virtual: Clock, fast_end: None) -> None:
    """A held line cannot fade, and the next frame needs it gone."""
    rig = Rig(9, clock=virtual)
    await video.play(rig.multi, [frame(i) for i in range(5)], "S", fps=1)
    assert all(d.keep_alive_started == 0 for d in rig.drawers)


async def test_play_draws_a_multi_colour_frame_from_one_attendee_per_layer(
    virtual: Clock, fast_end: None
) -> None:
    rig = Rig(6, clock=virtual)
    frames = [frame(i, layers=3, segments=2) for i in range(2)]
    stats = await video.play(rig.multi, frames, "S", fps=1, colors=3)
    assert stats.frames_drawn == 2
    assert len(rig.usage) == 6, "three attendees per frame, none reused"
    assert len({who for _, who in rig.usage}) == 6
    assert sorted(d.hue for d in rig.drawers) == [30, 30, 60, 60, 90, 90]


async def test_play_interleaves_a_multi_colour_frame(
    virtual: Clock, fast_end: None
) -> None:
    """Colours must arrive together, or the first ones start fading early."""
    tape: list[int] = []

    rig = Rig(4, clock=virtual)
    for session in rig.sessions:
        original = session.drawer

        def drawer(share, *, hue=None, _original=original):  # noqa: ANN001
            made = _original(share, hue=hue)
            inner = made.stroke

            async def stroke(segments, *, end=True):  # noqa: ANN001
                tape.append(made.hue or 0)
                await inner(segments, end=end)

            made.stroke = stroke  # type: ignore[method-assign]
            return made

        session.drawer = drawer  # type: ignore[method-assign]

    one = video.Frame.of(
        0, [Layer(hue=10, segments=[seg()] * 12), Layer(hue=200, segments=[seg()] * 3)]
    )
    await video.play(rig.multi, [one], "S", fps=1, colors=2)
    last_small = max(i for i, hue in enumerate(tape) if hue == 200)
    assert last_small >= len(tape) * 0.5, tape


async def test_play_raises_when_a_frame_needs_more_attendees_than_exist(
    virtual: Clock, fast_end: None
) -> None:
    rig = Rig(2, clock=virtual)
    with pytest.raises(HuddleError, match="colour layer"):
        await video.play(
            rig.multi, [frame(0, layers=3)], "S", fps=1, colors=3
        )


async def test_play_counts_an_empty_frame_without_spending_an_attendee(
    virtual: Clock, fast_end: None
) -> None:
    rig = Rig(2, clock=virtual)
    frames = [video.Frame.of(0, []), frame(1), video.Frame.of(2, [])]
    stats = await video.play(rig.multi, frames, "S", fps=1)
    assert (stats.frames_drawn, stats.frames_empty, stats.frames_skipped) == (1, 2, 0)
    assert len(rig.usage) == 1


# --------------------------------------------------------------------------- #
# play: pacing
# --------------------------------------------------------------------------- #


async def test_play_pacing_does_not_drift(virtual: Clock, fast_end: None) -> None:
    """Deadlines accumulate from t0; per-frame work must not add up."""
    # 30 ms of "tracing and sending" inside a 50 ms frame.
    rig = Rig(40, clock=virtual, cost=0.01)
    frames = [frame(i, segments=3) for i in range(10)]
    stats = await video.play(
        rig.multi, frames, "S", fps=20, clear_wait=0.001
    )
    assert stats.frames_drawn == 10
    expected = [1000.0 + n * 0.05 for n in range(10)]
    assert [t for t, _ in rig.usage] == pytest.approx(expected, abs=1e-6)
    # Sleeping 1/fps per frame would have taken 10 x (0.05 + 0.03) = 0.8 s.
    assert stats.seconds == pytest.approx(0.45 + 0.03, abs=1e-6)


async def test_play_resyncs_instead_of_accumulating_a_debt(
    virtual: Clock, fast_end: None, caplog: pytest.LogCaptureFixture
) -> None:
    """A frame that overruns its slot must not push every later frame back."""
    rig = Rig(40, clock=virtual, cost=0.1)  # 0.2 s of work per 2-segment frame
    frames = [frame(i, segments=2) for i in range(5)]
    with caplog.at_level("WARNING"):
        stats = await video.play(
            rig.multi, frames, "S", fps=20, clear_wait=0.001
        )
    assert stats.frames_drawn == 5
    assert "falling behind" in caplog.text
    # Each frame costs 0.2 s and nothing waits, so playback is work-bound.
    assert stats.seconds == pytest.approx(1.0, abs=1e-6)
    times = [t for t, _ in rig.usage]
    assert times == pytest.approx([1000.0 + 0.2 * n for n in range(5)], abs=1e-6)


async def test_play_reports_effective_fps_and_adds_up(
    virtual: Clock, fast_end: None
) -> None:
    rig = Rig(9, clock=virtual)
    frames = [frame(i, segments=4) for i in range(5)]
    stats = await video.play(rig.multi, frames, "S", fps=1)
    assert stats.frames_total == 5
    assert stats.frames_drawn == 5
    assert stats.segments_sent == 20 == rig.segments
    assert stats.segments_dropped == 0
    assert stats.seconds == pytest.approx(4.0, abs=1e-6)
    # Five frames at 1 fps: the last one is drawn at t=4, but its second has
    # still to run, so the honest rate is 1.0 and not 5/4.
    assert stats.effective_fps == pytest.approx(1.0, abs=1e-6)
    assert stats.attendees == 9
    assert stats.attendees_needed == 9
    assert "5 frame(s) drawn" in stats.describe()


async def test_play_counts_segments_the_limiter_dropped(
    virtual: Clock, fast_end: None
) -> None:
    rig = Rig(9, clock=virtual)

    original = rig.sessions[0].drawer

    def drawer(share, *, hue=None):  # noqa: ANN001
        made = original(share, hue=hue)
        made.dropped = 7
        return made

    rig.sessions[0].drawer = drawer  # type: ignore[method-assign]
    stats = await video.play(rig.multi, [frame(0)], "S", fps=1)
    assert stats.segments_dropped == 7
    assert "dropped by the limiter" in stats.describe()


async def test_play_on_frame_is_called_once_per_drawn_frame(
    virtual: Clock, fast_end: None
) -> None:
    seen: list[tuple[int, float]] = []
    rig = Rig(9, clock=virtual)
    await video.play(
        rig.multi,
        [frame(i) for i in range(3)],
        "S",
        fps=1,
        on_frame=lambda f, t: seen.append((f.index, t)),
    )
    assert [i for i, _ in seen] == [0, 1, 2]
    assert [t for _, t in seen] == pytest.approx([0.0, 1.0, 2.0], abs=1e-6)


async def test_play_survives_a_failing_on_frame_callback(
    virtual: Clock, fast_end: None
) -> None:
    def boom(frame_: video.Frame, elapsed: float) -> None:
        raise RuntimeError("nope")

    rig = Rig(9, clock=virtual)
    stats = await video.play(
        rig.multi, [frame(i) for i in range(2)], "S", fps=1, on_frame=boom
    )
    assert stats.frames_drawn == 2


async def test_play_pulls_frames_lazily(virtual: Clock, fast_end: None) -> None:
    """Tracing must overlap with the previous frame, not happen up front."""
    pulled: list[int] = []

    def source():
        for i in range(4):
            pulled.append(i)
            yield frame(i)

    rig = Rig(9, clock=virtual)
    seen: list[int] = []
    await video.play(
        rig.multi,
        source(),
        "S",
        fps=1,
        on_frame=lambda f, t: seen.append(len(pulled)),
    )
    # One frame ahead at most: pulled 1 when frame 0 was drawn, and so on.
    assert seen == [1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# play: stop and loop
# --------------------------------------------------------------------------- #


async def test_play_stop_abandons_playback_between_frames(fast_end: None) -> None:
    rig = Rig(40)
    stop = asyncio.Event()
    frames = [frame(i) for i in range(500)]

    def after_two(f: video.Frame, elapsed: float) -> None:
        if f.index >= 1:
            stop.set()

    stats = await asyncio.wait_for(
        video.play(
            rig.multi,
            frames,
            "S",
            fps=200,
            stop=stop,
            clear_wait=0.001,
            on_frame=after_two,
        ),
        timeout=5,
    )
    assert stats.stopped
    assert stats.frames_drawn == 2, "playback must stop at the next frame"
    assert all(d.ended >= 1 for d in rig.drawers), "an abandoned frame still ENDs"


async def test_play_stop_abandons_a_frame_mid_send(fast_end: None) -> None:
    stop = asyncio.Event()
    rig = Rig(4, on_segment=lambda n: stop.set() if n >= 5 else None)
    big = video.Frame.of(0, [Layer(hue=10, segments=[seg()] * 400)])
    stats = await asyncio.wait_for(
        video.play(rig.multi, [big], "S", fps=1, stop=stop, clear_wait=0.001),
        timeout=5,
    )
    assert stats.stopped
    assert stats.frames_drawn == 1, "the interrupted frame still counts as drawn"
    assert 5 <= rig.segments < 400, rig.segments
    assert stats.segments_sent == rig.segments
    assert all(d.ended >= 1 for d in rig.drawers), (
        "an interrupted line must still be ENDed, or it hangs around ~8s"
    )


async def test_play_stop_before_the_first_frame_draws_nothing(
    fast_end: None,
) -> None:
    rig = Rig(4)
    stop = asyncio.Event()
    stop.set()
    stats = await video.play(
        rig.multi, [frame(0)], "S", fps=1, stop=stop, clear_wait=0.001
    )
    assert stats.frames_drawn == 0
    assert stats.stopped
    assert not rig.usage


async def test_play_loops_until_stopped(fast_end: None) -> None:
    rig = Rig(40)
    stop = asyncio.Event()
    drawn: list[int] = []

    def count(f: video.Frame, elapsed: float) -> None:
        drawn.append(f.index)
        if len(drawn) >= 7:
            stop.set()

    stats = await asyncio.wait_for(
        video.play(
            rig.multi,
            [frame(0), frame(1)],
            "S",
            fps=200,
            loop=True,
            stop=stop,
            clear_wait=0.001,
            on_frame=count,
        ),
        timeout=5,
    )
    assert stats.loops >= 4, stats.loops
    assert stats.frames_drawn >= 7
    assert drawn[:4] == [0, 1, 0, 1], "the clip must restart from frame 0"


async def test_play_loop_of_no_frames_returns_instead_of_spinning(
    fast_end: None,
) -> None:
    rig = Rig(2)
    stats = await asyncio.wait_for(
        video.play(rig.multi, [], "S", fps=1, loop=True), timeout=5
    )
    assert stats.frames_total == 0
    assert stats.frames_drawn == 0


async def test_play_of_no_frames_is_a_noop(virtual: Clock, fast_end: None) -> None:
    rig = Rig(2, clock=virtual)
    stats = await video.play(rig.multi, [], "S", fps=1)
    assert stats.frames_total == 0
    assert not rig.usage


async def test_play_never_closes_a_session(virtual: Clock, fast_end: None) -> None:
    """Chime drops an attendee whose audio leg stops, so stay joined."""
    rig = Rig(9, clock=virtual)
    await video.play(rig.multi, [frame(i) for i in range(3)], "S", fps=1)
    assert not any(s.closed for s in rig.sessions)
    assert len(rig.multi.sessions) == 9


async def test_play_uses_the_sessions_shared_limiter(
    virtual: Clock, fast_end: None
) -> None:
    """A second bucket would overrun the meeting by exactly N."""
    rig = Rig(9, clock=virtual)
    before = rig.multi.limiter
    await video.play(rig.multi, [frame(0)], "S", fps=1)
    assert rig.multi.limiter is before


async def test_play_rejects_nonsense(virtual: Clock) -> None:
    rig = Rig(2, clock=virtual)
    with pytest.raises(ValueError):
        await video.play(rig.multi, [], "S", fps=0)
    with pytest.raises(ValueError):
        await video.play(rig.multi, [], "S", fps=1, colors=0)
    empty = MultiSession(sessions=[], limiter=TokenBucket())
    with pytest.raises(HuddleError, match="nobody to draw"):
        await video.play(empty, [frame(0)], "S", fps=1)


# --------------------------------------------------------------------------- #
# end to end, still with no network: decode -> trace -> play
# --------------------------------------------------------------------------- #


async def test_decode_trace_and_play_a_real_clip(
    clip: str, virtual: Clock, fast_end: None
) -> None:
    fps = 2.0
    budget = video.segments_per_frame(fps)
    rig = Rig(video.attendees_needed(fps), clock=virtual)
    frames = video.trace_frames(
        video.decode_frames(clip, fps=fps), max_segments=budget, fill=True
    )
    stats = await video.play(rig.multi, frames, "SHARER", fps=fps)

    assert stats.frames_total == 4
    assert stats.frames_skipped == 0
    assert stats.frames_drawn == 4
    assert stats.segments_sent == rig.segments > 0
    assert stats.segments_sent <= 4 * budget
    assert len({who for _, who in rig.usage}) == 4, "a fresh attendee per frame"

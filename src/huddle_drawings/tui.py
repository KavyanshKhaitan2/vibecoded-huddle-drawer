"""Interactive console: join once, then keep choosing what to do.

Joining is the expensive part -- a ``rooms.join``, a websocket, a TURN
allocation and a DTLS handshake per attendee, a couple of seconds each -- so
an interactive session that stays connected is much nicer than re-joining for
every drawing. It also lets the UI own two things that are easy to get wrong
by hand:

* **The fade state machine.** A line cannot be erased; it fades 0.7 s after
  ``END`` (or 5 s after the last stroke) over 3 s. Any stroke sent before that
  finishes brings the *whole previous line* back at full opacity. So before
  starting a fresh drawing this waits out the remainder, with a countdown,
  instead of letting you produce a confusing double-exposure.
* **The shared message budget.** Every segment is one Chime message and the
  whole meeting shares ~150/s, so each action prints what it will cost before
  sending it.

No curses and no extra dependencies: it is a line-driven menu, which also
means it works over ssh and in a dumb terminal. ``Console`` is a protocol, so
tests drive it with a scripted transcript rather than a terminal.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from functools import partial
import sys
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from . import shapes
from . import image_trace as image_trace_module
from .image_trace import DEFAULT_MAX_SEGMENTS
from .config import CLIENT_DATA_MESSAGE_RATE, LINE_CLEAR_WAIT, Config
from .geometry import Rect, Segment
from .multi import Layer, MultiSession
from .peer_messages import build_reacji, describe_peer_message, parse_peer_message
from .types import DataMessage, HuddleError, StreamSource

logger = logging.getLogger(__name__)

HELP_WIDTH = 72


class Console(Protocol):
    """Terminal I/O, narrowed so tests can script it."""

    def write(self, text: str = "") -> None: ...

    async def ask(self, prompt: str, default: str = "") -> str: ...


class StdConsole:
    """Real terminal I/O that does not block the event loop.

    ``input()`` is run in a worker thread; the loop keeps running, so Chime
    pings, keep-alives and the audio leg all stay alive while you are reading
    the menu. Without that, sitting at a prompt for more than a few seconds
    would let the drawing fade and could drop the attendee.
    """

    def write(self, text: str = "") -> None:
        print(text)

    async def ask(self, prompt: str, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        try:
            raw = await asyncio.to_thread(input, f"{prompt}{suffix}: ")
        except (EOFError, KeyboardInterrupt):
            return "q"
        raw = raw.strip()
        return raw or default


@contextlib.contextmanager
def _sigint_sets(event: asyncio.Event, tui: Tui) -> Iterator[None]:
    """Make Ctrl-C set ``event`` instead of killing the session.

    While a drawing is being sent or held, Ctrl-C should mean "stop this
    drawing and give me the menu back", not "quit" -- re-joining costs a
    `rooms.join`, a websocket, a TURN allocation and a DTLS handshake per
    attendee, which is far too much to pay for a stray keypress.

    Uses the loop's signal handler so the interrupt is delivered as an event
    rather than a ``KeyboardInterrupt`` raised at an arbitrary await. Falls
    back to leaving the default handler in place where that is unavailable
    (Windows, or a non-main thread), in which case Ctrl-C behaves as it did
    before.
    """
    loop = asyncio.get_running_loop()
    installed = False

    def on_sigint() -> None:
        if not event.is_set():
            event.set()
            tui.write("\n  Ctrl-C: stopping this drawing…")

    try:
        loop.add_signal_handler(signal.SIGINT, on_sigint)
        installed = True
    except (NotImplementedError, RuntimeError, ValueError):
        logger.debug("cannot install a SIGINT handler here", exc_info=True)
    try:
        yield
    finally:
        if installed:
            with contextlib.suppress(Exception):
                loop.remove_signal_handler(signal.SIGINT)

@dataclass
class TuiState:
    """What the session knows between actions."""

    multi: MultiSession
    share: StreamSource | None = None
    hues: list[int] = field(default_factory=lambda: [200])
    rect: Rect = field(default_factory=lambda: Rect(0.25, 0.25, 0.75, 0.75))
    hold: float = 15.0
    #: Segment budget for the next drawing; the budget prompt sets it.
    budget: int = DEFAULT_MAX_SEGMENTS
    #: When the last drawing was ``END``ed, for the fade bookkeeping.
    last_end: float | None = None
    watching: bool = False
    #: Fill solid regions instead of outlining them. ``None`` means "use the
    #: per-source default", which is to fill: an outline reads as a hollow
    #: wireframe and usually costs *more* messages than filling. Set False
    #: from the fill menu for line art, where the drawing is its outlines.
    fill: bool | None = None
    #: Kept for emoji name lookups (`emoji.list`); None means literal
    #: characters only.
    slack: object | None = None
    _watch_unsub: Callable[[], None] | None = None

    @property
    def attendees(self) -> list[str]:
        return [s.attendee_id for s in self.multi.sessions]

    def seconds_until_clear(self) -> float:
        """How long until the previous line is certainly gone."""
        if self.last_end is None:
            return 0.0
        return max(0.0, LINE_CLEAR_WAIT - (time.monotonic() - self.last_end))


Handler = Callable[["Tui"], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class MenuItem:
    key: str
    label: str
    handler: Handler
    needs_share: bool = False


class Tui:
    """The interactive loop."""

    def __init__(
        self,
        state: TuiState,
        console: Console | None = None,
        *,
        share_timeout: float = 120.0,
    ) -> None:
        self.state = state
        self.console = console or StdConsole()
        self.share_timeout = share_timeout
        self.running = True
        self.menu: list[MenuItem] = [
            MenuItem("1", "draw a shape", _draw_shape, needs_share=True),
            MenuItem("2", "draw an image", _draw_image, needs_share=True),
            MenuItem("3", "draw text", _draw_text, needs_share=True),
            MenuItem("e", "draw an emoji", _draw_emoji, needs_share=True),
            MenuItem("4", "draw an SVG path", _draw_svg, needs_share=True),
            MenuItem("5", "set colour(s)", _set_hues),
            MenuItem("6", "set the area to draw in", _set_rect),
            MenuItem("f", "fill or outline", _set_fill),
            MenuItem("7", "send a reaction", _send_reacji),
            MenuItem("8", "find the screen share", _find_share),
            MenuItem("9", "watch peer messages", _toggle_watch),
            MenuItem("s", "status", _status),
            MenuItem("q", "leave the huddle and quit", _quit),
        ]

    # -- plumbing ----------------------------------------------------------

    def write(self, text: str = "") -> None:
        self.console.write(text)

    async def ask(self, prompt: str, default: str = "") -> str:
        return await self.console.ask(prompt, default)

    async def ask_float(self, prompt: str, default: float) -> float:
        raw = await self.ask(prompt, f"{default:g}")
        try:
            return float(raw)
        except ValueError:
            self.write(f"  not a number: {raw!r}, keeping {default:g}")
            return default

    async def ask_int(
        self, prompt: str, default: int, *, low: int | None = None, high: int | None = None
    ) -> int:
        raw = await self.ask(prompt, str(default))
        try:
            value = int(raw)
        except ValueError:
            self.write(f"  not a whole number: {raw!r}, keeping {default}")
            return default
        if low is not None and value < low:
            self.write(f"  {value} is below {low}, keeping {default}")
            return default
        if high is not None and value > high:
            self.write(f"  {value} is above {high}, keeping {default}")
            return default
        return value

    # -- the loop ----------------------------------------------------------

    def trace_options(
        self, *, default_fill: bool, budget: int | None = None
    ) -> dict[str, object]:
        """Tracer options for the next drawing.

        ``default_fill`` is what this source wants when the session is on
        "auto": emoji fill, images outline. An explicit choice from the fill
        menu overrides it. Goes through
        :func:`image_trace.trace_options` so the console and the command line
        cannot drift apart -- they did once, and the console kept drawing
        hollow outlines after the command line was fixed.
        """
        fill = default_fill if self.state.fill is None else self.state.fill
        return image_trace_module.trace_options(
            fill=fill,
            max_segments=budget if budget is not None else self.state.budget,
        )

    def render_menu(self) -> None:
        state = self.state
        share = (
            f"share by {state.share.sharer_attendee_id[:8]}"
            if state.share
            else "no screen share found yet"
        )
        hues = ", ".join(str(h) for h in state.hues)
        self.write()
        self.write("-" * HELP_WIDTH)
        self.write(
            f" {len(state.attendees)} attendee(s) | {share} | hue(s) {hues}"
        )
        area = state.rect
        self.write(
            f" area x {area.x0:.2f}-{area.x1:.2f}, y {area.y0:.2f}-{area.y1:.2f}"
            f" | hold {state.hold:g}s"
            f" | {'outline' if state.fill is False else 'fill'}"
            + (" | watching" if state.watching else "")
        )
        remaining = state.seconds_until_clear()
        if remaining:
            self.write(
                f" previous line still fading: {remaining:.0f}s until a new "
                "drawing is safe"
            )
        self.write("-" * HELP_WIDTH)
        for item in self.menu:
            self.write(f"  [{item.key}] {item.label}")

    async def run(self) -> None:
        self.write(
            f"connected as {', '.join(a[:8] for a in self.state.attendees)}"
        )
        if self.state.share is None:
            with contextlib.suppress(Exception):
                await _find_share(self)

        while self.running:
            self.render_menu()
            choice = (await self.ask("choose")).lower()
            item = next((i for i in self.menu if i.key == choice), None)
            if item is None:
                self.write(f"  no such option: {choice!r}")
                continue
            if item.needs_share and self.state.share is None:
                self.write("  no screen share yet — use [8] to look for one")
                continue
            try:
                await item.handler(self)
            except HuddleError as exc:
                self.write(f"  error: {exc}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the session alive
                self.write(f"  failed: {type(exc).__name__}: {exc}")

    # -- drawing -----------------------------------------------------------

    async def wait_for_clear(self) -> bool:
        """Wait out the previous line's fade, with a countdown.

        Returns False if the user chose not to wait. Drawing anyway is
        allowed, but the renderer will bring the whole old line back at full
        opacity, so this asks rather than deciding.
        """
        remaining = self.state.seconds_until_clear()
        if remaining <= 0:
            return True
        self.write(
            f"  the previous line has {remaining:.0f}s of fade left. Drawing now "
            "would bring all of it back at full opacity."
        )
        answer = (await self.ask("  wait for it to clear? (y/n)", "y")).lower()
        if answer.startswith("n"):
            return False
        while True:
            remaining = self.state.seconds_until_clear()
            if remaining <= 0:
                break
            self.write(f"  waiting… {remaining:.0f}s")
            await asyncio.sleep(min(1.0, remaining))
        return True

    async def draw(self, layers: Sequence[Layer]) -> None:
        """Draw ``layers`` and record the fade bookkeeping."""
        state = self.state
        if state.share is None:
            raise HuddleError("no screen share to draw on")
        total = sum(len(layer.segments) for layer in layers)
        if not total:
            self.write("  nothing to draw")
            return
        if len(layers) > len(state.multi.sessions):
            raise HuddleError(
                f"{len(layers)} colour(s) need {len(layers)} attendees but only "
                f"{len(state.multi.sessions)} joined; restart with --colors "
                f"{len(layers)}"
            )
        self.write(
            f"  {total} segments across {len(layers)} colour(s) "
            f"(~{total / CLIENT_DATA_MESSAGE_RATE:.1f}s of the meeting's shared "
            "budget)"
        )
        if not await self.wait_for_clear():
            self.write("  drawing anyway — expect the old line to reappear")
        hold = state.hold
        stop = asyncio.Event()
        with _sigint_sets(stop, self):
            await state.multi.draw_layers(
                layers, state.share, hold=hold, stop=stop
            )
        state.last_end = time.monotonic()
        if stop.is_set():
            self.write(
                "  interrupted; the lines were ENDed and fade from now. "
                "Back to the menu."
            )
        else:
            self.write(
                "  drawn; it fades ~3.7s from now and cannot be erased sooner"
            )

    def layers_from_segments(self, segments: Sequence[Segment]) -> list[Layer]:
        """One layer per configured hue, splitting the segments between them."""
        hues = self.state.hues or [200]
        if len(hues) == 1:
            return [Layer(hue=hues[0], segments=list(segments))]
        # Interleave so each colour gets a spread of the drawing rather than
        # one contiguous chunk of it.
        buckets: list[list[Segment]] = [[] for _ in hues]
        for index, segment in enumerate(segments):
            buckets[index % len(hues)].append(segment)
        return [
            Layer(hue=hue, segments=bucket)
            for hue, bucket in zip(hues, buckets, strict=True)
            if bucket
        ]


# --------------------------------------------------------------------------- #
# handlers
# --------------------------------------------------------------------------- #


async def _status(tui: Tui) -> None:
    state = tui.state
    tui.write(f"  attendees: {', '.join(state.attendees)}")
    sources = state.multi.sessions[0].signaling.sources if state.multi.sessions else {}
    if not sources:
        tui.write("  no media sources in INDEX yet")
    for src in sources.values():
        kind = "content share" if src.is_content_share else (src.media_type or "?")
        size = f" {src.width}x{src.height}" if src.width else ""
        tui.write(f"  stream {src.stream_id:>4}  {kind:<14} {src.attendee_id}{size}")


async def _find_share(tui: Tui) -> None:
    tui.write("  looking for a screen share…")
    try:
        share = await tui.state.multi.wait_for_share(timeout=tui.share_timeout)
    except (TimeoutError, HuddleError) as exc:
        tui.write(f"  no screen share found: {exc}")
        return
    tui.state.share = share
    tui.write(f"  found a share by {share.sharer_attendee_id}")


async def _set_hues(tui: Tui) -> None:
    tui.write(
        "  Hue is degrees 0-359. The renderer always draws hsla(hue,100%,75%), "
        "so every colour is a pastel —"
    )
    tui.write("  nothing can be dark, grey or white, and only the hue varies.")
    tui.write(
        f"  You have {len(tui.state.multi.sessions)} attendee(s), so at most "
        f"{len(tui.state.multi.sessions)} colour(s) at once."
    )
    raw = await tui.ask(
        "  hue(s), comma separated", ",".join(str(h) for h in tui.state.hues)
    )
    hues: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            tui.write(f"  skipping {part!r}: not a whole number")
            continue
        if not 0 <= value <= 359:
            tui.write(f"  skipping {value}: outside 0-359")
            continue
        hues.append(value)
    if not hues:
        tui.write("  no valid hues given; keeping the current ones")
        return
    if len(hues) > len(tui.state.multi.sessions):
        tui.write(
            f"  only {len(tui.state.multi.sessions)} attendee(s) joined, so "
            f"keeping the first {len(tui.state.multi.sessions)} hue(s)"
        )
        hues = hues[: len(tui.state.multi.sessions)]
    tui.state.hues = hues
    tui.write(f"  colours set to {', '.join(str(h) for h in hues)}")


async def _set_rect(tui: Tui) -> None:
    tui.write("  Coordinates are 0..1 over the shared video.")
    current = tui.state.rect
    x = await tui.ask_float("  left edge", current.x0)
    y = await tui.ask_float("  top edge", current.y0)
    w = await tui.ask_float("  width", current.width)
    h = await tui.ask_float("  height", current.height)
    rect = Rect(x, y, x + w, y + h)
    if rect.width <= 0 or rect.height <= 0:
        tui.write("  that area has no size; keeping the current one")
        return
    tui.state.rect = rect
    tui.write(
        f"  area set to x {rect.x0:.2f}-{rect.x1:.2f}, y {rect.y0:.2f}-{rect.y1:.2f}"
    )


async def _set_fill(tui: Tui) -> None:
    """Choose filling or outlining, or leave it per source."""
    tui.write(
        "  fill = solid regions via the hairpin trick; outline = just the "
        "edges, which looks hollow"
    )
    tui.write(
        "  filling is usually *cheaper* too: a party popper is 91 segments "
        "filled against 216 outlined"
    )
    current = (
        "fill" if tui.state.fill else "outline" if tui.state.fill is False else "auto"
    )
    answer = (await tui.ask("  fill / outline / auto", current)).strip().lower()
    if answer.startswith("f"):
        tui.state.fill = True
    elif answer.startswith("o") or answer.startswith("l"):
        tui.state.fill = False
    elif answer.startswith("a"):
        tui.state.fill = None
    else:
        tui.write(f"  not one of fill/outline/auto: {answer!r}")
        return
    tui.write(
        f"  now: {'outline' if tui.state.fill is False else 'fill'}"
    )

async def _draw_shape(tui: Tui) -> None:
    tui.write(f"  shapes: {', '.join(shapes.SHAPES)}")
    name = await tui.ask("  shape", "heart")
    if name not in shapes.SHAPES:
        tui.write(f"  unknown shape {name!r}")
        return
    segments = shapes.shape_segments(name, tui.state.rect)
    await tui.draw(tui.layers_from_segments(segments))


async def _draw_text(tui: Tui) -> None:
    from .geometry import text_coverage

    tui.write(f"  the built-in font covers: {text_coverage().strip()}")
    tui.write("  anything else draws as a placeholder box, 4 segments each")
    text = await tui.ask("  text")
    if not text:
        tui.write("  nothing to draw")
        return
    segments = shapes.text_in(text, tui.state.rect)
    await tui.draw(tui.layers_from_segments(segments))


async def _draw_svg(tui: Tui) -> None:
    tui.write("  an SVG path 'd' string, in any coordinate space (it gets fitted)")
    tui.write("  arcs (A/a) are not supported; convert them to curves first")
    path = await tui.ask("  path")
    if not path:
        return
    segments = shapes.svg_path_segments(path, tui.state.rect)
    await tui.draw(tui.layers_from_segments(segments))


async def _draw_image(tui: Tui) -> None:
    from . import image_trace

    path = await tui.ask("  image path")
    if not path:
        return
    attendees = len(tui.state.multi.sessions)
    colors = await tui.ask_int(
        f"  colours (1-{attendees}, one attendee each)", min(len(tui.state.hues), attendees),
        low=1, high=attendees,
    )
    budget = await tui.ask_int("  max segments", tui.state.budget, low=1)
    tui.state.budget = budget
    tui.write("  tracing…")
    if colors > 1:
        traced = await asyncio.to_thread(
            partial(
                image_trace.trace_image_colors,
                path,
                tui.state.rect,
                layers=colors,
                **tui.trace_options(default_fill=True),
            )
        )
        layers = [Layer(hue=t.hue, segments=t.segments) for t in traced if t.segments]
        for layer in layers:
            tui.write(f"    hue {layer.hue:3d}: {len(layer.segments)} segments")
    else:
        result = await asyncio.to_thread(
            partial(
                image_trace.trace_image,
                path,
                tui.state.rect,
                **tui.trace_options(default_fill=True),
            )
        )
        tui.write(f"    {result.contours} contours -> {len(result.segments)} segments")
        layers = tui.layers_from_segments(result.segments)
    if not layers:
        tui.write("  that image traced to nothing")
        return
    await tui.draw(layers)


async def _draw_emoji(tui: Tui) -> None:
    """Draw an emoji, in colour if there are attendees for it.

    Emoji are the best thing to draw here: flat regions and hard edges, which
    is exactly what contour tracing wants.
    """
    from . import image_trace
    from .emoji import emoji_image, looks_like_unicode_emoji

    tui.write("  a literal emoji (🎉) or a Slack name (:squirrel:, :my-emoji:)")
    spec = await tui.ask("  emoji")
    if not spec:
        return
    attendees = len(tui.state.multi.sessions)
    colors = await tui.ask_int(
        f"  colours (1-{attendees}, one attendee each)",
        min(len(tui.state.hues), attendees),
        low=1,
        high=attendees,
    )
    budget = await tui.ask_int("  max segments", tui.state.budget, low=1)
    tui.state.budget = budget

    tui.write("  fetching…")
    if looks_like_unicode_emoji(spec):
        image = await emoji_image(spec)
    else:
        slack = tui.state.slack
        if slack is None:
            tui.write(
                "  no Slack client available for name lookup; pass the literal "
                "emoji character instead"
            )
            return
        image = await emoji_image(spec, slack=slack)

    tui.write("  tracing…")
    if colors > 1:
        traced = await asyncio.to_thread(
            partial(
                image_trace.trace_image_colors,
                image,
                tui.state.rect,
                layers=colors,
                **tui.trace_options(default_fill=True),
            )
        )
        layers = [Layer(hue=t.hue, segments=t.segments) for t in traced if t.segments]
        for layer in layers:
            tui.write(f"    hue {layer.hue:3d}: {len(layer.segments)} segments")
        if len(layers) < colors:
            tui.write(
                f"    {spec} only yielded {len(layers)} distinguishable "
                "colour(s); Slack forces saturation and lightness so similar "
                "hues collapse"
            )
    else:
        result = await asyncio.to_thread(
            partial(
                image_trace.trace_image,
                image,
                tui.state.rect,
                **tui.trace_options(default_fill=True),
            )
        )
        tui.write(f"    {result.contours} contours -> {len(result.segments)} segments")
        layers = tui.layers_from_segments(result.segments)
    if not layers:
        tui.write(f"  {spec} traced to nothing")
        return
    await tui.draw(layers)

async def _send_reacji(tui: Tui) -> None:
    emoji = await tui.ask("  emoji name (without colons)", "tada")
    if not emoji:
        return
    session = tui.state.multi.sessions[0]
    payload = build_reacji(from_peer_id=session.attendee_id, emoji=emoji)
    sent = await session.signaling.send_data_message("data-channel", payload)
    tui.write(f"  reaction :{emoji}: {'sent' if sent else 'dropped by the rate limiter'}")


async def _toggle_watch(tui: Tui) -> None:
    state = tui.state
    if state.watching:
        if state._watch_unsub:
            state._watch_unsub()
            state._watch_unsub = None
        state.watching = False
        tui.write("  stopped watching")
        return

    session = state.multi.sessions[0]
    mine = set(state.attendees)

    def show(message: DataMessage) -> None:
        if message.sender_attendee_id in mine:
            return  # our own messages are echoed back; Slack ignores them too
        try:
            parsed = parse_peer_message(message.data)
        except Exception:
            tui.write(
                f"  [{message.topic}] {message.sender_attendee_id[:8]}: "
                f"<{len(message.data)}B unparsed>"
            )
            return
        tui.write(
            f"  [{message.topic}] {message.sender_attendee_id[:8]}: "
            f"{describe_peer_message(parsed)}"
        )

    state._watch_unsub = session.signaling.on_data_message(show)
    state.watching = True
    tui.write("  watching peer messages (choose [9] again to stop)")


async def _quit(tui: Tui) -> None:
    tui.running = False
    tui.write("  leaving the huddle…")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


async def run_tui(
    accounts: Sequence[Config],
    *,
    channel_id: str,
    colors: int = 1,
    console: Console | None = None,
    share_timeout: float = 120.0,
    hold: float = 15.0,
) -> int:
    """Join, then hand control to the interactive menu."""
    from .slack import SlackClient

    console = console or StdConsole()
    console.write(
        f"joining as {colors} attendee(s)… (each is a visible participant)"
    )
    multi = await MultiSession.join_all(
        accounts, channel_id=channel_id, count=colors
    )
    # Held open for the whole session so `:emoji-name:` lookups work from the
    # menu without re-authenticating each time.
    slack = SlackClient(accounts[0])
    state = TuiState(multi=multi, hold=hold, slack=slack)
    if colors > 1:
        # Spread the default hues around the wheel so the colours are
        # immediately distinguishable.
        state.hues = [int(i * 360 / colors) for i in range(colors)]
    tui = Tui(state, console, share_timeout=share_timeout)
    try:
        await tui.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.write("interrupted")
    finally:
        if state._watch_unsub:
            with contextlib.suppress(Exception):
                state._watch_unsub()
        with contextlib.suppress(Exception):
            await slack.close()
        await multi.close()
        console.write("left")
    return 0


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin
    from .cli import main as cli_main

    return cli_main(["tui", *(argv if argv is not None else sys.argv[1:])])

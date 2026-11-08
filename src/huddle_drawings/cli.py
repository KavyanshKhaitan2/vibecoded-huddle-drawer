"""Command line interface.

    huddle-drawings whoami
    huddle-drawings join       C0123456789 --seconds 30
    huddle-drawings draw       U0123456789 --shape heart   # DM with a user
    huddle-drawings shares     C0123456789
    huddle-drawings draw       C0123456789 --shape heart --hue 320 --hold 20
    huddle-drawings draw       C0123456789 --svg-path "M .1 .1 L .9 .9" --hue 60
    huddle-drawings watch      C0123456789
    huddle-drawings tui        U0A7776A2MT --colors 3
    huddle-drawings autojoin   --then-draw --shape star --hold 30

``autojoin`` is the hands-off mode: it waits to be invited to a huddle and
walks in by itself, so you never need a channel id.

Every command joins a **real** huddle as the account whose session tokens are
in ``.env``, and is visible to everyone in it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
from collections.abc import Sequence

from . import geometry, shapes
from .autojoin import AutoJoiner
from .multi import Layer, MultiSession
from . import image_trace
from .image_trace import DEFAULT_MAX_SEGMENTS
from .config import (
    LINE_CLEAR_WAIT,
    CHIME_DATA_MESSAGE_RATE,
    Config,
    ConfigError,
    load_accounts,
)
from .peer_messages import describe_peer_message, parse_peer_message
from .realtime import HuddleInvite
from .session import HuddleSession
from .slack import SlackClient
from .types import DataMessage, HuddleError, SlackApiError

log = logging.getLogger("huddle_drawings")

#: Named shapes, defined in shapes.py.
SHAPES = shapes.SHAPES


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # aiortc and websockets are chatty at DEBUG.
    if verbosity < 3:
        for noisy in ("aioice", "aiortc", "websockets.client"):
            logging.getLogger(noisy).setLevel(max(logging.INFO, level))


# -- commands --------------------------------------------------------------


async def cmd_whoami(config: Config, args: argparse.Namespace) -> int:
    async with SlackClient(config) as slack:
        info = await slack.auth_test()
    print(f"user:       {info.get('user')} ({info.get('user_id')})")
    print(f"team:       {info.get('team')} ({info.get('team_id')})")
    if url := info.get("url"):
        print(f"url:        {url}")
    if ent := info.get("enterprise_id"):
        print(f"enterprise: {ent}")
    print(f"api base:   {config.api_base}")
    return 0


async def cmd_shares(config: Config, args: argparse.Namespace) -> int:
    async with await HuddleSession.join(config, channel_id=args.channel) as session:
        print(f"present as {session.attendee_id}")
        if args.wait:
            print("waiting for a screen share…")
            share = await session.wait_for_share(timeout=args.wait)
            print(f"  share: stream {share.stream_id} by {share.sharer_attendee_id}")
            return 0
        sources = session.signaling.sources.values()
        if not sources:
            print("no media sources in INDEX yet")
        for src in sources:
            kind = "content share" if src.is_content_share else (src.media_type or "?")
            size = f" {src.width}x{src.height}" if src.width else ""
            print(f"  stream {src.stream_id:>4}  {kind:<14} {src.attendee_id}{size}")
    return 0


async def cmd_join(config: Config, args: argparse.Namespace) -> int:
    async with await HuddleSession.join(
        config, channel_id=args.channel, muted=not args.unmuted
    ) as session:
        print(f"joined call {session.join_info.call_id} as {session.attendee_id}")
        print(f"holding for {args.seconds:g}s — Ctrl-C to leave early")
        with contextlib.suppress(asyncio.CancelledError, KeyboardInterrupt):
            await asyncio.sleep(args.seconds)
    print("left")
    return 0


async def cmd_watch(config: Config, args: argparse.Namespace) -> int:
    """Log every peer message in the huddle.

    Our own messages are echoed back by Chime but ignored by Slack clients;
    we skip them here too.
    """
    async with await HuddleSession.join(config, channel_id=args.channel) as session:
        me = session.attendee_id

        def show(msg: DataMessage) -> None:
            if msg.sender_attendee_id == me:
                return
            try:
                parsed = parse_peer_message(msg.data)
            except Exception:
                print(f"[{msg.topic}] {msg.sender_attendee_id}: <{len(msg.data)}B unparsed>")
                return
            print(f"[{msg.topic}] {msg.sender_attendee_id}: {describe_peer_message(parsed)}")

        session.signaling.on_data_message(show)
        print(f"watching as {me} for {args.seconds:g}s — Ctrl-C to stop")
        with contextlib.suppress(asyncio.CancelledError, KeyboardInterrupt):
            await asyncio.sleep(args.seconds)
    return 0


def _wants_fill(args: argparse.Namespace) -> bool:
    """Whether to fill regions rather than outline them. Defaults to filling.

    An outline renders as a hollow wireframe, which reads as wrong for almost
    anything recognisable, and filling is usually *cheaper* too: one hairpin
    Bezier covers a band two pen widths tall, where an outline pays per
    contour step (a party popper: 91 segments filled against 216 outlined).

    ``--no-fill`` is there for line art and sketches, where the drawing really
    is its outlines.
    """
    if args.fill is not None:
        return bool(args.fill)
    return True


def _trace_mode(args: argparse.Namespace) -> str:
    """The trace mode to use, honouring --fill and an explicit --trace-mode."""
    if args.trace_mode != "auto":
        return args.trace_mode  # an explicit choice wins
    return "fill" if _wants_fill(args) else "auto"


def _trace_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Tracer options for this invocation, via the shared fill policy."""
    return image_trace.trace_options(
        fill=_wants_fill(args),
        mode=args.trace_mode,
        min_contour_length=args.min_contour_length,
        fill_row_height=args.fill_row_height,
        max_segments=args.max_segments,
        invert=args.invert,
    )


def _maybe_preview(args: argparse.Namespace, layers: Sequence[Layer]) -> bool:
    """Write --preview if asked. Returns True when we should not join."""
    if not getattr(args, "preview", None):
        return False
    from .preview import render_preview

    render_preview(layers, args.preview)
    total = sum(len(layer.segments) for layer in layers)
    print(
        f"wrote {args.preview}: {len(layers)} colour layer(s), {total} segments "
        f"(~{total / CHIME_DATA_MESSAGE_RATE:.1f}s of the meeting's budget). "
        "Nothing was sent and no huddle was joined."
    )
    return True

def _source_label(args: argparse.Namespace) -> str:
    """What the user asked to draw, for messages."""
    return (
        getattr(args, "emoji", None)
        or getattr(args, "image", None)
        or getattr(args, "text", None)
        or getattr(args, "svg_path", None)
        or getattr(args, "shape", None)
        or "the drawing"
    )


async def _emoji_image(args: argparse.Namespace, config: Config):
    """Resolve --emoji to an image.

    A literal emoji character is rendered from a local font and needs no
    network. A Slack name needs `emoji.list`, so a short-lived client is
    opened just for the lookup -- doing it before joining means a typo fails
    without putting an attendee into a real call.
    """
    from .emoji import emoji_image, looks_like_unicode_emoji

    spec = args.emoji
    if looks_like_unicode_emoji(spec):
        return await emoji_image(spec)
    async with SlackClient(config) as slack:
        return await emoji_image(spec, slack=slack)

def _target_rect(args: argparse.Namespace) -> geometry.Rect:
    """The sub-rect of the share to draw into, as normalized 0..1 bounds."""
    return geometry.Rect(args.x, args.y, args.x + args.width, args.y + args.height)


async def _segments_for(
    args: argparse.Namespace, config: Config
) -> list[geometry.Segment]:
    """Build the Bézier segments to draw from whichever source was asked for.

    One segment is one Chime data message, and the whole meeting shares about
    150 of those per second, so the count this returns is the real cost.
    """
    rect = _target_rect(args)
    if getattr(args, "emoji", None):
        from . import image_trace

        image = await _emoji_image(args, config)
        result = image_trace.trace_image(image, rect, **_trace_kwargs(args))
        log.info(
            "emoji %s: %d contours -> %d segments",
            args.emoji, result.contours, len(result.segments),
        )
        return result.segments
    if getattr(args, "image", None):
        # Imported lazily: image tracing pulls in Pillow and numpy, and the
        # rest of the CLI should work without paying for them.
        from . import image_trace

        result = image_trace.trace_image(args.image, rect, **_trace_kwargs(args))
        log.info(
            "traced %s: %d contours -> %d segments (~%.1fs of messages)",
            args.image,
            result.contours,
            len(result.segments),
            result.estimated_seconds(),
        )
        return result.segments
    if getattr(args, "svg_path", None):
        return shapes.svg_path_segments(args.svg_path, rect)
    if getattr(args, "text", None):
        return shapes.text_in(args.text, rect)
    return shapes.shape_segments(args.shape, rect, simplify=args.simplify)


async def _layers_for(args: argparse.Namespace, config: Config) -> list[Layer]:
    """Build the colour layers to draw, one per Slack account.

    Only `--image` can be decomposed into colours; a shape, SVG path or text
    is a single stroke, so asking for several colours there is an error rather
    than a silent single-colour drawing.
    """
    rect = _target_rect(args)
    source = getattr(args, "emoji", None) or getattr(args, "image", None)
    if not source:
        raise HuddleError(
            "--colors only applies to --image or --emoji: a shape, SVG path "
            "or text is a single stroke. Draw it with --hue instead."
        )
    from . import image_trace

    if getattr(args, "emoji", None):
        source = await _emoji_image(args, config)

    traced = image_trace.trace_image_colors(
        source, rect, layers=args.colors, **_trace_kwargs(args)
    )
    layers = [Layer(hue=t.hue, segments=t.segments) for t in traced if t.segments]
    if not layers:
        raise HuddleError(f"{_source_label(args)} traced to nothing in any colour")

    # Asking for N colours and silently getting fewer is confusing, and the
    # reasons are specific enough to be worth naming.
    if len(layers) < args.colors:
        reasons = []
        if any(t.is_background for t in traced):
            reasons.append("one was the background")
        if any(getattr(t, "merged_clusters", 0) for t in traced):
            reasons.append("some were too close in hue to tell apart")
        if any(not t.segments for t in traced):
            reasons.append("some traced to nothing")
        log.warning(
            "asked for %d colour(s) but %s yielded %d%s. Slack forces "
            "saturation 100%% and lightness 75%%, so only the hue survives "
            "and near-duplicate colours are indistinguishable",
            args.colors,
            _source_label(args),
            len(layers),
            f" ({'; '.join(reasons)})" if reasons else "",
        )

    if args.hues:
        try:
            overrides = [int(h) for h in args.hues.split(",") if h.strip()]
        except ValueError as exc:
            raise HuddleError(f"--hues must be comma-separated integers: {exc}") from exc
        for hue in overrides:
            if not 0 <= hue <= 359:
                raise HuddleError(f"--hues value {hue} is outside 0-359")
        if len(overrides) < len(layers):
            log.warning(
                "--hues gave %d hue(s) for %d layer(s); the rest keep their "
                "traced hues",
                len(overrides),
                len(layers),
            )
        layers = [
            Layer(hue=overrides[i] if i < len(overrides) else layer.hue,
                  segments=layer.segments)
            for i, layer in enumerate(layers)
        ]

    for layer in layers:
        log.info("layer hue=%3d  %4d segments", layer.hue, len(layer.segments))
    return layers


async def cmd_draw_colors(config: Config, args: argparse.Namespace) -> int:
    """Draw an image in several colours, one Slack account per colour."""
    layers = await _layers_for(args, config)

    accounts = load_accounts(args.env_file, args.accounts_file)
    if _maybe_preview(args, layers):
        return 0
    total = sum(len(layer.segments) for layer in layers)
    if len(layers) < args.colors:
        print(
            f"note: asked for {args.colors} colours, {_source_label(args)} yielded "
            f"{len(layers)} (run with -v for why)"
        )
    print(
        f"prepared {len(layers)} colour layer(s), {total} segments total "
        f"(~{total / CHIME_DATA_MESSAGE_RATE:.1f}s of the meeting's shared "
        f"message budget)"
    )

    # One attendee per colour. With a single account these are multidevice
    # attendees of it; with several accounts they are spread across them.
    # Each is a visible participant in the call.
    async with await MultiSession.join_all(
        accounts, channel_id=args.channel, count=len(layers)
    ) as multi:
        if len(multi) < len(layers):
            print(
                f"only {len(multi)} of {len(layers)} accounts joined; "
                f"drawing the {len(multi)} largest colour layers",
                file=sys.stderr,
            )
            layers = layers[: len(multi)]
        print(f"present as {', '.join(s.attendee_id[:8] for s in multi.sessions)}")
        print("waiting for a screen share…")
        share = await multi.wait_for_share(timeout=args.share_timeout)
        print(f"drawing on the share by {share.sharer_attendee_id}")
        await multi.draw_layers(layers, share, hold=args.hold)
        print("done — the lines fade ~3.7s after END and cannot be erased sooner")
    return 0


async def cmd_draw(config: Config, args: argparse.Namespace) -> int:
    if getattr(args, "colors", 1) > 1:
        return await cmd_draw_colors(config, args)
    segments = await _segments_for(args, config)
    if not segments:
        print("nothing to draw", file=sys.stderr)
        return 1
    hue = args.hue if args.hue is not None else 200
    if _maybe_preview(args, [Layer(hue=hue, segments=segments)]):
        return 0
    print(
        f"prepared {len(segments)} segments "
        f"(~{len(segments) / CHIME_DATA_MESSAGE_RATE:.1f}s of the meeting's "
        f"message budget)"
    )

    async with await HuddleSession.join(config, channel_id=args.channel) as session:
        print(f"present as {session.attendee_id}; waiting for a screen share…")
        share = await session.wait_for_share(timeout=args.share_timeout)
        print(f"drawing on the share by {share.sharer_attendee_id}")

        drawer = session.drawer(share, hue=args.hue)
        await drawer.stroke(segments, end=False)

        if args.hold > 0:
            # Nothing erases a line early, and with no END the renderer fades
            # it 5s after the last non-POSITION message. Keep-alives hold it.
            print(f"holding the drawing for {args.hold:g}s")
            drawer.start_keep_alive()
            with contextlib.suppress(asyncio.CancelledError, KeyboardInterrupt):
                await asyncio.sleep(args.hold)
            await drawer.stop_keep_alive()

        await drawer.end_stroke()
        print("done — the line fades ~3.7s after END and cannot be erased sooner")
    return 0


# -- argument parsing ------------------------------------------------------


async def cmd_autojoin(config: Config, args: argparse.Namespace) -> int:
    """Wait to be invited to a huddle, then walk in by itself.

    Start this, then invite the account from any Slack client.
    """
    # Build the drawing up front: a bad image path or SVG should fail before
    # we join a real call, not after.
    segments = await _segments_for(args, config) if args.then_draw else []

    async def handler(session: HuddleSession, invite: HuddleInvite) -> None:
        print(f"joined {invite.call_id} (invited by {invite.sender_user_id})")
        if segments:
            print("waiting for a screen share to draw on…")
            try:
                share = await session.wait_for_share(timeout=args.share_timeout)
            except TimeoutError:
                print("nobody shared a screen; nothing to draw on")
            else:
                drawer = session.drawer(share, hue=args.hue)
                await drawer.stroke(segments, end=False)
                if args.hold > 0:
                    drawer.start_keep_alive()
                    await asyncio.sleep(args.hold)
                    await drawer.stop_keep_alive()
                await drawer.end_stroke()
                print("drawn")
        if args.stay > 0:
            print(f"staying for {args.stay:g}s")
            with contextlib.suppress(asyncio.CancelledError, KeyboardInterrupt):
                await asyncio.sleep(args.stay)

    joiner = AutoJoiner(
        config=config,
        handler=handler,
        allow_from=set(args.only_from) if args.only_from else None,
        allow_channels=set(args.only_channel) if args.only_channel else None,
        decline_rejected=args.decline_others,
        accept_invite=args.accept_invite,
        once=args.once,
    )
    print("listening for huddle invites — invite this account from Slack. Ctrl-C to stop.")
    with contextlib.suppress(KeyboardInterrupt, asyncio.CancelledError):
        await joiner.run()
    return 0


async def cmd_video(config: Config, args: argparse.Namespace) -> int:
    """Play a video onto a screen share as a flipbook.

    The attendee pool is sized from the frame rate, not chosen by the user: a
    frame's line cannot be erased and takes ~8.5s to fade, so a fresh attendee
    is needed for every frame drawn within that window. See
    `video.attendees_needed`.
    """
    from . import video

    needed = video.attendees_needed(args.fps, colors=args.colors)
    budget = args.max_segments or video.segments_per_frame(args.fps)
    accounts = load_accounts(args.env_file, args.accounts_file)

    print(
        f"{args.fps:g} fps needs {needed} attendee(s) "
        f"({args.colors} colour(s) x ceil({args.fps:g} x "
        f"{LINE_CLEAR_WAIT:g}s fade)) and allows ~{budget} segments per frame"
    )
    if needed > args.max_attendees:
        print(
            f"that is more than --max-attendees {args.max_attendees}. Each "
            "attendee is a visible participant in the huddle and runs its own "
            "WebRTC leg, so this is capped on purpose — lower --fps, lower "
            "--colors, or raise --max-attendees.",
            file=sys.stderr,
        )
        return 2

    rect = _target_rect(args)
    print(f"decoding {args.video} at {args.fps:g} fps…")
    images = list(
        video.decode_frames(
            args.video,
            fps=args.fps,
            max_frames=args.max_frames,
            start=args.start,
            duration=args.duration,
        )
    )
    if not images:
        print(f"{args.video} yielded no frames", file=sys.stderr)
        return 1
    print(f"tracing {len(images)} frame(s)…")
    frames = list(
        video.trace_frames(
            images,
            rect,
            colors=args.colors,
            max_segments=budget,
            fill=_wants_fill(args),
        )
    )
    total = sum(f.segments for f in frames)
    print(
        f"{len(frames)} frame(s), {total} segments total, "
        f"{total / max(1, len(frames)):.0f} per frame "
        f"(~{len(frames) / args.fps:.0f}s of playback)"
    )

    if args.preview:
        from .preview import render_preview

        written = []
        for frame in frames[: args.preview_frames]:
            path = f"{args.preview.removesuffix('.png')}_{frame.index:04d}.png"
            render_preview(frame.layers, path)
            written.append(path)
        print(
            f"wrote {len(written)} preview frame(s); nothing was sent and no "
            "huddle was joined"
        )
        return 0

    async with await MultiSession.join_all(
        accounts, channel_id=args.channel, count=needed
    ) as multi:
        print(f"present as {', '.join(s.attendee_id[:8] for s in multi.sessions)}")
        print("waiting for a screen share…")
        share = await multi.wait_for_share(timeout=args.share_timeout)
        print(f"playing on the share by {share.sharer_attendee_id}")

        stop = asyncio.Event()

        def report(frame: video.Frame, when: float) -> None:
            print(
                f"  frame {frame.index + 1}/{len(frames)} "
                f"({frame.segments} segments) at {when:5.1f}s",
                flush=True,
            )

        with contextlib.suppress(KeyboardInterrupt):
            stats = await video.play(
                multi,
                frames,
                share,
                fps=args.fps,
                colors=args.colors,
                loop=args.loop,
                stop=stop,
                on_frame=report if args.verbose else None,
            )
            print(
                f"played {stats.frames_drawn} frame(s), skipped "
                f"{stats.frames_skipped}, {stats.segments_sent} segments in "
                f"{stats.elapsed:.1f}s"
            )
            if stats.frames_skipped:
                print(
                    "skipped frames mean the attendee pool ran dry — a line "
                    "cannot be erased, so drawing early would have brought the "
                    "previous frame back. Lower --fps or raise --max-attendees.",
                    file=sys.stderr,
                )
    return 0


async def cmd_tui(config: Config, args: argparse.Namespace) -> int:
    """Join once, then take instructions from a menu."""
    from .tui import run_tui

    accounts = load_accounts(args.env_file, args.accounts_file)
    if args.colors < 1:
        print("--colors must be at least 1", file=sys.stderr)
        return 2
    return await run_tui(
        accounts,
        channel_id=args.channel,
        colors=args.colors,
        share_timeout=args.share_timeout,
        hold=args.hold,
    )


def _add_drawing_source_args(p: argparse.ArgumentParser) -> None:
    """The four mutually exclusive ways to say *what* to draw, plus placement."""
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--emoji",
        metavar="EMOJI",
        help="draw an emoji: a literal character (🎉) or a Slack name "
        "(:squirrel:, :my-custom-emoji:). Emoji trace well — flat, "
        "high-contrast shapes — and look best with --colors",
    )
    src.add_argument(
        "--image",
        metavar="PATH",
        help="trace an image file and draw it (PNG/JPG/…). Works best on logos, "
        "line art and high-contrast silhouettes; photographs trace poorly",
    )
    src.add_argument("--shape", choices=SHAPES, default="heart")
    src.add_argument("--svg-path", help="an SVG path 'd' string (no arcs)")
    src.add_argument("--text", help="draw this text with the built-in stroke font")

    p.add_argument(
        "--trace-mode",
        choices=("auto", "outline", "threshold", "fill"),
        default="auto",
        help="how to trace --image (default: auto)",
    )
    p.add_argument(
        "--max-segments",
        type=int,
        default=DEFAULT_MAX_SEGMENTS,
        help="hard cap on segments when tracing. One segment is one Chime "
        "message and the whole meeting shares ~150/s, so the default "
        f"{DEFAULT_MAX_SEGMENTS} is roughly "
        f"{DEFAULT_MAX_SEGMENTS / CHIME_DATA_MESSAGE_RATE:.0f}s of solid "
        "sending. Lower it if others are drawing too, or to finish sooner "
        f"(default: {DEFAULT_MAX_SEGMENTS})",
    )
    p.add_argument(
        "--invert",
        action="store_true",
        help="trace the light regions of --image instead of the dark ones",
    )
    p.add_argument(
        "--colors",
        "--colours",
        dest="colors",
        type=int,
        default=1,
        metavar="N",
        help="draw --image in N colours. Slack gives one line in one hue per "
        "attendee, so this joins N times with multidevice=true to become N "
        "attendees — one account is enough. Default 1",
    )
    p.add_argument(
        "--accounts-file",
        default=None,
        metavar="PATH",
        help="JSON file of accounts, instead of numbered env vars",
    )
    p.add_argument(
        "--hues",
        default=None,
        metavar="H,H,…",
        help="override the traced hues with this comma-separated list of "
        "degrees (0-359)",
    )
    fill = p.add_mutually_exclusive_group()
    fill.add_argument(
        "--fill",
        dest="fill",
        action="store_true",
        default=None,
        help="fill solid regions instead of just outlining them, using the "
        "doc's hairpin trick. Usually *cheaper* than an outline as well as "
        "looking right. Default: on for --emoji, off for --image (filling "
        "loses line art)",
    )
    fill.add_argument(
        "--no-fill", dest="fill", action="store_false", help="force outlines",
    )
    p.add_argument(
        "--preview",
        metavar="PATH",
        help="render the drawing to a PNG locally and exit without joining, "
        "so you can see it before spending the meeting's message budget",
    )
    p.add_argument(
        "--min-contour-length",
        type=float,
        default=None,
        metavar="L",
        help="smallest feature to trace, relative to the frame. In --fill "
        f"mode this is the narrowest band laid down (default "
        f"{image_trace.FILL_MIN_RUN} "
        "when filling, higher otherwise to suppress speckle)",
    )
    p.add_argument(
        "--fill-row-height",
        type=float,
        default=None,
        metavar="H",
        help="band height for --fill, in 0..1 units. Smaller fills more "
        "densely for more messages; the default aims at just-overlapping "
        "rows for a 7px pen",
    )
    p.add_argument(
        "--simplify",
        type=float,
        default=0.01,
        help="curve simplification tolerance for --shape; higher is cheaper "
        "and coarser (default: 0.01)",
    )
    p.add_argument("--hue", type=int, default=None, metavar="0-359",
                   help="hue in degrees. The renderer always draws "
                        "hsla(hue,100%%,75%%), so only the hue is controllable: "
                        "every colour is a pastel and nothing can be dark, grey "
                        "or white")
    p.add_argument("--x", type=float, default=0.25, help="left edge, 0..1")
    p.add_argument("--y", type=float, default=0.25, help="top edge, 0..1")
    p.add_argument("--width", type=float, default=0.5, help="width, 0..1")
    p.add_argument("--height", type=float, default=0.5, help="height, 0..1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="huddle-drawings",
        description=(
            "Draw on Slack huddle screen shares with no browser and no Slack app. "
            "Joins a real call as a real user; use a dedicated account in a huddle "
            "whose participants are in on it."
        ),
    )
    # Global options are also attached to every subcommand, so both
    # `huddle-drawings -v join C123` and `huddle-drawings join C123 -v` work.
    # argparse otherwise only accepts them before the subcommand, which is a
    # trap when you reach for -v precisely because something is going wrong.
    # SUPPRESS matters: without it the subparser would write its own default
    # into the namespace and silently clobber a value given before the
    # subcommand, so `-vvv join` would end up non-verbose.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v", "--verbose", action="count", default=argparse.SUPPRESS
    )
    common.add_argument(
        "--env-file", default=argparse.SUPPRESS, help="path to a .env file"
    )

    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--env-file", default=None, help="path to a .env file")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=lambda **kw: argparse.ArgumentParser(parents=[common], **kw))

    def channel_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "channel",
            metavar="CHANNEL_OR_USER",
            help="channel (C…), DM (D…) or group (G…) id, or a user id "
            "(U…/W…) to start a huddle in the DM with that person",
        )

    p = sub.add_parser("whoami", help="check the session tokens with auth.test")
    p.set_defaults(func=cmd_whoami)

    p = sub.add_parser("join", help="join a huddle and sit in it")
    channel_arg(p)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--unmuted", action="store_true", help="join unmuted (still silent)")
    p.set_defaults(func=cmd_join)

    p = sub.add_parser("shares", help="list the huddle's media sources")
    channel_arg(p)
    p.add_argument(
        "--wait",
        type=float,
        default=None,
        metavar="SECONDS",
        help="wait up to SECONDS for a screen share instead of listing now",
    )
    p.set_defaults(func=cmd_shares)

    p = sub.add_parser("watch", help="log peer messages seen in the huddle")
    channel_arg(p)
    p.add_argument("--seconds", type=float, default=60.0)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("draw", help="draw on a screen share")
    channel_arg(p)
    _add_drawing_source_args(p)
    p.add_argument(
        "--hold",
        type=float,
        default=0.0,
        help="keep the drawing alive this long. Without this the renderer fades "
        "the line ~3.7s after the last END, and nothing can erase it sooner",
    )
    p.add_argument("--share-timeout", type=float, default=120.0)
    p.set_defaults(func=cmd_draw)

    p = sub.add_parser(
        "video",
        help="play a video onto a screen share as a flipbook",
        description=(
            "Draws a video frame by frame. A frame's line cannot be erased and "
            "takes ~8.5s to fade, so every frame drawn inside that window needs "
            "its own attendee: the pool is sized as ceil(fps x 8.5) x colours. "
            "One account is enough (it joins repeatedly with multidevice), but "
            "each attendee is a visible participant in the huddle."
        ),
    )
    channel_arg(p)
    p.add_argument("video", help="path to a video file")
    p.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="frames per second. Higher is smoother but needs more attendees "
        "and allows fewer segments per frame: 1 fps is ~9 attendees and ~120 "
        "segments/frame, 2 fps is ~18 and ~60 (default: 1)",
    )
    p.add_argument("--colors", "--colours", dest="colors", type=int, default=1,
                   metavar="N", help="colour layers per frame; multiplies the "
                   "attendee count (default: 1)")
    p.add_argument(
        "--max-attendees",
        type=int,
        default=12,
        help="refuse to join more than this many attendees. They are all "
        "visible participants and each runs a WebRTC leg (default: 12)",
    )
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--start", type=float, default=0.0, metavar="SECONDS")
    p.add_argument("--duration", type=float, default=None, metavar="SECONDS")
    p.add_argument("--loop", action="store_true", help="repeat until Ctrl-C")
    p.add_argument("--max-segments", type=int, default=None,
                   help="per-frame cap; default is the fps's fair share of the "
                   "meeting's ~150/s budget")
    p.add_argument("--preview", default=None, metavar="PATH",
                   help="write preview PNGs instead of joining")
    p.add_argument("--preview-frames", type=int, default=8)
    p.add_argument("--accounts-file", default=None, metavar="PATH")
    p.add_argument("--share-timeout", type=float, default=120.0)
    fillv = p.add_mutually_exclusive_group()
    fillv.add_argument("--fill", dest="fill", action="store_true", default=None)
    fillv.add_argument("--no-fill", dest="fill", action="store_false")
    p.add_argument("--trace-mode", choices=("auto","outline","threshold","fill"),
                   default="auto")
    p.add_argument("--invert", action="store_true")
    p.add_argument("--min-contour-length", type=float, default=None)
    p.add_argument("--fill-row-height", type=float, default=None)
    p.add_argument("--x", type=float, default=0.2)
    p.add_argument("--y", type=float, default=0.15)
    p.add_argument("--width", type=float, default=0.6)
    p.add_argument("--height", type=float, default=0.7)
    p.set_defaults(func=cmd_video)

    p = sub.add_parser(
        "tui",
        help="join, then interactively choose what to draw",
        description=(
            "Joins the huddle once and keeps the session alive while you pick "
            "actions from a menu. Joining costs a couple of seconds per "
            "attendee, and the menu also manages the fade timing for you: a "
            "line cannot be erased, and drawing before the previous one has "
            "finished fading brings all of it back at full opacity."
        ),
    )
    channel_arg(p)
    p.add_argument(
        "--colors",
        "--colours",
        dest="colors",
        type=int,
        default=1,
        metavar="N",
        help="join as N attendees so you can draw in N colours at once "
        "(one account is enough; it uses multidevice). Default 1",
    )
    p.add_argument("--accounts-file", default=None, metavar="PATH")
    p.add_argument("--share-timeout", type=float, default=120.0)
    p.add_argument(
        "--hold",
        type=float,
        default=15.0,
        help="how long each drawing is held alive with keep-alives "
        "(default: 15)",
    )
    p.set_defaults(func=cmd_tui)

    p = sub.add_parser(
        "autojoin",
        help="wait to be invited to a huddle, then join automatically",
        description=(
            "Listens on Slack's realtime gateway for a huddle_invite aimed at this "
            "account and joins straight from the invite's Chime blob. Start it, then "
            "invite the account from any Slack client."
        ),
    )
    p.add_argument(
        "--only-from",
        action="append",
        default=[],
        metavar="U012345",
        help="only accept invites from this user id (repeatable)",
    )
    p.add_argument(
        "--only-channel",
        action="append",
        default=[],
        metavar="C012345",
        help="only accept invites for this channel id (repeatable)",
    )
    p.add_argument(
        "--decline-others",
        action="store_true",
        help="actively decline invites that the filters reject",
    )
    p.add_argument(
        "--accept-invite",
        action="store_true",
        help="also answer rooms.inviteResponse with an accept — the accept value is "
        "an UNVERIFIED guess, and joining the Chime meeting already puts us in "
        "the huddle, so this is not needed",
    )
    p.add_argument("--once", action="store_true", help="exit after the first huddle")
    p.add_argument("--stay", type=float, default=0.0, help="stay this long once in")
    p.add_argument(
        "--then-draw",
        action="store_true",
        help="after joining, wait for a screen share and draw on it",
    )
    _add_drawing_source_args(p)
    p.add_argument("--hold", type=float, default=0.0)
    p.add_argument("--share-timeout", type=float, default=120.0)
    p.set_defaults(func=cmd_autojoin)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    hue = getattr(args, "hue", None)
    if hue is not None and not 0 <= hue <= 359:
        parser.error("--hue must be between 0 and 359 (degrees on the HSL wheel)")

    try:
        config = Config.from_env(args.env_file)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        return asyncio.run(args.func(config, args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except SlackApiError as exc:
        print(f"Slack API error: {exc}", file=sys.stderr)
        return 1
    except HuddleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

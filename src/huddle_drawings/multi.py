"""Multi-colour drawing: several attendees at once, one hue each.

Slack's renderer keeps **one line in one hue per attendee**, so a picture in N
colours needs N attendees. The useful part is that those do **not** have to be
N different Slack accounts:

**``rooms.join`` with ``multidevice=true`` returns a fresh ``AttendeeId`` on
every call**, so one account can be several attendees at once and draw several
lines in several hues. Verified live on 2026-11-08: three ``multidevice=true``
joins for one user in one room returned three distinct attendee ids, and two
such sessions drawing with ``hue=0`` and ``hue=210`` rendered as a red line
and a blue line simultaneously.

This corrects ``docs/huddles-and-chime.md``, which states that the same user
gets the same AttendeeId "whatever the params" and lists what ``multidevice``
does as an open question. Without the flag that claim holds -- extra sessions
share one attendee and so one line -- which is why the flag matters.

Several accounts are still supported, and are the only option if a workspace
ever rejects ``multidevice``.

Two protocol facts shape everything here:

* **The data-message budget is per *meeting*, not per sender.** Chime allows
  about 150 messages/second across every attendee together, bursting to ~200,
  and silently drops the rest. So the accounts must share a single
  :class:`~huddle_drawings.ratelimit.TokenBucket`; giving each its own would
  overrun the meeting by a factor of N and lose strokes with no error. That
  bucket is sized at :data:`~huddle_drawings.config.DATA_MESSAGE_HEADROOM`
  (90%) of the limit, so the remaining tenth is left for the rest of the
  meeting -- other people's drawings, cursors and reactions.
* **Lines fade, and cannot be erased.** A line starts fading 5 s after that
  attendee's last non-``POSITION`` message (0.7 s after an ``END``), then
  fades for 3 s. For all the colours to be on screen *together*, every layer
  that has finished drawing must be kept alive while the others are still
  going -- and those keep-alives come out of the same shared budget.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from .config import (
    CLIENT_DATA_MESSAGE_BURST,
    CLIENT_DATA_MESSAGE_RATE,
    Config,
)
from .draw import DrawClient
from .geometry import Segment
from .ratelimit import TokenBucket
from .session import HuddleSession
from .types import HuddleError, StreamSource

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Layer:
    """One colour of a drawing: a hue plus the segments to draw in it."""

    hue: int
    segments: Sequence[Segment]

    def __len__(self) -> int:
        return len(self.segments)


@dataclass(slots=True)
class MultiSession:
    """Several accounts in one huddle, each able to draw its own colour.

    Build with :meth:`join_all`. All sessions share one rate limiter, because
    Chime's limit is per meeting.
    """

    sessions: list[HuddleSession]
    limiter: TokenBucket

    @classmethod
    async def join_all(
        cls,
        accounts: Sequence[Config],
        *,
        channel_id: str,
        count: int | None = None,
        muted: bool = True,
        rate: float = CLIENT_DATA_MESSAGE_RATE,
        capacity: float = CLIENT_DATA_MESSAGE_BURST,
        require_all: bool = False,
        multidevice: bool | None = None,
    ) -> MultiSession:
        """Join ``channel_id`` as ``count`` separate attendees, concurrently.

        ``count`` defaults to the number of ``accounts``. When more attendees
        are wanted than there are accounts, the accounts are cycled and each
        session asks ``rooms.join`` for a **fresh attendee** via
        ``multidevice=true`` -- which is what lets a single account draw in
        several colours.

        ``multidevice`` defaults to ``True`` whenever ``count > 1`` and to the
        server default for a single session (so a lone session reuses the
        account's stable attendee instead of accumulating new ones).

        Sessions are brought up in parallel: each costs a ``rooms.join``, a
        websocket, a TURN allocation and a DTLS handshake, which is a couple
        of seconds serially per colour.

        By default a session that fails to join is skipped with a warning and
        the rest proceed -- losing one layer of a picture beats losing all of
        it. ``require_all=True`` makes any failure fatal.
        """
        if not accounts:
            raise HuddleError("no accounts configured")
        count = len(accounts) if count is None else count
        if count < 1:
            raise HuddleError(f"count must be at least 1, got {count}")
        if multidevice is None:
            multidevice = count > 1

        limiter = TokenBucket(rate=rate, capacity=capacity)
        # Cycle the accounts so N attendees are spread over whatever accounts
        # exist; with one account they are all multidevice attendees of it.
        assignment = [accounts[i % len(accounts)] for i in range(count)]

        async def join(config: Config) -> HuddleSession | BaseException:
            try:
                return await HuddleSession.join(
                    config,
                    channel_id=channel_id,
                    muted=muted,
                    rate_limiter=limiter,
                    multidevice=multidevice,
                )
            except Exception as exc:  # noqa: BLE001 - reported per session below
                return exc

        results = await asyncio.gather(*(join(c) for c in assignment))

        sessions: list[HuddleSession] = []
        failures: list[tuple[Config, BaseException]] = []
        for config, result in zip(assignment, results, strict=True):
            if isinstance(result, BaseException):
                failures.append((config, result))
            else:
                sessions.append(result)

        for config, exc in failures:
            log.error("account %s could not join: %s", config.account_label, exc)

        if failures and (require_all or not sessions):
            for session in sessions:
                with contextlib.suppress(Exception):
                    await session.close()
            raise HuddleError(
                f"{len(failures)} of {count} sessions could not join "
                f"(first error: {failures[0][1]})"
            )

        attendees = {s.attendee_id for s in sessions}
        if multidevice and len(attendees) != len(sessions):
            # Without distinct attendees the colours would collapse onto one
            # line, so say so rather than drawing a confusing picture.
            log.warning(
                "%d session(s) but only %d distinct attendee(s): this workspace "
                "may not honour multidevice, so the colours will collapse onto "
                "one line per attendee",
                len(sessions),
                len(attendees),
            )
        log.info(
            "joined as %d attendee(s) across %d account(s): %s",
            len(attendees),
            len({c.xoxc_token for c in assignment}),
            ", ".join(s.attendee_id[:8] for s in sessions),
        )
        return cls(sessions=sessions, limiter=limiter)

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> MultiSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Leave every huddle. Idempotent, and never raises."""
        await asyncio.gather(
            *(self._close_one(s) for s in self.sessions), return_exceptions=True
        )
        self.sessions = []

    @staticmethod
    async def _close_one(session: HuddleSession) -> None:
        with contextlib.suppress(Exception):
            await session.close()

    def __len__(self) -> int:
        return len(self.sessions)

    # -- shares ------------------------------------------------------------

    async def wait_for_share(self, timeout: float | None = None) -> StreamSource:
        """Wait for a screen share, using whichever session sees it first.

        Every attendee receives the same ``INDEX`` frames, so any session will
        do; racing them just avoids depending on one.
        """
        if not self.sessions:
            raise HuddleError("no sessions")
        tasks = [
            asyncio.create_task(s.wait_for_share(timeout=timeout))
            for s in self.sessions
        ]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            errors = []
            for task in done:
                try:
                    return task.result()
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
            raise HuddleError(f"no session saw a screen share: {errors[0]}")
        finally:
            for task in tasks:
                task.cancel()

    # -- drawing -----------------------------------------------------------

    async def draw_layers(
        self,
        layers: Sequence[Layer],
        share: StreamSource | str,
        *,
        hold: float = 0.0,
        keep_alive_while_drawing: bool = True,
        interleave: bool = True,
        stop: asyncio.Event | None = None,
    ) -> None:
        """Draw ``layers`` concurrently, one account (one hue) per layer.

        Needs at least as many accounts as layers; extra accounts are unused
        and extra layers raise rather than being silently dropped, since
        quietly losing a colour makes a picture wrong in a way that is hard to
        see.

        All layers are drawn concurrently and against the shared rate limiter,
        so the total wall-clock time is roughly
        ``total_segments / 150`` seconds however many accounts there are --
        more accounts buy colours, not throughput.

        ``interleave`` (the default) sends the layers' segments round-robin so
        every colour is still being drawn until the last one finishes, and so
        they all fade together afterwards. Without it each layer is drawn to
        completion and the short ones go quiet early -- the renderer starts
        fading a line 5 s after its sender's last message, so the first
        colours visibly vanish while the biggest is still going. See
        :meth:`_interleave_order`.

        ``keep_alive_while_drawing`` only applies when ``interleave`` is off:
        it holds finished layers up with no-op ``EDIT``s, which works but
        depends on every keep-alive getting through a shared budget that drops
        excess silently.
        """
        if not layers:
            return
        if len(layers) > len(self.sessions):
            raise HuddleError(
                f"{len(layers)} colour layer(s) need {len(layers)} attendees, "
                f"but only {len(self.sessions)} joined. Slack keys drawing "
                "lines by attendee, so each colour needs its own attendee — "
                "join with count=<layers> (which uses multidevice to get "
                "extra attendees from one account), or ask for fewer colours."
            )

        total = sum(len(layer) for layer in layers)
        log.info(
            "drawing %d layer(s), %d segments total (~%.1fs at our %.0f/s share "
            "of the meeting's budget)",
            len(layers),
            total,
            total / self.limiter.rate,
            self.limiter.rate,
        )

        drawers = [
            session.drawer(share, hue=layer.hue)
            for session, layer in zip(self.sessions, layers, strict=False)
        ]

        if interleave:
            await self._draw_interleaved(drawers, layers, stop=stop)
        else:
            await self._draw_sequentially(
                drawers, layers, keep_alive_while_drawing=keep_alive_while_drawing
            )

        try:
            if hold > 0 and not (stop is not None and stop.is_set()):
                for drawer in drawers:
                    drawer.start_keep_alive()
                log.info("holding all %d layer(s) for %.0fs", len(layers), hold)
                if stop is None:
                    await asyncio.sleep(hold)
                else:
                    # Interruptible: whoever owns `stop` can cut the hold
                    # short (the console wires Ctrl-C to it). The lines are
                    # still ENDed below, so they fade together instead of
                    # being left to time out one by one.
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(stop.wait(), timeout=hold)
                    if stop.is_set():
                        log.info("hold interrupted; ending the lines")
        finally:
            for drawer in drawers:
                with contextlib.suppress(Exception):
                    await drawer.stop_keep_alive()
            # END every line so they fade together rather than at staggered
            # times as each one's 5s idle timer expires.
            await asyncio.gather(
                *(d.end_stroke() for d in drawers), return_exceptions=True
            )

    @staticmethod
    def _interleave_order(sizes: Sequence[int]) -> list[int]:
        """Layer indices ordered so every layer finishes at the same time.

        Each layer's ``k``-th segment is given the fractional position
        ``(k + 0.5) / n`` through the drawing and the whole lot is sorted by
        it, so a 50-segment layer and a 500-segment layer both start at the
        beginning and finish at the end, spread evenly in between.

        This is what keeps the colours alive. Drawing one layer to completion
        at a time leaves the short layers silent for the rest of the drawing,
        and the renderer starts fading a line 5 s after its sender's last
        message -- so on a big picture the first colours visibly disappear
        before the last one is finished. Keep-alives can paper over that, but
        only if every single one gets through: they share the meeting's
        message budget with the drawing itself, and Chime drops the excess
        *silently*. Never going quiet in the first place needs no luck.
        """
        positions: list[tuple[float, int]] = []
        for index, size in enumerate(sizes):
            if size <= 0:
                continue
            for k in range(size):
                positions.append(((k + 0.5) / size, index))
        positions.sort()
        return [index for _, index in positions]

    async def _draw_interleaved(
        self,
        drawers: Sequence[DrawClient],
        layers: Sequence[Layer],
        *,
        stop: asyncio.Event | None = None,
    ) -> None:
        """Send the layers' segments round-robin so they finish together.

        ``stop`` abandons the rest of the drawing when set. What is already on
        screen stays (nothing can erase it) and the caller's ``END`` still
        runs, so an interrupted drawing fades normally rather than hanging
        around for its 5 s idle timer.
        """
        cursors = [0] * len(layers)
        order = self._interleave_order([len(layer.segments) for layer in layers])
        for index in order:
            if stop is not None and stop.is_set():
                drawn = sum(cursors)
                log.info(
                    "drawing interrupted after %d of %d segment(s)",
                    drawn,
                    len(order),
                )
                break
            layer = layers[index]
            await drawers[index].stroke(
                [layer.segments[cursors[index]]], end=False
            )
            cursors[index] += 1
        for layer, drawer in zip(layers, drawers, strict=False):
            log.info(
                "layer hue=%d done (%d segments, %d dropped)",
                layer.hue,
                drawer.sent,
                drawer.dropped,
            )

    async def _draw_sequentially(
        self,
        drawers: Sequence[DrawClient],
        layers: Sequence[Layer],
        *,
        keep_alive_while_drawing: bool,
    ) -> None:
        """One layer at a time, concurrently -- the old behaviour.

        Kept because it is the obvious thing and useful for comparison, but
        the short layers go quiet once finished and depend on keep-alives
        surviving a shared, silently-dropping budget. :meth:`_draw_interleaved`
        is the default for that reason.
        """

        async def draw_one(drawer: DrawClient, layer: Layer) -> None:
            await drawer.stroke(layer.segments, end=False)
            log.info("layer hue=%d done (%d segments)", layer.hue, len(layer))
            if keep_alive_while_drawing:
                drawer.start_keep_alive()

        await asyncio.gather(
            *(draw_one(d, layer) for d, layer in zip(drawers, layers, strict=False))
        )

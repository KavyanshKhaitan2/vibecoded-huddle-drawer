"""Auto-join: sit on the realtime gateway and join huddles we're invited to.

The flow, end to end:

1. Connect to Slack's realtime (flannel) gateway as this account.
2. Wait for a ``huddle_invite`` event. It carries ``channel_id``, ``call_id``,
   ``sender_user_id`` and -- the useful part -- the ``free_willy`` blob, which
   is the Chime meeting plus *our* attendee credentials. So an invite is
   everything needed to join; no ``rooms.join`` round trip is required.
3. Bring up Chime signaling and the audio leg, exactly as a normal join would.
4. Hand the live session to a callback, which can draw, watch, or just sit
   there.

That means the usage is: start this, then invite the account to a huddle from
any Slack client, and it walks in by itself.

Caveats worth knowing before you rely on it:

- The realtime gateway is reverse-engineered from third-party clients; see
  ``docs/huddles-and-chime.md``, which marks several of its sources UNCHECKED.
- Joining is visible to everyone in the huddle, and the account is a real user.
- We do not *have* to answer the invite through ``rooms.inviteResponse`` to
  join the media session, and the accept value for that call has never been
  observed -- only ``"decline"`` has. So accepting is opt-in and best-effort.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Container
from dataclasses import dataclass, field

from .config import Config
from .realtime import HuddleInvite, RealtimeClient
from .session import HuddleSession
from .slack import SlackClient

log = logging.getLogger(__name__)

#: Called with a live session once we are in. When it returns, we leave.
SessionHandler = Callable[[HuddleSession, HuddleInvite], Awaitable[None]]


@dataclass(slots=True)
class AutoJoiner:
    """Joins huddles this account is invited to, one at a time.

    :param handler: what to do once in. Receives the live
        :class:`~huddle_drawings.session.HuddleSession` and the invite that
        caused the join; when it returns, the session is closed and we go back
        to listening.
    :param allow_from: if set, only invites from these user ids are accepted.
        Everything else is ignored (and optionally declined).
    :param allow_channels: if set, only invites for these channel ids are
        accepted.
    :param decline_rejected: answer unwanted invites with
        ``rooms.inviteResponse`` ``"decline"``. That is the one value the doc
        has actually observed, so unlike accepting, this is not guesswork.
    :param accept_invite: also send an *accept* through
        ``rooms.inviteResponse``. Off by default: the accept value is an
        unverified guess, and joining the Chime meeting is what actually puts
        us in the huddle.
    :param once: leave the loop after the first handled huddle.
    """

    config: Config
    handler: SessionHandler
    allow_from: Container[str] | None = None
    allow_channels: Container[str] | None = None
    decline_rejected: bool = False
    accept_invite: bool = False
    once: bool = False
    muted: bool = True
    #: Injection points, for tests and for callers that already hold a client.
    #: Left as ``None``, :meth:`run` builds its own and owns their lifetime.
    slack: SlackClient | None = None
    realtime: RealtimeClient | None = None
    #: How a joined session is produced. Overridden in tests to avoid standing
    #: up a real Chime meeting.
    joiner: Callable[[HuddleInvite, SlackClient], Awaitable[HuddleSession]] | None = None

    _stop: asyncio.Event = field(init=False, default_factory=asyncio.Event)
    _busy: bool = field(init=False, default=False)

    def stop(self) -> None:
        """Ask :meth:`run` to return after the current huddle, if any."""
        self._stop.set()

    def _wanted(self, invite: HuddleInvite) -> bool:
        # An invite with neither a call id nor a Chime blob is unusable: there
        # is nothing to join and nothing to decline. Drop it rather than
        # failing a join attempt on it.
        if not invite.call_id and not invite.has_chime_credentials:
            log.warning(
                "ignoring a huddle_invite with no call_id and no free_willy blob: %r",
                invite.raw,
            )
            return False
        if self.allow_from is not None and invite.sender_user_id not in self.allow_from:
            log.info(
                "ignoring invite to %s: sender %s is not in the allow list",
                invite.call_id,
                invite.sender_user_id,
            )
            return False
        if (
            self.allow_channels is not None
            and invite.channel_id not in self.allow_channels
        ):
            log.info(
                "ignoring invite to %s: channel %s is not in the allow list",
                invite.call_id,
                invite.channel_id,
            )
            return False
        return True

    async def run(self) -> None:
        """Listen for invites and handle them until :meth:`stop` is called."""
        owned_slack = self.slack is None
        slack = self.slack or SlackClient(self.config)
        try:
            await self._run_with(slack)
        finally:
            if owned_slack:
                with contextlib.suppress(Exception):
                    await slack.close()

    async def _run_with(self, slack: SlackClient) -> None:
        identity = await slack.auth_test()
        log.info(
            "auto-join listening as %s (%s)",
            identity.get("user"),
            identity.get("user_id"),
        )

        realtime = self.realtime or await RealtimeClient.from_slack(
            self.config, slack
        )
        invites: asyncio.Queue[HuddleInvite] = asyncio.Queue()

        def queue_invite(event: dict[str, object]) -> None:
            try:
                invite = HuddleInvite.from_event(event)
            except Exception:
                log.warning("could not parse a huddle_invite", exc_info=True)
                return
            invites.put_nowait(invite)

        # Subscribe before connect(): the reader dispatches events as soon as
        # it starts, so a callback registered afterwards can miss one.
        realtime.on_event("huddle_invite", queue_invite)

        await realtime.connect()
        reader = asyncio.create_task(realtime.run_forever(), name="realtime")
        try:
            await self._loop(slack, invites, reader)
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
            await realtime.close()

    async def _loop(
        self,
        slack: SlackClient,
        invites: asyncio.Queue[HuddleInvite],
        reader: asyncio.Task[None],
    ) -> None:
        stop = asyncio.create_task(self._stop.wait(), name="stop")
        try:
            while not self._stop.is_set():
                getter = asyncio.create_task(invites.get(), name="invite")
                done, _ = await asyncio.wait(
                    {getter, stop, reader}, return_when=asyncio.FIRST_COMPLETED
                )
                if reader in done:
                    getter.cancel()
                    # Surface why the gateway died rather than spinning.
                    await reader
                    return
                if stop in done:
                    getter.cancel()
                    return

                invite = getter.result()
                if not self._wanted(invite):
                    # Only decline something we can actually name. "decline" is
                    # the one rooms.inviteResponse value the doc has observed,
                    # so unlike accepting this is not guesswork.
                    if (
                        self.decline_rejected
                        and invite.call_id
                        and invite.channel_id
                    ):
                        with contextlib.suppress(Exception):
                            await slack.rooms_invite_response(
                                channel_id=invite.channel_id,
                                room_id=invite.call_id,
                                response="decline",
                            )
                    continue

                joined = await self._handle(slack, invite)
                # `once` means "one huddle", not "one attempt": a failed join
                # must not count, or a single bad invite would make --once
                # exit having done nothing while the real invite is still
                # coming.
                if joined and self.once:
                    return
        finally:
            stop.cancel()

    async def _handle(self, slack: SlackClient, invite: HuddleInvite) -> bool:
        """Join, run the handler, leave. Returns whether we got in."""
        log.info(
            "invited to call %s in %s by %s — joining",
            invite.call_id,
            invite.channel_id,
            invite.sender_user_id,
        )
        self._busy = True
        try:
            if self.joiner is not None:
                session = await self.joiner(invite, slack)
            else:
                session = await HuddleSession.from_invite(
                    self.config,
                    invite,
                    slack=slack,
                    muted=self.muted,
                    accept=self.accept_invite,
                )
        except Exception:
            log.error("failed to join call %s", invite.call_id, exc_info=True)
            self._busy = False
            return False

        try:
            async with session:
                await self.handler(session, invite)
        except Exception:
            log.error(
                "handler failed while in call %s", invite.call_id, exc_info=True
            )
        finally:
            self._busy = False
            log.info("left call %s", invite.call_id)
        return True


async def wait_for_invite(
    config: Config,
    *,
    timeout: float | None = None,
    from_user: str | None = None,
    slack: SlackClient | None = None,
) -> HuddleInvite:
    """Block until this account is invited to a huddle, then return the invite.

    A one-shot convenience around :class:`RealtimeClient` for scripts that just
    want to say "invite me and I'll start".
    """
    owned = slack is None
    slack = slack or SlackClient(config)
    try:
        realtime = await RealtimeClient.from_slack(config, slack)
        await realtime.connect()
        reader = asyncio.create_task(realtime.run_forever(), name="realtime")
        try:
            def wanted(event: dict[str, object]) -> bool:
                return from_user is None or event.get("sender_user_id") == from_user

            event = await realtime.wait_for_event(
                "huddle_invite", timeout=timeout, predicate=wanted
            )
            return HuddleInvite.from_event(event)
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
            await realtime.close()
    finally:
        if owned:
            await slack.close()

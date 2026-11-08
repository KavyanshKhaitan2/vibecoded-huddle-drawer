"""High-level huddle session: join, get present, draw.

This module sequences the four layers in the order the protocol demands. That
order is not negotiable, and the reason for each step is in
``docs/huddles-and-chime.md``:

1. ``rooms.join`` over the Slack Web API -> the Chime meeting/attendee blob
   (``free_willy``).
2. Open the Chime signaling websocket and ``JOIN``.
3. Bring up the **audio leg**. Signaling alone can send data messages, but
   Slack clients only treat an attendee as present once Chime reports it in
   ``AUDIO_STREAM_ID_INFO``, which only happens once real audio is flowing --
   and clients drop draw messages from attendees they do not know. So an
   attendee without audio can send perfectly well-formed drawings that nobody
   ever sees.
4. Only then draw, with ``itemId`` set to the *sharer's* attendeeId taken from
   an ``INDEX`` frame.

Safety: joining puts a real user into a real call, visibly. Drawing acts on
other people's clients. Use a dedicated account, in a huddle whose
participants are in on it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from .audio import AudioLeg, ice_servers_from_turn_credentials
from .config import Config
from .draw import DrawClient
from .peer_messages import ContentShareSource, build_capabilities
from .ratelimit import TokenBucket
from .realtime import HuddleInvite
from .signaling import ChimeSignaling
from .slack import SlackClient
from .types import (
    AttendeePresence,
    HuddleError,
    HuddleJoin,
    StreamSource,
)

log = logging.getLogger(__name__)

#: How long to wait for Chime to report *us* in AUDIO_STREAM_ID_INFO. Until it
#: does, our drawings are dropped by every Slack client in the huddle.
PRESENCE_TIMEOUT = 20.0

#: How long to give ICE + DTLS before calling the media leg a failure. Chime
#: itself gives up and closes the signaling socket in a second or two, so this
#: mostly exists to produce an accurate error when it does.
MEDIA_TIMEOUT = 15.0


@dataclass(slots=True)
class HuddleSession:
    """A joined huddle: Slack state, Chime signaling and the media leg.

    Build one with :meth:`join`, or use it as an async context manager:

    .. code-block:: python

        async with await HuddleSession.join(config, channel_id="C…") as session:
            share = await session.wait_for_share()
            drawer = session.drawer(share, hue=200)
            await drawer.draw_path(points)
    """

    config: Config
    slack: SlackClient
    join_info: HuddleJoin
    signaling: ChimeSignaling
    audio: AudioLeg
    _closed: bool = False

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    async def join(
        cls,
        config: Config,
        *,
        channel_id: str,
        muted: bool = True,
        receive_stream_ids: tuple[int, ...] = (0,),
        wait_for_presence: bool = True,
        rate_limiter: TokenBucket | None = None,
        slack: SlackClient | None = None,
        resolve: bool = True,
        multidevice: bool | None = None,
    ) -> Self:
        """Join (or start) the huddle in ``channel_id`` and become present.

        ``channel_id`` may be a channel (``C…``), DM (``D…``) or group (``G…``)
        id, or a **user** id (``U…``/``W…``) -- a user is resolved to that
        person's DM channel, so you can start a huddle with someone by id
        alone. Pass ``resolve=False`` to skip that lookup and send whatever you
        gave straight to ``rooms.join``.

        ``multidevice=True`` asks ``rooms.join`` for a **fresh attendee** so
        this session is a separate participant from any other session of the
        same account -- which is what makes several drawing colours possible
        without several accounts. See :meth:`SlackClient.rooms_join`.

        Returns once the session can actually be seen by other clients, unless
        ``wait_for_presence`` is false.
        """
        owned_slack = slack is None
        slack = slack or SlackClient(config)
        try:
            if resolve:
                channel_id = await slack.resolve_conversation_id(channel_id)
            join_info = await slack.rooms_join(
                channel_id,
                regions=config.media_region,
                multidevice=multidevice,
            )
        except BaseException:
            if owned_slack:
                with contextlib.suppress(Exception):
                    await slack.close()
            raise
        return await cls._bring_up(
            config,
            slack,
            join_info,
            muted=muted,
            receive_stream_ids=receive_stream_ids,
            wait_for_presence=wait_for_presence,
            rate_limiter=rate_limiter,
            close_slack_on_error=owned_slack,
        )

    @classmethod
    async def from_invite(
        cls,
        config: Config,
        invite: HuddleInvite,
        *,
        muted: bool = True,
        receive_stream_ids: tuple[int, ...] = (0,),
        wait_for_presence: bool = True,
        rate_limiter: TokenBucket | None = None,
        slack: SlackClient | None = None,
        accept: bool = False,
    ) -> Self:
        """Join a huddle we were *invited* to, straight from the invite.

        A ``huddle_invite`` event on the realtime gateway carries the
        ``free_willy`` blob itself, so no ``rooms.join`` is needed: we already
        have the meeting and our attendee credentials. That makes this the
        fastest path in and the one that works when we do not know, or cannot
        read, the channel the huddle is in.

        If the invite has no usable ``free_willy`` blob, falls back to
        ``rooms.join`` on the invite's channel.

        ``accept`` additionally answers the invite through
        ``rooms.inviteResponse``. It is off by default because the accept value
        for that call is an **unverified guess** -- the doc lists it as an open
        question, having only ever observed ``"decline"``. Joining the Chime
        meeting is what actually puts us in the huddle; accepting is cosmetic
        as far as media is concerned.
        """
        owned_slack = slack is None
        slack = slack or SlackClient(config)
        try:
            if invite.has_chime_credentials:
                # realtime.HuddleInvite.chime_credentials() already strips a
                # null MeetingFeatures and builds the creds without an API
                # call; use it rather than re-deriving them here.
                chime = invite.chime_credentials()
                join_info = HuddleJoin(
                    call_id=invite.call_id,
                    chime=chime,
                    created_by=invite.sender_user_id,
                    raw=invite.raw,
                )
                log.info(
                    "joining invited call %s from the invite's free_willy blob "
                    "(invited by %s)",
                    invite.call_id,
                    invite.sender_user_id,
                )
            else:
                log.info(
                    "invite for call %s carried no free_willy blob; "
                    "falling back to rooms.join on %s",
                    invite.call_id,
                    invite.channel_id,
                )
                join_info = await slack.rooms_join(
                    invite.channel_id, regions=config.media_region
                )

            if accept:
                try:
                    await slack.accept_room_invite(
                        channel_id=invite.channel_id, room_id=invite.call_id
                    )
                except Exception:
                    # The accept value is guesswork; a failure here must not
                    # stop us joining the media session, which is what counts.
                    log.warning(
                        "could not accept the invite through rooms.inviteResponse "
                        "(the accept value is unverified); joining anyway",
                        exc_info=True,
                    )
        except BaseException:
            if owned_slack:
                with contextlib.suppress(Exception):
                    await slack.close()
            raise

        return await cls._bring_up(
            config,
            slack,
            join_info,
            muted=muted,
            receive_stream_ids=receive_stream_ids,
            wait_for_presence=wait_for_presence,
            rate_limiter=rate_limiter,
            close_slack_on_error=owned_slack,
        )

    @classmethod
    async def _bring_up(
        cls,
        config: Config,
        slack: SlackClient,
        join_info: HuddleJoin,
        *,
        muted: bool,
        receive_stream_ids: tuple[int, ...],
        wait_for_presence: bool,
        rate_limiter: TokenBucket | None,
        close_slack_on_error: bool,
    ) -> Self:
        """Bring up signaling + the media leg for an already-obtained join.

        Shared by :meth:`join` and :meth:`from_invite`, because the Chime side
        does not care how we got the credentials.
        """
        signaling: ChimeSignaling | None = None
        audio: AudioLeg | None = None
        try:
            log.info(
                "call %s: attendee %s (meeting %s, region %s)",
                join_info.call_id,
                join_info.chime.attendee_id,
                join_info.chime.meeting_id,
                join_info.chime.media_region,
            )

            signaling = ChimeSignaling(join_info.chime, rate_limiter=rate_limiter)
            await signaling.connect()

            # The audio leg. Two things here are load-bearing, and getting
            # either wrong looks the same from outside -- a 1006 close a
            # couple of seconds in:
            #   * AudioLeg.__init__ pins DTLS-SRTP to
            #     SRTP_AES128_CM_HMAC_SHA1_80. With AES-GCM, Chime's audio
            #     server cannot decrypt us and drops the attendee.
            #   * Chime media only traverses Chime's own TURN relay, whose
            #     credentials arrive in the JOIN_ACK we just received. The
            #     real client sets iceTransportPolicy 'relay' with them, and
            #     AudioLeg does the same once it is given TURN servers.
            ice_servers = ice_servers_from_turn_credentials(
                signaling.turn_credentials
            )
            if not ice_servers:
                log.warning(
                    "no TURN credentials from JOIN_ACK; falling back to aiortc's "
                    "default ICE, which Chime's media server is not expected to "
                    "accept"
                )
            audio = AudioLeg(muted=muted, ice_servers=ice_servers or None)
            offer = await audio.create_offer()
            answer = await signaling.subscribe(
                sdp_offer=offer,
                audio_host=join_info.chime.media_placement.audio_host_url,
                receive_stream_ids=receive_stream_ids,
                audio_muted=muted,
            )
            await audio.accept_answer(answer)

            # Confirm our own media leg before blaming Chime for anything. If
            # ICE or DTLS never completes, Chime closes the *signaling*
            # websocket with 1006 a second or two later, which reads as an
            # SRTP or rate-limit problem and is neither.
            await audio.wait_until_connected(MEDIA_TIMEOUT)

            session = cls(
                config=config,
                slack=slack,
                join_info=join_info,
                signaling=signaling,
                audio=audio,
            )
            if wait_for_presence:
                await session.wait_until_present()
            return session
        except BaseException:
            if audio is not None:
                with contextlib.suppress(Exception):
                    await audio.close()
            if signaling is not None:
                with contextlib.suppress(Exception):
                    await signaling.close()
            if close_slack_on_error:
                with contextlib.suppress(Exception):
                    await slack.close()
            raise

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Leave the huddle and release everything. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for name, coro in (
            ("audio", self.audio.close()),
            ("leave", self.signaling.leave()),
            ("signaling", self.signaling.close()),
            ("slack", self.slack.close()),
        ):
            try:
                await coro
            except Exception:
                log.warning("error while closing %s", name, exc_info=True)

    # -- presence ----------------------------------------------------------

    @property
    def attendee_id(self) -> str:
        """Our own Chime attendeeId.

        One account is always one attendee, however many connections it opens:
        ``rooms.join`` hands the same user the same AttendeeId and JoinToken
        for the life of the room. That means one drawing line in one hue --
        N colours at once needs N accounts.
        """
        return self.join_info.chime.attendee_id

    async def wait_until_present(
        self, timeout: float = PRESENCE_TIMEOUT
    ) -> AttendeePresence:
        """Wait until Chime reports *us* in ``AUDIO_STREAM_ID_INFO``.

        Before this happens, Slack clients do not know we exist and drop our
        draw messages ("Dropping draw message from unknown peer").
        """
        try:
            presence = await self.signaling.wait_for_attendee(
                self.attendee_id, timeout=timeout
            )
        except TimeoutError as exc:
            raise HuddleError(
                f"Chime never reported us ({self.attendee_id}) in AUDIO_STREAM_ID_INFO "
                f"within {timeout:g}s, so our drawings would be dropped as coming from "
                "an unknown peer. The audio leg is probably not actually flowing -- "
                "check the SRTP profile and that Opus packets are being sent."
            ) from exc
        log.info("present in the huddle as %s", presence.attendee_id)
        return presence

    # -- shares ------------------------------------------------------------

    @property
    def shares(self) -> list[StreamSource]:
        """Current screen/content shares, from the latest ``INDEX`` frame."""
        return [s for s in self.signaling.sources.values() if s.is_content_share]

    async def wait_for_share(self, timeout: float | None = None) -> StreamSource:
        """Wait for someone to share their screen.

        A share's attendeeId changes every time the share is restarted, so a
        long-running drawer should re-read :attr:`shares` (or call this again)
        rather than caching the id forever.
        """
        return await self.signaling.wait_for_content_share(timeout=timeout)

    def drawer(
        self,
        share: StreamSource | str,
        *,
        hue: int | None = None,
    ) -> DrawClient:
        """A :class:`DrawClient` aimed at ``share``.

        ``share`` may be a :class:`StreamSource` from ``INDEX`` or a bare
        sharer attendeeId. The renderer drops draws whose ``itemId`` does not
        match the share it is drawing over, so this must be the **sharer's**
        attendeeId -- ``#content`` stripped.
        """
        item_id = share.sharer_attendee_id if isinstance(share, StreamSource) else share
        return DrawClient(
            self.signaling,
            attendee_id=self.attendee_id,
            item_id=item_id,
            hue=hue,
        )

    async def announce_draw_capability(self, enabled: bool = True) -> bool:
        """Broadcast ``capabilities{isDrawEnabled}``.

        The real client sends this when a screenshare starts or stops.
        Receivers just store it per peer; it does not gate our own drawing.
        """
        return await self.signaling.send_data_message(
            "data-channel",
            build_capabilities(self.attendee_id, is_draw_enabled=enabled),
        )

    async def announce_content_share(
        self,
        source_type: ContentShareSource = ContentShareSource.DESKTOP_SCREEN,
    ) -> bool:
        """Broadcast ``contentShareDetails`` for a share *we* are making.

        Only meaningful if this session is itself sharing; drawing on someone
        else's share does not need it.
        """
        from .peer_messages import build_content_share_details

        return await self.signaling.send_data_message(
            "data-channel",
            build_content_share_details(
                self.attendee_id,
                source_type=source_type,
                item_id=self.attendee_id,
            ),
        )


async def draw_on_first_share(
    config: Config,
    channel_id: str,
    paths: list[list[tuple[float, float]]],
    *,
    hue: int | None = None,
    hold: float = 0.0,
) -> None:
    """Join ``channel_id``, wait for a share, draw ``paths``, then leave.

    ``paths`` are lists of ``(x, y)`` points in normalized 0..1 coordinates
    over the shared video. ``hold`` keeps the drawing alive for that many extra
    seconds (the renderer otherwise fades it ~3.7s after the last ``END``).
    """
    async with await HuddleSession.join(config, channel_id=channel_id) as session:
        share = await session.wait_for_share()
        log.info("drawing on %s's share", share.sharer_attendee_id)
        drawer = session.drawer(share, hue=hue)
        for path in paths:
            await drawer.draw_path([(x, y) for x, y in path], end=False)
        if hold > 0:
            drawer.start_keep_alive()
            await asyncio.sleep(hold)
            await drawer.stop_keep_alive()
        await drawer.end_stroke()

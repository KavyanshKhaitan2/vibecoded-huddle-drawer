"""Slack's **realtime (flannel) gateway** -- the websocket the web/desktop app uses.

This is *not* the classic bot RTM flow: there is no ``rtm.connect`` /
``rtm.start`` call and no ``xoxb`` token. We connect straight to
``wss://wss-primary.slack.com/`` with the browser-session identity (the ``xoxc``
token in the query string, the ``d`` cookie in a raw ``Cookie`` header), exactly
as ``docs/huddles-and-chime.md`` describes in "The realtime websocket (RTM-style,
but not ``rtm.connect``)".

Why this module exists: the **``huddle_invite``** event carries the Chime
``free_willy`` blob directly, so when someone invites this account to a huddle we
can build :class:`~huddle_drawings.types.ChimeCredentials` and join the media
session *without* a second API call. Everything else here (room join/leave/update
tracking) is a bonus.

Two URL forms, from two independent clients
-------------------------------------------
``docs/huddles-and-chime.md`` records two different URLs for the same gateway,
seen in two independent reverse-engineered clients. Both are implemented:

* **huddlefm form** (default, ``simple_url=False``) -- the long desktop-client
  URL with ``sync_desync``, ``slack_client=desktop``, ``start_args``,
  ``no_query_on_subscribe``, ``flannel=3``, ``lazy_channels``,
  ``gateway_server``, ``enterprise_id`` and ``batch_presence_aware``. Its
  ``gateway_server`` is the team id with its **first character replaced by
  ``T``**, suffixed ``-1`` (the doc writes it as ``T<team_id minus its first
  letter>-1``; huddlefm's own variable for the team id is misleadingly named
  ``enterpriseId``).
* **super-platinum form** (``simple_url=True``) -- a much shorter URL,
  ``?token=<xoxc>&flannel=3&gateway_server=<team_id>-1``, where
  ``gateway_server`` keeps the **plain** team id. ("Flannel" is the gateway's
  internal name.)

Neither form has been verified by this project; the doc marks its sources as
read-but-unverified, so if one is rejected, try the other.

Protocol (all from the doc):

* the server sends ``{"type":"hello"}`` once it is ready;
* the client sends ``{"type":"ping","id":<ms>}`` every 5 s;
* the server may send ``{"type":"reconnect_url","url":…}``, which is stored and
  used for the next reconnect;
* reconnects use exponential backoff capped at 30 s.

Secrets: the gateway URL *contains* the ``xoxc`` token, so every log line and
every exception message built here goes through :meth:`RealtimeClient._scrub`,
which blanks the configured tokens, any ``token=`` query value and any
``JoinToken``. Event payloads are never logged in full -- a ``huddle_invite``
carries a Chime ``JoinToken``.

Safety (``docs/huddles-and-chime.md``, "Safety"): this connects as a real user.
Do not point it at an account you do not own.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlencode, urlsplit, urlunsplit

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, WebSocketException

from .config import Config
from .types import ChimeCredentials, HuddleError

if TYPE_CHECKING:  # ``slack`` is imported lazily, see ``_slack_helpers``
    from .slack import SlackClient

__all__ = [
    "EVENT_HELLO",
    "EVENT_HUDDLE_INVITE",
    "EVENT_RECONNECT_URL",
    "GATEWAY_BASE_URL",
    "HUDDLE_EVENT_TYPES",
    "MESSAGE_SUBTYPE_HUDDLE_THREAD",
    "ROOM_EVENT_TYPES",
    "START_ARGS",
    "HuddleInvite",
    "RealtimeClient",
    "RealtimeError",
    "RoomEvent",
    "build_gateway_url",
    "extract_room_id",
    "gateway_server_for",
    "reconnect_delay",
]

logger = logging.getLogger(__name__)

#: Base URL of the realtime gateway. Module-level so a test can point the client
#: at a local websockets server.
GATEWAY_BASE_URL: str = "wss://wss-primary.slack.com/"

#: The ``start_args`` value, verbatim from the doc. It is itself a query string
#: beginning with ``?``, so it must be percent-encoded as a *single* parameter
#: value (``urlencode`` does that for us).
START_ARGS: Final = (
    "?agent=client&org_wide_aware=true&eac_cache_ts=true&cache_ts=0"
    "&name_tagging=true&only_self_subteams=true&connect_only=true&ms_latest=true"
)

EVENT_HELLO: Final = "hello"
EVENT_PONG: Final = "pong"
EVENT_RECONNECT_URL: Final = "reconnect_url"
EVENT_HUDDLE_INVITE: Final = "huddle_invite"

#: Wildcard accepted by :meth:`RealtimeClient.on_event`.
WILDCARD: Final = "*"

#: The huddle-relevant event types the doc enumerates. ``message`` is in here
#: because a huddle thread root arrives as a ``message`` with subtype
#: ``huddle_thread`` carrying a ``room`` object.
HUDDLE_EVENT_TYPES: Final = (
    "huddle_invite",
    "sh_room_join",
    "sh_room_leave",
    "sh_room_update",
    "message",
)

#: ``sh_room_*`` event types, which :class:`RoomEvent` models.
ROOM_EVENT_TYPES: Final = ("sh_room_join", "sh_room_leave", "sh_room_update")

MESSAGE_SUBTYPE_HUDDLE_THREAD: Final = "huddle_thread"

#: "client sends ``{"type":"ping","id":<ms>}`` every 5 s".
PING_INTERVAL: float = 5.0

#: How long to wait for ``{"type":"hello"}`` after the socket opens.
HELLO_TIMEOUT: float = 20.0

#: Websocket open timeout, and how long ``close()`` waits for the close
#: handshake. Not from the doc.
OPEN_TIMEOUT: float = 20.0
CLOSE_TIMEOUT: float = 5.0

#: Reconnect backoff: ``BASE * 2 ** attempt``, "capped at 30 s", with
#: decorrelated jitter in ``[0.5, 1.0]`` of that so a fleet of clients does not
#: reconnect in lockstep. The cap is from the doc; the jitter is ours.
RECONNECT_BACKOFF_BASE: float = 1.0
RECONNECT_BACKOFF_MAX: float = 30.0

_TOKEN_IN_URL = re.compile(r"(?i)\btoken=[^&\s\"']+")
_JOIN_TOKEN = re.compile(r"(?i)(\"?JoinToken\"?\s*[:=]\s*\"?)[^\"',}\s]+")

_CallbackResult = None | Awaitable[None]
#: Callbacks receive the raw event dict; build :class:`HuddleInvite` /
#: :class:`RoomEvent` from it with their ``from_event`` classmethods.
EventCallback = Callable[[dict[str, Any]], _CallbackResult]


class RealtimeError(HuddleError):
    """The realtime gateway connection failed, or was used after ``close()``."""


# --------------------------------------------------------------------------- #
# URL construction
# --------------------------------------------------------------------------- #


def gateway_server_for(team_id: str, *, simple: bool = False) -> str:
    """Derive the ``gateway_server`` query parameter from a team id.

    Two mutually exclusive rules, one per source (see the module docstring):

    * default (huddlefm): the team id with its **first character replaced by
      ``T``**, suffixed ``-1`` -- ``E0123ABC`` becomes ``T0123ABC-1``.
    * ``simple=True`` (super-platinum): the **plain** team id suffixed ``-1`` --
      ``E0123ABC`` becomes ``E0123ABC-1``.
    """
    if not team_id:
        raise ValueError("team_id is required to derive gateway_server")
    if simple:
        return f"{team_id}-1"
    if len(team_id) < 2:
        raise ValueError(f"team_id {team_id!r} is too short to derive gateway_server")
    return f"T{team_id[1:]}-1"


def build_gateway_url(
    token: str,
    *,
    team_id: str,
    enterprise_id: str | None = None,
    simple: bool = False,
    base_url: str | None = None,
) -> str:
    """Build the realtime gateway URL.

    ``simple=False`` (default) produces the long huddlefm/desktop URL,
    ``simple=True`` the short super-platinum one. The parameter order matches
    the doc, and every value -- in particular ``start_args``, which is itself a
    query string -- is percent-encoded as one parameter value.

    ``enterprise_id`` is omitted when unknown (the doc only ever shows it
    populated; dropping it rather than sending an empty value is our guess).
    """
    gateway_server = gateway_server_for(team_id, simple=simple)
    if simple:
        params: list[tuple[str, str]] = [
            ("token", token),
            ("flannel", "3"),
            ("gateway_server", gateway_server),
        ]
    else:
        params = [
            ("token", token),
            ("sync_desync", "1"),
            ("slack_client", "desktop"),
            ("start_args", START_ARGS),
            ("no_query_on_subscribe", "1"),
            ("flannel", "3"),
            ("lazy_channels", "1"),
            ("gateway_server", gateway_server),
        ]
        if enterprise_id:
            params.append(("enterprise_id", enterprise_id))
        params.append(("batch_presence_aware", "1"))

    parts = urlsplit(base_url if base_url is not None else GATEWAY_BASE_URL)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path or "/", urlencode(params), "")
    )


def reconnect_delay(attempt: int, *, rng: random.Random | None = None) -> float:
    """Seconds to wait before reconnect attempt ``attempt`` (0-based).

    Exponential, capped at :data:`RECONNECT_BACKOFF_MAX` (the doc's "capped at
    30 s"), then scaled by a random factor in ``[0.5, 1.0]`` so the delay never
    exceeds the cap.
    """
    ceiling = min(RECONNECT_BACKOFF_MAX, RECONNECT_BACKOFF_BASE * 2.0**attempt)
    choose = rng.uniform if rng is not None else random.uniform
    return ceiling * choose(0.5, 1.0)


# --------------------------------------------------------------------------- #
# event models
# --------------------------------------------------------------------------- #


def extract_room_id(event: dict[str, Any]) -> str | None:
    """Pull the call/room id out of an event, whatever shape it arrived in.

    The ``sh_room_*`` payload shape is irregular across the doc's sources, which
    list the id at ``call_id``, ``room.call_id``, ``room.id`` **or**
    ``huddle.id``. All four are tried, in that order; ``None`` means none of
    them was a non-empty string. (super-platinum's fixtures put it at
    ``room.id`` and room ids start with ``R``; huddlefm reads ``call_id``.)
    """
    if not isinstance(event, dict):
        return None
    candidates = (
        event.get("call_id"),
        _path(event, "room", "call_id"),
        _path(event, "room", "id"),
        _path(event, "huddle", "id"),
    )
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    return None


def _path(event: dict[str, Any], *keys: str) -> Any:
    """Follow ``keys`` through nested dicts, returning ``None`` if it breaks."""
    node: Any = event
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _first_str(*values: Any) -> str | None:
    for value in values:
        found = _opt_str(value)
        if found is not None:
            return found
    return None


def _opt_int(value: Any) -> int | None:
    """A positive timestamp, or ``None``.

    ``date_end`` shows up as ``0``/``null`` for a live room, so only a positive
    number counts as an end marker.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value > 0 else None


def _normalise_participants(raw: Any) -> tuple[str, ...]:
    """Participants as user ids, from either shape the doc warns about."""
    from .slack import normalise_participants  # local: see ``_slack_helpers``

    return normalise_participants(raw)


@dataclass(frozen=True, slots=True)
class HuddleInvite:
    """A ``huddle_invite`` event: someone invited this account to a huddle.

    The important part is :attr:`free_willy`, the Chime meeting/attendee blob --
    the same object ``rooms.join`` returns under ``call.free_willy``. Because
    the invite carries it, an invited huddle can be joined with **no** extra API
    call: ``ChimeCredentials.from_free_willy(invite.free_willy)``, or just
    :meth:`chime_credentials`.

    ``free_willy`` holds a Chime ``JoinToken`` and is a secret: it is kept out
    of ``repr`` and must never be logged.
    """

    channel_id: str | None
    call_id: str | None
    sender_user_id: str | None
    free_willy: dict[str, Any] | None = field(default=None, repr=False)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> HuddleInvite:
        """Parse a ``huddle_invite`` event. Missing fields become ``None``.

        Fields are the ones huddlefm uses (``channel_id``, ``call_id``,
        ``sender_user_id``, ``free_willy``); the ``call_id`` fallbacks of
        :func:`extract_room_id` are applied too, since the invite and the
        ``sh_room_*`` events disagree about where the id lives.
        """
        free_willy = event.get("free_willy")
        return cls(
            channel_id=_channel_id_of(event),
            call_id=extract_room_id(event),
            sender_user_id=_first_str(
                event.get("sender_user_id"), event.get("user"), event.get("user_id")
            ),
            free_willy=free_willy if isinstance(free_willy, dict) else None,
            raw=event,
        )

    @property
    def has_chime_credentials(self) -> bool:
        """Whether :meth:`chime_credentials` can succeed without an API call."""
        return isinstance(self.free_willy, dict) and bool(
            self.free_willy.get("meeting") and self.free_willy.get("attendee")
        )

    def chime_credentials(self) -> ChimeCredentials:
        """Build :class:`~huddle_drawings.types.ChimeCredentials` from the invite.

        A null ``meeting.MeetingFeatures`` is stripped exactly as it is for a
        ``rooms.join`` response. Raises :class:`RealtimeError` when the invite
        did not carry a usable blob -- fall back to ``rooms.join`` /
        ``screenhero.rooms.info`` then.
        """
        from .slack import strip_null_meeting_features  # local, see below

        if not isinstance(self.free_willy, dict):
            raise RealtimeError(
                "this huddle_invite carried no 'free_willy' blob; join with "
                "rooms.join instead"
            )
        try:
            return ChimeCredentials.from_free_willy(
                strip_null_meeting_features(self.free_willy)
            )
        except (KeyError, TypeError) as exc:
            raise RealtimeError(
                f"huddle_invite 'free_willy' is missing {exc.args[0]!r}"
            ) from exc


@dataclass(frozen=True, slots=True)
class RoomEvent:
    """An ``sh_room_join`` / ``sh_room_leave`` / ``sh_room_update`` event.

    Also parses a ``message`` with subtype ``huddle_thread``, whose ``room``
    object carries the same fields. The payload shape differs between the
    doc's sources, so every field is best-effort and the untouched event stays
    on :attr:`raw`.
    """

    type: str
    call_id: str | None
    user: str | None
    channel_id: str | None
    participants: tuple[str, ...] = ()
    has_ended: bool = False
    date_end: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> RoomEvent:
        room = event.get("room") if isinstance(event.get("room"), dict) else {}
        huddle = event.get("huddle") if isinstance(event.get("huddle"), dict) else {}
        participants = _normalise_participants(
            room.get("participants")
            if room.get("participants") is not None
            else huddle.get("participants")
        )
        return cls(
            type=_opt_str(event.get("type")) or "",
            call_id=extract_room_id(event),
            user=_first_str(event.get("user"), event.get("user_id")),
            channel_id=_channel_id_of(event),
            participants=participants,
            has_ended=bool(
                room.get("has_ended") or huddle.get("has_ended") or event.get("has_ended")
            ),
            date_end=_opt_int(
                room.get("date_end")
                if room.get("date_end") is not None
                else huddle.get("date_end")
                if huddle.get("date_end") is not None
                else event.get("date_end")
            ),
            raw=event,
        )

    @property
    def ended(self) -> bool:
        """True when either end marker is set.

        The doc: ``sh_room_update`` means the room "ended when either end marker
        is set" (``has_ended`` or ``date_end``).
        """
        return self.has_ended or self.date_end is not None


def _channel_id_of(event: dict[str, Any]) -> str | None:
    """Channel id from any of the places the sources put it."""
    channels = _path(event, "room", "channels")
    first_channel = channels[0] if isinstance(channels, (list, tuple)) and channels else None
    return _first_str(
        event.get("channel_id"),
        event.get("channel"),
        _path(event, "huddle", "channel_id"),
        _path(event, "room", "channel_id"),
        first_channel,
    )


# --------------------------------------------------------------------------- #
# the client
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Subscription:
    types: frozenset[str]
    callback: EventCallback


class RealtimeClient:
    """One connection to Slack's realtime (flannel) gateway.

    Usage::

        rt = await RealtimeClient.from_slack(config, slack)
        rt.on_event("huddle_invite", on_invite)
        await rt.connect()
        task = asyncio.create_task(rt.run_forever())
        ...
        await rt.close()

    :meth:`connect` does one attempt and raises on failure; :meth:`run_forever`
    keeps the connection up, reconnecting with backoff. Subscriptions and
    :meth:`wait_for_event` waiters survive reconnects -- only :meth:`close`
    ends them.
    """

    def __init__(
        self,
        config: Config,
        *,
        team_id: str,
        enterprise_id: str | None = None,
        simple_url: bool = False,
    ) -> None:
        """Wrap ``config``'s identity.

        ``team_id`` is what ``auth.test`` returns (on Enterprise Grid with an
        enterprise-level token that is an ``E…`` id); it is what both
        ``gateway_server`` derivations are built from. ``simple_url`` selects
        the short super-platinum URL instead of the long huddlefm one.
        """
        if not team_id:
            raise ValueError("team_id is required")
        # Fail now rather than at connect time if the id cannot be converted.
        gateway_server_for(team_id, simple=simple_url)

        self._config = config
        self._team_id = team_id
        self._enterprise_id = enterprise_id
        self._simple_url = simple_url

        self._ws: ClientConnection | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._ping_task: asyncio.Task[None] | None = None
        self._subscriptions: list[_Subscription] = []
        self._waiter_futures: set[asyncio.Future[dict[str, Any]]] = set()
        self._reconnect_url: str | None = None
        self._hello_seen = False
        self._closing = False
        #: Set by :meth:`close`, so a reconnect backoff can be cut short.
        self._close_event = asyncio.Event()
        self._ping_id = 0

    # -- plumbing ---------------------------------------------------------- #

    def __repr__(self) -> str:
        state = "connected" if self.is_connected else "disconnected"
        return (
            f"<RealtimeClient {state} team={self._team_id!r} "
            f"simple_url={self._simple_url}>"
        )

    @property
    def config(self) -> Config:
        return self._config

    @property
    def team_id(self) -> str:
        return self._team_id

    @property
    def enterprise_id(self) -> str | None:
        return self._enterprise_id

    @property
    def url(self) -> str:
        """The gateway URL for a fresh connection. **Contains the xoxc token.**"""
        return build_gateway_url(
            self._config.xoxc_token,
            team_id=self._team_id,
            enterprise_id=self._enterprise_id,
            simple=self._simple_url,
        )

    @property
    def safe_url(self) -> str:
        """:attr:`url` with the token blanked -- the only form safe to log."""
        return self._scrub(self.url)

    @property
    def gateway_server(self) -> str:
        """The ``gateway_server`` parameter value this client sends."""
        return gateway_server_for(self._team_id, simple=self._simple_url)

    @property
    def reconnect_url(self) -> str | None:
        """The last ``reconnect_url`` the server sent, if it has not been used."""
        return self._reconnect_url

    @property
    def is_connected(self) -> bool:
        """True while the socket is open, the hello is in and we are not closing."""
        return self._ws is not None and self._hello_seen and not self._closing

    @property
    def is_closed(self) -> bool:
        return self._closing

    async def __aenter__(self) -> RealtimeClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def _scrub(self, text: str) -> str:
        """Blank this identity's secrets, any ``token=`` value and any JoinToken.

        The gateway URL carries the ``xoxc`` token in the query string and a
        ``reconnect_url`` carries one we never configured, so the generic
        ``token=`` pattern matters as much as the configured values.
        """
        for secret in (self._config.xoxc_token, self._config.xoxd_token):
            if secret:
                text = text.replace(secret, "<redacted>")
        text = _TOKEN_IN_URL.sub("token=<redacted>", text)
        return _JOIN_TOKEN.sub(r"\g<1><redacted>", text)

    @classmethod
    async def from_slack(
        cls, config: Config, slack: SlackClient, **kw: Any
    ) -> RealtimeClient:
        """Build a client with ``team_id``/``enterprise_id`` from ``auth.test``.

        The doc says both ``gateway_server`` and ``enterprise_id`` are "derived
        from the ``team_id`` that ``auth.test`` returns (huddlefm's variable is
        misleadingly named ``enterpriseId``)". So: ``enterprise_id`` is taken
        from the response when present, and otherwise -- **inference** -- from
        ``team_id`` itself when that is an ``E…`` (enterprise) id, which is what
        a token minted on an Enterprise Grid org returns and what makes
        huddlefm's naming make sense. Anything in ``kw`` wins.
        """
        auth = await slack.auth_test()
        team_id = _first_str(auth.get("team_id"), auth.get("team"))
        if team_id is None:
            raise RealtimeError(
                "auth.test returned no 'team_id', so gateway_server cannot be derived"
            )
        enterprise_id = _first_str(
            auth.get("enterprise_id"), _path(auth, "enterprise", "id")
        )
        if enterprise_id is None and team_id.startswith("E"):
            enterprise_id = team_id
        kwargs: dict[str, Any] = {"team_id": team_id, "enterprise_id": enterprise_id}
        kwargs.update(kw)
        logger.debug(
            "realtime identity from auth.test: team_id=%s enterprise_id=%s",
            kwargs.get("team_id"),
            kwargs.get("enterprise_id"),
        )
        return cls(config, **kwargs)

    # -- subscriptions ------------------------------------------------------ #

    def on_event(
        self, type_or_types: str | Iterable[str], cb: EventCallback
    ) -> Callable[[], None]:
        """Subscribe ``cb`` to one event type, several, or ``"*"`` for all.

        ``cb`` receives the raw event dict and may be sync or async (an async
        one is awaited in the reader). A callback that raises is logged and the
        reader carries on. Returns a callable that unsubscribes; calling it more
        than once is harmless.
        """
        if isinstance(type_or_types, str):
            types = frozenset({type_or_types})
        else:
            types = frozenset(type_or_types)
        if not types:
            raise ValueError("on_event needs at least one event type")

        subscription = _Subscription(types=types, callback=cb)
        self._subscriptions.append(subscription)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._subscriptions.remove(subscription)

        return unsubscribe

    def on_huddle_invite(
        self, cb: Callable[[HuddleInvite], _CallbackResult]
    ) -> Callable[[], None]:
        """Convenience wrapper: ``cb`` gets a parsed :class:`HuddleInvite`."""

        def adapt(event: dict[str, Any]) -> _CallbackResult:
            return cb(HuddleInvite.from_event(event))

        return self.on_event(EVENT_HUDDLE_INVITE, adapt)

    def on_room_event(
        self, cb: Callable[[RoomEvent], _CallbackResult]
    ) -> Callable[[], None]:
        """Convenience wrapper: ``cb`` gets a parsed :class:`RoomEvent`."""

        def adapt(event: dict[str, Any]) -> _CallbackResult:
            return cb(RoomEvent.from_event(event))

        return self.on_event(ROOM_EVENT_TYPES, adapt)

    async def wait_for_event(
        self,
        type_: str,
        *,
        timeout: float | None = None,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]:
        """Wait for the next event of ``type_`` (``"*"`` for any) and return it.

        ``predicate`` filters candidates -- events it rejects are left for
        other waiters. Raises :class:`TimeoutError` on timeout and
        :class:`RealtimeError` if the client is closed while waiting. Waiting
        survives a reconnect.
        """
        if self._closing:
            raise RealtimeError("the realtime client is closed")
        future, release = self._register_waiter(type_, predicate)
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"timed out after {timeout}s waiting for a {type_!r} event"
            ) from exc
        finally:
            release()

    def _register_waiter(
        self,
        type_: str,
        predicate: Callable[[dict[str, Any]], bool] | None,
    ) -> tuple[asyncio.Future[dict[str, Any]], Callable[[], None]]:
        """Arm a one-shot future for the next matching event.

        Split out of :meth:`wait_for_event` so :meth:`connect` can arm the
        ``hello`` waiter **synchronously, before the reader task starts** --
        otherwise the reader can dispatch the hello before the waiting
        coroutine has had a turn to subscribe, and the handshake hangs.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()

        def receive(event: dict[str, Any]) -> None:
            if future.done():
                return
            try:
                if predicate is not None and not predicate(event):
                    return
            except Exception as exc:  # a bad predicate is the waiter's problem
                future.set_exception(exc)
                return
            future.set_result(event)

        unsubscribe = self.on_event(type_, receive)
        self._waiter_futures.add(future)

        def release() -> None:
            unsubscribe()
            self._waiter_futures.discard(future)

        return future, release

    # -- connect / run ------------------------------------------------------ #

    async def connect(self) -> None:
        """Open the websocket and wait for ``{"type":"hello"}``.

        One attempt; raises :class:`RealtimeError` if the socket cannot be
        opened, closes during the handshake, or never says hello. The hello
        waiter is armed before the reader task starts, so the hello cannot be
        dispatched before anyone is listening.
        """
        if self._closing:
            raise RealtimeError("the realtime client is closed")
        if self._ws is not None:
            raise RealtimeError("already connected to the realtime gateway")

        url, reused = self._next_url()
        logger.info(
            "opening the Slack realtime gateway (%s): %s",
            "reconnect_url" if reused else "simple" if self._simple_url else "desktop",
            self._scrub(url),
        )
        try:
            ws = await connect(
                url,
                # The browser-session identity: xoxc in the query string (part
                # of ``url``) plus the ``d`` cookie in a raw Cookie header.
                additional_headers={"Cookie": self._config.cookie_header},
                user_agent_header=self._config.user_agent,
                # Slack keeps the connection alive with its own ping/pong
                # *events*; RFC6455 pings are not part of the documented
                # protocol, so leave them off.
                ping_interval=None,
                open_timeout=OPEN_TIMEOUT,
                close_timeout=CLOSE_TIMEOUT,
                max_queue=256,
            )
        except (OSError, WebSocketException) as exc:
            # The URL (with the token) can appear in the message.
            raise RealtimeError(
                f"could not open the Slack realtime gateway: "
                f"{self._scrub(f'{type(exc).__name__}: {exc}')}"
            ) from exc

        self._ws = ws
        self._hello_seen = False
        hello, release_hello = self._register_waiter(EVENT_HELLO, None)
        self._reader_task = asyncio.create_task(
            self._read_loop(ws), name="slack-realtime-reader"
        )
        try:
            await self._await_hello(hello)
        finally:
            release_hello()
        self._hello_seen = True
        self._ping_task = asyncio.create_task(
            self._ping_loop(ws), name="slack-realtime-ping"
        )
        logger.info("realtime gateway ready (hello received)")

    def _next_url(self) -> tuple[str, bool]:
        """The URL for this attempt, and whether it came from ``reconnect_url``.

        A stored ``reconnect_url`` is **consumed**: the doc says the client uses
        it "for the next reconnect", and if it is stale we must not retry it
        forever, so the attempt after it falls back to a freshly built URL.
        """
        if self._reconnect_url:
            url, self._reconnect_url = self._reconnect_url, None
            return url, True
        return self.url, False

    async def _await_hello(self, hello: asyncio.Future[dict[str, Any]]) -> None:
        """Wait for the armed ``hello`` future, or for the reader to die trying.

        Waiting on the reader too means a socket that closes during the
        handshake fails fast instead of burning the whole hello timeout.
        """
        reader = self._reader_task
        assert reader is not None
        try:
            await asyncio.wait(
                {hello, reader},
                timeout=HELLO_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except BaseException:  # cancellation of connect() itself
            await self._disconnect()
            raise

        if hello.done() and not hello.cancelled():
            hello.result()  # a hello future never carries an exception
            return

        closed = reader.done()
        await self._disconnect()
        if closed:
            raise RealtimeError(
                "the realtime websocket closed during the hello handshake"
            )
        raise RealtimeError(
            f"the realtime gateway sent no hello within {HELLO_TIMEOUT}s"
        )

    async def run_forever(self) -> None:
        """Keep the connection up: read, ping, and reconnect with backoff.

        Returns when :meth:`close` is called (cancelling the task that runs this
        works too). Connection failures are logged and retried with exponential
        backoff capped at 30 s; the attempt counter resets after a connection
        that got as far as the hello.
        """
        attempt = 0
        while not self._closing:
            try:
                if self._ws is None:
                    await self.connect()
                    attempt = 0
                reader = self._reader_task
                if reader is not None:
                    # ``asyncio.wait`` so a cancelled reader (from close())
                    # does not cancel us.
                    await asyncio.wait({reader})
            except asyncio.CancelledError:
                raise
            except (RealtimeError, TimeoutError, OSError, WebSocketException) as exc:
                logger.warning(
                    "realtime connection problem: %s", self._scrub(str(exc))
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("realtime connection failed unexpectedly")

            await self._disconnect()
            if self._closing:
                break
            delay = reconnect_delay(attempt)
            attempt += 1
            logger.info("reconnecting to the realtime gateway in %.1fs", delay)
            # Waiting on the close event instead of a plain sleep means
            # close() returns promptly even mid-backoff.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._close_event.wait(), delay)
        logger.debug("realtime run_forever finished")

    # -- close -------------------------------------------------------------- #

    async def close(self) -> None:
        """Close the websocket and stop the background tasks. Idempotent.

        Safe before :meth:`connect`, after a failure, and from inside a
        callback (the reader does not cancel itself). Pending
        :meth:`wait_for_event` waiters get a :class:`RealtimeError`.
        """
        already = self._closing
        self._closing = True
        self._close_event.set()
        await self._disconnect()
        self._reconnect_url = None
        error = RealtimeError("the realtime client was closed")
        for future in list(self._waiter_futures):
            if not future.done():
                future.set_exception(error)
        self._waiter_futures.clear()
        if not already:
            logger.info("realtime client closed")

    async def _disconnect(self) -> None:
        """Tear down the current connection, leaving the client reusable."""
        ws, self._ws = self._ws, None
        self._hello_seen = False
        tasks = (self._ping_task, self._reader_task)
        self._ping_task = self._reader_task = None
        for task in tasks:
            await self._cancel(task)
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
            logger.debug("realtime websocket closed")

    @staticmethod
    async def _cancel(task: asyncio.Task[Any] | asyncio.Future[Any] | None) -> None:
        """Cancel and reap ``task``, unless it is the caller itself."""
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    # -- reader / ping ------------------------------------------------------ #

    async def _read_loop(self, ws: ClientConnection) -> None:
        """Read frames until the socket closes; never die on bad input."""
        try:
            async for raw in ws:
                await self._handle_frame(raw)
        except asyncio.CancelledError:
            raise
        except ConnectionClosed:
            logger.info(
                "the realtime websocket closed (code %s%s)",
                ws.close_code,
                f", {ws.close_reason!r}" if ws.close_reason else "",
            )
        except (OSError, WebSocketException) as exc:
            logger.warning("the realtime reader stopped: %s", self._scrub(str(exc)))
        except Exception:  # pragma: no cover - defensive
            logger.exception("the realtime reader crashed")
        else:
            logger.info(
                "the realtime websocket closed (code %s)", getattr(ws, "close_code", None)
            )

    async def _handle_frame(self, raw: str | bytes) -> None:
        """Decode one frame and dispatch it. Bad frames are logged and dropped."""
        if isinstance(raw, (bytes, bytearray, memoryview)):
            # Not something the documented protocol uses; be liberal anyway.
            try:
                text = bytes(raw).decode("utf-8")
            except UnicodeDecodeError:
                logger.warning("ignoring an undecodable binary realtime frame")
                return
        else:
            text = raw

        try:
            event = json.loads(text)
        except (ValueError, TypeError):
            logger.warning(
                "ignoring a malformed realtime frame (%d chars, not JSON)", len(text)
            )
            return
        if not isinstance(event, dict):
            logger.warning(
                "ignoring a realtime frame that is not a JSON object (%s)",
                type(event).__name__,
            )
            return
        await self._dispatch(event)

    async def _dispatch(self, event: dict[str, Any]) -> None:
        """Handle the protocol events, then fan out to subscribers."""
        event_type = event.get("type")
        event_type = event_type if isinstance(event_type, str) else ""

        if event_type == EVENT_HELLO:
            logger.debug("realtime hello")
        elif event_type == EVENT_RECONNECT_URL:
            url = _opt_str(event.get("url"))
            if url:
                self._reconnect_url = url
                logger.debug("stored a reconnect_url: %s", self._scrub(url))
            else:
                logger.debug("ignoring a reconnect_url event with no 'url'")
        elif event_type == EVENT_PONG:
            logger.debug("realtime pong (reply_to=%s)", event.get("reply_to"))
        elif event_type == EVENT_HUDDLE_INVITE:
            # Never log ``free_willy``: it carries a Chime JoinToken.
            invite = HuddleInvite.from_event(event)
            logger.info(
                "huddle_invite: channel=%s call=%s from=%s chime_credentials=%s",
                invite.channel_id,
                invite.call_id,
                invite.sender_user_id,
                "yes" if invite.has_chime_credentials else "no",
            )
        elif event_type in ROOM_EVENT_TYPES:
            if logger.isEnabledFor(logging.DEBUG):
                room = RoomEvent.from_event(event)
                logger.debug(
                    "%s: call=%s user=%s participants=%d ended=%s",
                    event_type,
                    room.call_id,
                    room.user,
                    len(room.participants),
                    room.ended,
                )
        elif not event_type:
            logger.debug("realtime frame with no 'type': keys=%s", sorted(event))
        else:
            # Unknown types are normal: the gateway multiplexes the whole
            # client's event stream.
            logger.debug("realtime event %s", event_type)

        for subscription in list(self._subscriptions):
            if event_type not in subscription.types and WILDCARD not in subscription.types:
                continue
            try:
                result = subscription.callback(event)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "a realtime %s callback raised; continuing",
                    event_type or "<untyped>",
                )

    async def _ping_loop(self, ws: ClientConnection) -> None:
        """Send ``{"type":"ping","id":<ms>}`` every :data:`PING_INTERVAL` seconds."""
        try:
            while True:
                await asyncio.sleep(PING_INTERVAL)
                payload = json.dumps({"type": "ping", "id": self._next_ping_id()})
                await ws.send(payload)
                logger.debug("sent realtime ping %s", self._ping_id)
        except asyncio.CancelledError:
            raise
        except (ConnectionClosed, OSError, WebSocketException) as exc:
            # The reader notices the same close and run_forever reconnects.
            logger.debug("the realtime ping loop stopped: %s", self._scrub(str(exc)))
        except Exception:  # pragma: no cover - defensive
            logger.exception("the realtime ping loop crashed")

    def _next_ping_id(self) -> int:
        """Milliseconds since the epoch, forced to keep increasing."""
        self._ping_id = max(int(time.time() * 1000), self._ping_id + 1)
        return self._ping_id


# ``slack.py`` is imported lazily inside ``_normalise_participants`` and
# ``HuddleInvite.chime_credentials`` on purpose: importing this module must not
# drag in ``httpx`` and the Web API client, so an event-only consumer (and the
# tests) stay independent of that layer.

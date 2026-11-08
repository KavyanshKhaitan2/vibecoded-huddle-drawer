"""Slack Web API over a browser-session identity (``xoxc`` token + ``d`` cookie).

Requests go to ``https://<config.slack_host>/api/<method>`` -- the *workspace*
host, **not** ``slack.com/api`` (docs/huddles-and-chime.md, "Auth model these
projects use") -- with ``token=<xoxc>`` in the form body and the ``d`` cookie in
a raw ``Cookie`` header.

The body shape follows ``hq-fishbowl`` (the oldest, simplest and
maintainer-trusted ``rooms.join`` client): **multipart** form data carrying only
``channel_id``, ``regions`` and ``token``, with no ``_x_*`` fields and no
``multidevice``. Those client-mimicking fields are therefore not required; they
can still be opted into per call with ``reason=`` (which adds ``_x_reason``
alongside ``_x_mode=online`` and ``_x_app_name=client``), because the real
client sends them and the doc documents a specific ``_x_reason`` for some
methods.

Safety (docs/huddles-and-chime.md, "Safety"): ``rooms.join`` **joins a real call
as a real user and is visible to participants**. Do not call it,
``rooms.inviteResponse``, or the realtime websocket against a real account
without a human in the loop, and never with credentials you do not own.

Methods whose request shape no source has actually observed (``rooms.getLink``,
``rooms.sendHuddleInvite``, ``rooms.notifyMember``, ``rooms.request``, and the
*accept* value for ``rooms.inviteResponse``) are implemented as thin wrappers
whose docstrings say so. They are guesses from the doc's "Open questions"
section; do not treat a 200 from one of them as confirmation of anything else.

Nothing here logs or embeds the ``xoxc``/``xoxd`` tokens or a Chime
``JoinToken``: request bodies are never logged, and any response text quoted in
an exception is passed through :func:`_scrub` first.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Config
from .types import ChimeCredentials, HuddleError, HuddleJoin, SlackApiError

logger = logging.getLogger(__name__)

#: Generous enough for a cold ``rooms.join`` (it may *start* the huddle), short
#: enough that a hung socket does not wedge the client forever.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)

#: Total attempts per call: the first try plus two retries.
MAX_ATTEMPTS = 3
#: Backoff is ``RETRY_BACKOFF * 2 ** attempt`` seconds.
RETRY_BACKOFF = 0.5
#: Longest ``Retry-After`` we will honour before giving up instead of sleeping.
MAX_RETRY_AFTER = 10.0

#: Actionable hints for the identity-level failures the doc names.
ERROR_HINTS: dict[str, str] = {
    "team_is_restricted": (
        "this identity may not act on that team. On Enterprise Grid the "
        "enterprise-level xoxc token (the 'E…' team) is required; a "
        "workspace-level ('T…') token gets exactly this error. Re-extract the "
        "token from a client booted on the enterprise org"
    ),
    "not_authed": (
        "the browser session is dead or was never sent. Re-extract XOXC_TOKEN "
        "and XOXD_TOKEN from a logged-in browser session and check "
        "SLACK_HOST points at the right workspace"
    ),
    "invalid_auth": (
        "Slack rejected the xoxc token / d cookie pair. The session is dead "
        "(or the two come from different sessions) -- re-extract both"
    ),
    "token_expired": (
        "the xoxc token has expired. Re-extract XOXC_TOKEN (and XOXD_TOKEN) "
        "from a logged-in browser session"
    ),
}

#: ``ok: false`` codes that ``rooms_leave`` reports instead of raising; the doc
#: records ``feature_not_enabled`` as the normal answer for this identity.
_LEAVE_TOLERATED_ERRORS = frozenset({"feature_not_enabled", "method_not_supported"})

#: **Unverified guesses.** ``rooms.inviteResponse`` is only *observed* with
#: ``response="decline"``; "the accept value for ``rooms.inviteResponse``" is
#: listed verbatim under "Open questions (not answered by these sources)" in
#: docs/huddles-and-chime.md. These are plausible spellings, in the order
#: :meth:`SlackClient.accept_room_invite` tries them -- none is known to work.
ACCEPT_INVITE_VALUES: tuple[str, ...] = ("accept", "accepted", "join", "accept_invite")

#: The known ``response`` value (the only one any source has actually seen).
DECLINE_INVITE_VALUE = "decline"

#: Methods the doc names for pushing an invite out, whose "exact params" it
#: explicitly says were **not seen**. Listed so a caller can pick one for
#: :meth:`SlackClient.rooms_invite`; the order is guesswork, not a fallback
#: chain (we never try a second one, to avoid inviting someone twice).
INVITE_METHOD_CANDIDATES: tuple[str, ...] = (
    "rooms.sendHuddleInvite",
    "rooms.notifyMember",
    "rooms.request",
)

#: Failures that mean retrying a *different* ``response`` value is pointless:
#: the identity or the room is the problem, not the spelling.
_INVITE_FATAL_ERRORS = frozenset(
    {
        "not_authed",
        "invalid_auth",
        "token_expired",
        "team_is_restricted",
        "account_inactive",
        "channel_not_found",
        "room_not_found",
        "call_not_found",
        "ratelimited",
    }
)

_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

#: Sentinel for "key absent", so a present-but-null value is distinguishable.
_MISSING = object()


@dataclass(frozen=True, slots=True)
class HuddleRoomInfo:
    """The bits of a ``screenhero.rooms.info`` response an auto-joiner needs.

    Lives here rather than in ``types.py`` because it is shaped by one Slack
    method rather than shared across layers. Field names follow the doc's
    ``room.*`` keys (docs/huddles-and-chime.md, "``screenhero.rooms.info``").
    """

    room_id: str | None
    #: True when **either** end marker is set: the doc's ``message``/
    #: ``sh_room_update`` rule is "ended when either end marker is set", so
    #: ``room.has_ended`` being false is not enough on its own.
    has_ended: bool
    #: ``room.date_end`` epoch seconds, or ``None`` when unset (Slack uses 0).
    date_end: int | None
    participants: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()
    created_by: str | None = None
    huddle_link: str | None = None
    #: ``room.thread_root_ts`` / ``room.canvas_thread_ts``, whichever is set.
    thread_root_ts: str | None = None
    #: Present only when the response carries a ``free_willy`` blob; the doc
    #: says the ``huddle_invite`` websocket event is where it reliably comes
    #: from, so do not count on it here.
    chime: ChimeCredentials | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_live(self) -> bool:
        """True when the room looks joinable (neither end marker set)."""
        return not self.has_ended


class SlackClient:
    """Minimal async Slack Web API client for the huddle methods we need."""

    def __init__(self, config: Config, *, client: httpx.AsyncClient | None = None) -> None:
        """Wrap ``config``'s identity.

        An injected ``client`` is used as-is (its own timeouts apply) and is
        never closed by :meth:`close`; tests pass one backed by
        ``httpx.MockTransport``.
        """
        self._config = config
        self._emoji_cache: dict[str, str] | None = None
        self._client = client if client is not None else httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
            follow_redirects=False,
        )
        #: Only close what we created.
        self._owns_client = client is None

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> SlackClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying client, unless one was handed in."""
        if self._owns_client:
            await self._client.aclose()

    # -- plumbing ----------------------------------------------------------

    @property
    def config(self) -> Config:
        return self._config

    def _headers(self) -> dict[str, str]:
        headers = {
            "Cookie": self._config.cookie_header,
            "User-Agent": self._config.user_agent,
            "Accept": "application/json",
        }
        headers.update(self._config.extra_headers)
        return headers

    def _scrub(self, text: str) -> str:
        """Replace this identity's secrets with a placeholder."""
        for secret in (self._config.xoxc_token, self._config.xoxd_token):
            if secret:
                text = text.replace(secret, "<redacted>")
        return text

    async def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """POST to ``https://<host>/api/<method>``. Raises SlackApiError on ok:false.

        ``params`` values are stringified (``bool`` becomes ``"true"``/
        ``"false"``); ``None`` values are dropped. ``token`` is added
        automatically. When ``reason`` is given the call also carries the real
        client's ``_x_reason`` / ``_x_mode=online`` / ``_x_app_name=client``
        trio; by default none of them are sent, matching hq-fishbowl.
        """
        fields = self._build_fields(params, reason=reason)
        url = f"{self._config.api_base}/{method}"
        response = await self._post_with_retries(method, url, fields)

        try:
            payload = response.json()
        except ValueError as exc:
            snippet = self._scrub(response.text[:200])
            raise HuddleError(
                f"{method}: expected JSON, got {response.headers.get('content-type')!r} "
                f"(HTTP {response.status_code}): {snippet!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise HuddleError(f"{method}: expected a JSON object, got {type(payload).__name__}")

        if not payload.get("ok", False):
            raise self._api_error(method, payload)

        logger.debug("%s ok", method)
        return payload

    def _build_fields(
        self, params: dict[str, Any] | None, *, reason: str | None
    ) -> dict[str, str]:
        fields: dict[str, str] = {}
        for key, value in (params or {}).items():
            if value is None:
                continue
            if isinstance(value, bool):
                fields[key] = "true" if value else "false"
            else:
                fields[key] = value if isinstance(value, str) else str(value)
        if reason is not None:
            # The real client sends these three together; hq-fishbowl sends
            # none of them, so they stay opt-in.
            fields.setdefault("_x_reason", reason)
            fields.setdefault("_x_mode", "online")
            fields.setdefault("_x_app_name", "client")
        fields["token"] = self._config.xoxc_token
        return fields

    async def _post_with_retries(
        self, method: str, url: str, fields: dict[str, str]
    ) -> httpx.Response:
        """POST the multipart body, retrying transport errors and 5xx/429."""
        # ``(None, value)`` makes httpx emit a plain multipart field (no
        # filename), which is the shape hq-fishbowl sends.
        files = {name: (None, value) for name, value in fields.items()}
        headers = self._headers()
        last_error: Exception = HuddleError(f"{method}: no attempt was made")

        for attempt in range(MAX_ATTEMPTS):
            is_last = attempt + 1 >= MAX_ATTEMPTS
            try:
                response = await self._client.post(url, files=files, headers=headers)
            except httpx.TransportError as exc:
                # Timeouts, connection and read errors. The message may carry
                # the URL but never the request body, so no token can leak.
                last_error = exc
                logger.debug("%s: transport error: %s", method, type(exc).__name__)
                if is_last:
                    break
                await self._sleep_before_retry(method, attempt, None)
                continue

            if response.status_code in _RETRY_STATUSES:
                last_error = HuddleError(f"{method}: HTTP {response.status_code}")
                retry_after = _parse_retry_after(response.headers.get("retry-after"))
                if is_last or (retry_after is not None and retry_after > MAX_RETRY_AFTER):
                    break
                await self._sleep_before_retry(method, attempt, retry_after)
                continue

            if response.status_code >= 400:
                snippet = self._scrub(response.text[:200])
                raise HuddleError(f"{method}: HTTP {response.status_code}: {snippet!r}")
            return response

        raise HuddleError(
            f"{method}: giving up after {MAX_ATTEMPTS} attempt(s) "
            f"({type(last_error).__name__}: {self._scrub(str(last_error))})"
        ) from last_error

    async def _sleep_before_retry(
        self, method: str, attempt: int, retry_after: float | None
    ) -> None:
        delay = retry_after if retry_after is not None else RETRY_BACKOFF * (2**attempt)
        logger.warning(
            "%s: retrying in %.1fs (attempt %d/%d)",
            method,
            delay,
            attempt + 2,
            MAX_ATTEMPTS,
        )
        await asyncio.sleep(delay)

    def _api_error(self, method: str, payload: dict[str, Any]) -> SlackApiError:
        """Build a :class:`SlackApiError`, adding a hint for known failures.

        ``SlackApiError.error`` stays the bare Slack code; only the rendered
        message gains the hint, so callers can still branch on the code.
        """
        code = payload.get("error")
        code = code if isinstance(code, str) else "unknown_error"
        error = SlackApiError(method, code, payload)
        hint = ERROR_HINTS.get(code)
        if hint:
            error.args = (f"{error.args[0]} — {hint}",)
        logger.warning("%s failed: %s", method, code)
        return error

    # -- methods -----------------------------------------------------------

    async def auth_test(self) -> dict[str, Any]:
        """``auth.test`` -- confirms the session and returns ``team_id``/``user_id``."""
        return await self.call("auth.test")

    async def open_dm(self, user_id: str) -> str:
        """Open (or reuse) the DM with ``user_id`` and return its channel id.

        ``rooms.join`` needs a *conversation* id, so starting a huddle with a
        person means resolving their ``U…``/``W…`` id to the ``D…`` DM channel
        first. Slack reuses an existing DM rather than creating duplicates, so
        this is safe to call repeatedly -- and it does not notify the user on
        its own; the subsequent ``rooms.join`` is what rings them.

        Tries ``conversations.open`` and falls back to the legacy ``im.open``,
        because on Enterprise Grid some ``conversations.*`` methods are blocked
        for a browser-session identity.
        """
        last: SlackApiError | None = None
        for method in ("conversations.open", "im.open"):
            try:
                payload = await self.call(
                    method, {"users": user_id, "return_im": "true"}
                )
            except SlackApiError as exc:
                if exc.error in _INVITE_FATAL_ERRORS:
                    raise
                logger.info("%s failed for %s: %s", method, user_id, exc.error)
                last = exc
                continue
            channel = payload.get("channel") or {}
            channel_id = channel.get("id") if isinstance(channel, dict) else None
            if channel_id:
                logger.info("resolved user %s to DM channel %s", user_id, channel_id)
                return str(channel_id)
            last = SlackApiError(method, "no_channel_in_response", payload)
        raise HuddleError(
            f"could not open a DM with {user_id}: "
            f"{last.error if last else 'no method succeeded'}. Pass the DM's "
            "channel id (D…) directly if this identity cannot open DMs."
        ) from last

    async def resolve_conversation_id(self, target: str) -> str:
        """Turn a channel, DM, group or **user** id into a conversation id.

        ``C…``/``D…``/``G…`` are returned unchanged. A user id (``U…``/``W…``)
        is resolved to that person's DM channel via :meth:`open_dm`, so a
        huddle can be started with someone by id alone.

        Anything else is returned unchanged and left for ``rooms.join`` to
        reject, so a new id prefix does not become our problem.
        """
        target = target.strip()
        if not target:
            raise HuddleError("empty channel or user id")
        prefix = target[0].upper()
        if prefix in ("U", "W"):
            return await self.open_dm(target)
        if prefix not in ("C", "D", "G"):
            logger.warning(
                "%r does not look like a channel (C…), DM (D…), group (G…) or "
                "user (U…) id; passing it through unchanged",
                target,
            )
        return target

    async def emoji_list(self, *, include_categories: bool = False) -> dict[str, str]:
        """``emoji.list`` -- every emoji name this workspace can render.

        Returns ``name -> value``, where a value is either an
        ``https://emoji.slack-edge.com/...`` image URL or ``alias:<other>``.
        Use :meth:`resolve_emoji` to follow the aliases.

        Observed on a real workspace: **65,583 entries, 65,118 of them image
        URLs** -- Slack serves the standard set as PNGs too, not just custom
        uploads, so one lookup covers ``:squirrel:`` and ``:my-custom-thing:``
        alike. The response is large (several MB), so callers should cache it;
        :meth:`resolve_emoji` does.
        """
        params: dict[str, Any] = {}
        if include_categories:
            params["include_categories"] = "true"
        payload = await self.call("emoji.list", params)
        emoji = payload.get("emoji")
        if not isinstance(emoji, dict):
            raise HuddleError("emoji.list returned no emoji map")
        return {str(k): str(v) for k, v in emoji.items()}

    async def resolve_emoji(self, name: str, *, max_hops: int = 8) -> str:
        """Resolve an emoji name to its image URL, following ``alias:`` chains.

        ``name`` may be given with or without surrounding colons. The emoji
        map is fetched once and cached on this client, because it is several
        megabytes.

        :raises HuddleError: if the name is unknown, or an alias chain loops or
            runs deeper than ``max_hops``.
        """
        name = name.strip().strip(":")
        if not name:
            raise HuddleError("empty emoji name")
        if self._emoji_cache is None:
            self._emoji_cache = await self.emoji_list()
        table = self._emoji_cache

        seen: list[str] = []
        current = name
        for _ in range(max_hops):
            if current in seen:
                raise HuddleError(
                    f"emoji alias loop: {' -> '.join(seen + [current])}"
                )
            seen.append(current)
            value = table.get(current)
            if value is None:
                raise HuddleError(
                    f"this workspace has no emoji called :{current}:"
                    + (f" (via :{name}:)" if current != name else "")
                )
            if value.startswith("alias:"):
                current = value[len("alias:") :]
                continue
            return value
        raise HuddleError(
            f":{name}: alias chain is deeper than {max_hops} hops: "
            f"{' -> '.join(seen)}"
        )

    async def rooms_join(
        self,
        channel_id: str,
        *,
        regions: str | None = None,
        multidevice: bool | None = None,
    ) -> HuddleJoin:
        """``rooms.join`` -- join (or start) the huddle in ``channel_id``.

        Works for channels and DMs, and starts a huddle if there isn't one.
        ``regions`` is the Chime media region to request and is a free choice
        (``us-east-2`` is live-tested); it defaults to ``config.media_region``.

        **``multidevice`` controls whether you get a fresh attendee.**
        Normally ``rooms.join`` hands the same user the same ``AttendeeId`` for
        the life of the room, so extra sessions are extra connections for *one*
        attendee -- and since Slack keys drawing lines by attendee, one line in
        one hue. With ``multidevice=True`` each call returns a **new, distinct**
        ``AttendeeId``, so one account can be several attendees at once and
        therefore draw several lines in several hues.

        Verified live (2026-11-08): three `multidevice=true` joins for one user
        in one room returned three different attendee ids, while
        ``multidevice`` unset or ``"false"`` returned the same id both times.
        This corrects ``docs/huddles-and-chime.md``, which says the id is
        stable "whatever the params" and lists this flag's meaning as unknown.

        This is visible to everyone in the call -- see the module docstring.
        """
        params: dict[str, Any] = {
            "channel_id": channel_id,
            "regions": regions or self._config.media_region,
        }
        if multidevice is not None:
            params["multidevice"] = "true" if multidevice else "false"
        payload = await self.call("rooms.join", params)
        return parse_rooms_join(payload)

    async def rooms_info(self, room_id: str) -> dict[str, Any]:
        """``screenhero.rooms.info`` -- read a room's state by call id."""
        return await self.call(
            "screenhero.rooms.info",
            {"room": room_id},
            reason="all-calls-store/conditional-fetch",
        )

    async def rooms_leave(self, room_id: str) -> dict[str, Any]:
        """``rooms.leave`` -- best effort; the Chime ``LEAVE`` frame is what counts.

        The doc records ``feature_not_enabled`` as the answer for a
        browser-session identity, so that (and ``method_not_supported``) is
        returned as a payload rather than raised.
        """
        try:
            return await self.call("rooms.leave", {"room": room_id})
        except SlackApiError as exc:
            if exc.error in _LEAVE_TOLERATED_ERRORS:
                logger.info(
                    "rooms.leave not available for this identity (%s); "
                    "rely on the Chime LEAVE frame",
                    exc.error,
                )
                return exc.response
            raise

    async def rooms_invite_response(
        self, *, channel_id: str, room_id: str, response: str = "decline"
    ) -> dict[str, Any]:
        """``rooms.inviteResponse`` -- answer a huddle invite.

        Only ``"decline"`` has been observed; **the accept value is unknown**
        (docs/huddles-and-chime.md, "Open questions"). Use
        :meth:`accept_room_invite` if you want to probe for it.
        """
        return await self.call(
            "rooms.inviteResponse",
            {"response": response, "channel_id": channel_id, "room_id": room_id},
            reason="respond-to-huddle-invite",
        )

    async def accept_room_invite(
        self,
        *,
        channel_id: str,
        room_id: str,
        candidates: Sequence[str] = ACCEPT_INVITE_VALUES,
    ) -> str:
        """Try to *accept* a huddle invite, and return the value that worked.

        .. warning::
           **This is guesswork, not a documented flow.** Every source in
           docs/huddles-and-chime.md only ever sends
           ``rooms.inviteResponse`` with ``response="decline"``; "the accept
           value for ``rooms.inviteResponse``" is listed under that doc's
           **open questions**. :data:`ACCEPT_INVITE_VALUES` is a list of
           plausible spellings and nothing more -- none of them has been seen
           to work, and it is equally possible that accepting is not done
           through this method at all.

           A working alternative that *is* live-verified: just call
           :meth:`rooms_join` on the invite's ``channel_id``. It joins (or
           starts) the huddle without answering the invite, which is what an
           auto-joiner actually needs; accepting is only about the invite's
           UI state.

        Each candidate is sent in turn until one returns ``ok: true``. A
        failure that is about the identity or the room rather than the value
        (see :data:`_INVITE_FATAL_ERRORS`) aborts immediately and is raised
        as-is, so a dead session is not hammered four times.

        :raises SlackApiError: on the first identity/room-level failure.
        :raises HuddleError: when every candidate was rejected; the message
            lists each value with the error code it got.
        """
        if not candidates:
            raise ValueError("accept_room_invite needs at least one candidate value")

        attempts: list[str] = []
        last_error: SlackApiError | None = None
        for value in candidates:
            try:
                payload = await self.rooms_invite_response(
                    channel_id=channel_id, room_id=room_id, response=value
                )
            except SlackApiError as exc:
                if exc.error in _INVITE_FATAL_ERRORS:
                    raise
                attempts.append(f"{value}={exc.error}")
                last_error = exc
                continue
            logger.info("rooms.inviteResponse accepted with response=%r", value)
            # Callers should cache this: the unknown is now known for this
            # workspace, so there is no need to probe again.
            payload.setdefault("_accept_value", value)
            return value

        raise HuddleError(
            "rooms.inviteResponse rejected every candidate accept value "
            f"({', '.join(attempts)}). The accept value is an open question in "
            "docs/huddles-and-chime.md -- call rooms_join(channel_id) instead, "
            "which is live-verified and does not need the invite answered."
        ) from last_error

    async def rooms_send_huddle_invite(
        self, *, channel_id: str, room_id: str, user_ids: Sequence[str] | str
    ) -> dict[str, Any]:
        """``rooms.sendHuddleInvite`` -- invite users into a huddle.

        .. warning::
           **Speculative params.** docs/huddles-and-chime.md names this
           method but says its "exact params" were **not seen**; no source
           here has ever called it. ``channel_id``/``room_id``/``user_ids``
           (comma-joined, the usual Slack convention) are the most plausible
           shape, not a verified one. Expect ``invalid_arguments`` or similar
           and adjust with :meth:`call` once someone observes a real request.
        """
        return await self.call(
            "rooms.sendHuddleInvite",
            {
                "channel_id": channel_id,
                "room_id": room_id,
                "user_ids": _join_ids(user_ids),
            },
            reason="calls-api/rooms.sendHuddleInvite",
        )

    async def rooms_notify_member(
        self, *, channel_id: str, room_id: str, user_ids: Sequence[str] | str
    ) -> dict[str, Any]:
        """``rooms.notifyMember`` -- ring someone about a live huddle.

        .. warning::
           **Speculative params**, exactly as :meth:`rooms_send_huddle_invite`
           -- the doc lists the method name only.
        """
        return await self.call(
            "rooms.notifyMember",
            {
                "channel_id": channel_id,
                "room_id": room_id,
                "user_ids": _join_ids(user_ids),
            },
            reason="calls-api/rooms.notifyMember",
        )

    async def rooms_request(self, *, channel_id: str, room_id: str | None = None) -> dict[str, Any]:
        """``rooms.request`` -- ask to be let into a huddle.

        .. warning::
           **Speculative params** (the doc lists the name only, under "Open
           questions"). Its relationship to ``huddles.knock`` is also unknown;
           both appear in the method catalogue with no observed request.
        """
        return await self.call(
            "rooms.request",
            {"channel_id": channel_id, "room_id": room_id},
            reason="calls-api/rooms.request",
        )

    async def rooms_invite(
        self,
        *,
        channel_id: str,
        room_id: str,
        user_ids: Sequence[str] | str,
        method: str = INVITE_METHOD_CANDIDATES[0],
    ) -> dict[str, Any]:
        """Send a huddle invite via one of the doc's candidate invite methods.

        .. warning::
           There is **no documented ``rooms.invite``**. This is sugar over
           :data:`INVITE_METHOD_CANDIDATES`, every one of which has
           unverified params (see :meth:`rooms_send_huddle_invite`).

        It deliberately calls exactly **one** method -- the candidates are not
        tried in sequence, because a second call that also happens to work
        would invite the same person twice. Pass ``method=`` to pick another.
        """
        if method not in INVITE_METHOD_CANDIDATES:
            raise ValueError(
                f"unknown invite method {method!r}; expected one of "
                f"{', '.join(INVITE_METHOD_CANDIDATES)}"
            )
        if method == "rooms.notifyMember":
            return await self.rooms_notify_member(
                channel_id=channel_id, room_id=room_id, user_ids=user_ids
            )
        if method == "rooms.request":
            return await self.rooms_request(channel_id=channel_id, room_id=room_id)
        return await self.rooms_send_huddle_invite(
            channel_id=channel_id, room_id=room_id, user_ids=user_ids
        )

    async def rooms_get_link(self, room_id: str) -> dict[str, Any]:
        """``rooms.getLink`` -- fetch the shareable link for a room.

        .. warning::
           **Speculative params**: the doc lists this method's "exact params"
           as not seen. ``room`` is used here because that is what the one
           documented room-reading method (``screenhero.rooms.info``) takes.
           ``screenhero.rooms.info`` already returns ``room.huddle_link``, so
           prefer :meth:`huddle_info` -- that path is sourced, this one is not.
        """
        return await self.call(
            "rooms.getLink", {"room": room_id}, reason="calls-api/rooms.getLink"
        )

    async def huddle_info(self, room_id: str) -> HuddleRoomInfo:
        """``screenhero.rooms.info``, parsed into :class:`HuddleRoomInfo`.

        Convenience over :meth:`rooms_info` for an auto-joiner: it pulls out
        the room id, the either-marker ended state, participants, channels,
        creator, huddle link, thread ts and the Chime blob if one is present.
        """
        return parse_room_info(await self.rooms_info(room_id))


def _parse_retry_after(value: str | None) -> float | None:
    """Seconds from a ``Retry-After`` header, if it is a plain number."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None  # HTTP-date form; fall back to our own backoff.
    return max(0.0, seconds)


def normalise_participants(raw: Any) -> tuple[str, ...]:
    """Normalise ``huddle.participants`` to a tuple of user ids.

    The doc warns this is **either** a list of user-id strings **or** a list of
    objects with a ``user_id`` (docs/huddles-and-chime.md, "Web API methods").
    Anything else in the list is skipped rather than crashing the join.
    """
    if not isinstance(raw, (list, tuple)):
        return ()
    ids: list[str] = []
    for item in raw:
        if isinstance(item, str):
            if item:
                ids.append(item)
        elif isinstance(item, dict):
            for key in ("user_id", "user", "id"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    ids.append(value)
                    break
    return tuple(ids)


def strip_null_meeting_features(free_willy: dict[str, Any]) -> dict[str, Any]:
    """Copy ``free_willy`` with a null ``meeting.MeetingFeatures`` removed.

    huddlefm deletes the key before handing the meeting to the Chime SDK --
    the SDK does not accept a null there. The input is never mutated, so the
    untouched response stays available on :attr:`HuddleJoin.raw`.
    """
    meeting = free_willy.get("meeting")
    if not isinstance(meeting, dict):
        return free_willy
    if meeting.get("MeetingFeatures", _MISSING) is not None:
        return free_willy
    cleaned_meeting = {k: v for k, v in meeting.items() if k != "MeetingFeatures"}
    return {**free_willy, "meeting": cleaned_meeting}


def parse_rooms_join(payload: dict[str, Any]) -> HuddleJoin:
    """Build a :class:`HuddleJoin` from a ``rooms.join`` response body."""
    call_obj = payload.get("call")
    if not isinstance(call_obj, dict):
        raise HuddleError("rooms.join response has no 'call' object")
    free_willy = call_obj.get("free_willy")
    if not isinstance(free_willy, dict):
        raise HuddleError(
            "rooms.join response has no 'call.free_willy' Chime blob "
            "(is this workspace on a non-Chime media backend?)"
        )
    try:
        chime = ChimeCredentials.from_free_willy(strip_null_meeting_features(free_willy))
    except KeyError as exc:
        raise HuddleError(
            f"rooms.join: call.free_willy is missing {exc.args[0]!r}"
        ) from exc

    huddle = _sub_dict(payload, "huddle")
    canvas = _sub_dict(payload, "canvas")

    call_id = call_obj.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise HuddleError("rooms.join response has no 'call.call_id'")

    return HuddleJoin(
        call_id=call_id,
        chime=chime,
        huddle_id=_opt_str(huddle.get("id")),
        created_by=_opt_str(huddle.get("created_by")),
        participants=normalise_participants(huddle.get("participants")),
        thread_channel_id=_opt_str(canvas.get("thread_channel_id")),
        # hq-fishbowl reads huddle.thread_root_ts; huddlefm reads the canvas.
        thread_root_ts=_opt_str(canvas.get("root_thread_ts"))
        or _opt_str(huddle.get("thread_root_ts")),
        raw=payload,
    )


def _join_ids(user_ids: Sequence[str] | str) -> str:
    """Comma-join user ids, the usual Slack convention for id lists."""
    if isinstance(user_ids, str):
        return user_ids
    ids = [uid.strip() for uid in user_ids if uid and uid.strip()]
    if not ids:
        raise ValueError("user_ids is empty")
    return ",".join(ids)


def _epoch_or_none(value: Any) -> int | None:
    """Slack sends ``0`` (or omits the key) for "no end time"."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) or None


def _str_tuple(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(item for item in raw if isinstance(item, str) and item)


def parse_room_info(payload: dict[str, Any]) -> HuddleRoomInfo:
    """Build a :class:`HuddleRoomInfo` from a ``screenhero.rooms.info`` body.

    ``has_ended`` is the OR of both end markers: the doc's rule for
    ``sh_room_update`` / the huddle thread root is "ended when either end
    marker is set", so a stale ``has_ended: false`` next to a ``date_end``
    still means the huddle is over.
    """
    room = payload.get("room")
    if not isinstance(room, dict):
        raise HuddleError("screenhero.rooms.info response has no 'room' object")

    date_end = _epoch_or_none(room.get("date_end"))
    chime: ChimeCredentials | None = None
    free_willy = room.get("free_willy")
    if not isinstance(free_willy, dict):
        free_willy = payload.get("free_willy")
    if isinstance(free_willy, dict) and "meeting" in free_willy:
        try:
            chime = ChimeCredentials.from_free_willy(strip_null_meeting_features(free_willy))
        except KeyError as exc:
            # A partial blob is not fatal here -- rooms.join can supply it.
            logger.warning(
                "screenhero.rooms.info: ignoring incomplete free_willy (missing %r)",
                exc.args[0],
            )

    return HuddleRoomInfo(
        room_id=_opt_str(room.get("id")),
        has_ended=bool(room.get("has_ended")) or date_end is not None,
        date_end=date_end,
        participants=normalise_participants(room.get("participants")),
        channels=_str_tuple(room.get("channels")),
        created_by=_opt_str(room.get("created_by")),
        huddle_link=_opt_str(room.get("huddle_link")),
        thread_root_ts=_opt_str(room.get("thread_root_ts"))
        or _opt_str(room.get("canvas_thread_ts")),
        chime=chime,
        raw=payload,
    )


def _sub_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None

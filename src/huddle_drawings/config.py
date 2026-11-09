"""Configuration: credentials and tunables, loaded from the environment/.env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Chime's per-meeting data-message token bucket, measured in
# docs/huddles-peer-messages.md#throughput: ~200 capacity refilling at ~150/s,
# shared by every sender in the meeting. Excess is dropped silently.
CHIME_DATA_MESSAGE_RATE = 150.0
CHIME_DATA_MESSAGE_BURST = 200.0

#: The share of that meeting-wide budget this client will take, leaving the
#: rest for everything else going on in the huddle: other people's drawings and
#: reactions, cursor ``POSITION`` updates, awareness-sync and transcript
#: traffic. The limit is per *meeting*, not per sender, and the excess is
#: dropped **silently** -- so spending all 150/s does not fail loudly, it just
#: starts eating other participants' messages (and theirs ours). 0.9 is the
#: single knob for that headroom; everything that paces itself derives from it.
DATA_MESSAGE_HEADROOM = 0.9

#: What this client actually sends at: 135 payloads/s sustained, 180 burst.
CLIENT_DATA_MESSAGE_RATE = CHIME_DATA_MESSAGE_RATE * DATA_MESSAGE_HEADROOM
CLIENT_DATA_MESSAGE_BURST = CHIME_DATA_MESSAGE_BURST * DATA_MESSAGE_HEADROOM

#: Chime drops payloads over this size.
MAX_DATA_MESSAGE_BYTES = 2048

#: Topics carrying PeerMessage payloads.
TOPIC_DATA_CHANNEL = "data-channel"
TOPIC_AWARENESS = "awareness-sync-channel"
TOPIC_TRANSCRIPT_CATCHUP = "transcript-catchup"

# Renderer timings (docs/huddles-peer-messages.md), in seconds.
FADE_AFTER_END = 0.7
FADE_AFTER_IDLE = 5.0
FADE_DURATION = 3.0
#: How long to wait after an END before an account's line is certainly gone.
#: The doc recommends ~8.5s before reusing an account for a fresh picture.
LINE_CLEAR_WAIT = 8.5


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


@dataclass(slots=True)
class Config:
    """Credentials and endpoints for one Slack browser-session identity."""

    xoxc_token: str
    xoxd_token: str
    #: Workspace host, e.g. ``hackclub.slack.com``. API calls go to
    #: ``https://<host>/api/<method>``, not ``slack.com/api``.
    slack_host: str = "slack.com"
    #: Chime media region requested from ``rooms.join``. ``us-east-2`` is
    #: live-tested; the region is a free choice, not just Slack's default.
    media_region: str = "us-east-2"
    user_agent: str = "insomnia/11.2.0"
    #: Extra headers to merge into every Slack API request.
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.xoxc_token:
            raise ConfigError("XOXC_TOKEN is empty")
        if not self.xoxd_token:
            raise ConfigError("XOXD_TOKEN is empty")
        if not self.xoxc_token.startswith("xoxc-"):
            raise ConfigError(
                f"XOXC_TOKEN should start with 'xoxc-', got {self.xoxc_token[:8]!r}"
            )
        if not self.xoxd_token.startswith("xoxd-"):
            raise ConfigError(
                f"XOXD_TOKEN should start with 'xoxd-', got {self.xoxd_token[:8]!r}"
            )
        self.slack_host = self.slack_host.removeprefix("https://").removeprefix(
            "http://"
        ).rstrip("/")

    @property
    def api_base(self) -> str:
        return f"https://{self.slack_host}/api"

    @property
    def cookie_header(self) -> str:
        return f"d={self.xoxd_token}"

    @classmethod
    def from_env(cls, env_file: str | Path | None = None, **overrides: object) -> Config:
        """Load config from ``.env`` plus the process environment.

        Values already in the environment win over the ``.env`` file; explicit
        ``overrides`` win over both.
        """
        load_dotenv(dotenv_path=env_file, override=False)

        def get(name: str, default: str | None = None) -> str | None:
            raw = os.environ.get(name, default)
            return raw.strip() if isinstance(raw, str) else raw

        xoxc = get("XOXC_TOKEN") or ""
        xoxd = get("XOXD_TOKEN") or ""
        if not xoxc or not xoxd:
            missing = [
                n
                for n, v in (("XOXC_TOKEN", xoxc), ("XOXD_TOKEN", xoxd))
                if not v
            ]
            raise ConfigError(
                f"missing {', '.join(missing)} — copy .env.example to .env and fill it in"
            )

        kwargs: dict[str, object] = {
            "xoxc_token": xoxc,
            "xoxd_token": xoxd,
            "slack_host": get("SLACK_HOST", "slack.com") or "slack.com",
            "media_region": get("CHIME_MEDIA_REGION", "us-east-2") or "us-east-2",
        }
        if ua := get("SLACK_USER_AGENT"):
            kwargs["user_agent"] = ua
        kwargs.update(overrides)
        return cls(**kwargs)  # type: ignore[arg-type]

    @property
    def account_label(self) -> str:
        """A short, non-secret label for logs and progress output."""
        return f"{self.slack_host}/{self.xoxc_token[5:13]}"

    def redacted(self) -> dict[str, str]:
        """Config as a dict safe to log."""

        def mask(tok: str) -> str:
            return f"{tok[:9]}…{tok[-4:]}" if len(tok) > 16 else "…"

        return {
            "xoxc_token": mask(self.xoxc_token),
            "xoxd_token": mask(self.xoxd_token),
            "slack_host": self.slack_host,
            "media_region": self.media_region,
        }


#: Highest ``XOXC_TOKEN_<n>`` suffix scanned when collecting accounts.
MAX_ACCOUNTS = 32


def load_accounts(
    env_file: str | Path | None = None,
    accounts_file: str | Path | None = None,
) -> list[Config]:
    """Load every configured Slack account, in order.

    **Why more than one account is ever needed:** Slack's renderer keys drawing
    lines by *attendee*, and ``rooms.join`` returns the same AttendeeId to the
    same user for the whole life of the room, however many connections that
    user opens. So extra websocket sessions from one account all draw onto one
    line in one hue -- ``N`` colours at once genuinely requires ``N``
    *accounts*. (Confirmed live: repeated joins returned an identical attendee
    id every time.)

    Accounts are collected from, in order:

    1. ``accounts_file`` -- JSON, either a list of
       ``{"xoxc": ..., "xoxd": ..., "slack_host": ...}`` objects or a single
       such object. ``slack_host`` and ``media_region`` are inherited from the
       environment when a entry omits them.
    2. ``XOXC_TOKEN`` / ``XOXD_TOKEN`` -- the primary account.
    3. ``XOXC_TOKEN_2`` / ``XOXD_TOKEN_2``, ``_3``, … up to
       :data:`MAX_ACCOUNTS`. Numbering must be contiguous; a gap stops the
       scan (and is logged), so a typo in one name does not silently drop
       every later account.

    Returns at least one :class:`Config`, or raises :class:`ConfigError`.
    """
    import json

    load_dotenv(dotenv_path=env_file, override=False)

    def env(name: str) -> str | None:
        raw = os.environ.get(name)
        return raw.strip() if isinstance(raw, str) else raw

    base_host = env("SLACK_HOST") or "slack.com"
    base_region = env("CHIME_MEDIA_REGION") or "us-east-2"

    if accounts_file is not None:
        path = Path(accounts_file)
        try:
            payload = json.loads(path.read_text())
        except OSError as exc:
            raise ConfigError(f"cannot read accounts file {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
        entries = payload if isinstance(payload, list) else [payload]
        accounts: list[Config] = []
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                raise ConfigError(f"{path} entry {index} is not an object")
            xoxc = str(entry.get("xoxc") or entry.get("xoxc_token") or "").strip()
            xoxd = str(entry.get("xoxd") or entry.get("xoxd_token") or "").strip()
            if not xoxc or not xoxd:
                raise ConfigError(
                    f"{path} entry {index} needs both 'xoxc' and 'xoxd'"
                )
            accounts.append(
                Config(
                    xoxc_token=xoxc,
                    xoxd_token=xoxd,
                    slack_host=str(entry.get("slack_host") or base_host),
                    media_region=str(entry.get("media_region") or base_region),
                )
            )
        if not accounts:
            raise ConfigError(f"{path} contained no accounts")
        return accounts

    accounts = [Config.from_env(env_file)]
    for n in range(2, MAX_ACCOUNTS + 1):
        xoxc, xoxd = env(f"XOXC_TOKEN_{n}"), env(f"XOXD_TOKEN_{n}")
        if not xoxc and not xoxd:
            # Warn about a half-configured account rather than ignoring it.
            break
        if not xoxc or not xoxd:
            raise ConfigError(
                f"account {n} is half-configured: set both XOXC_TOKEN_{n} and "
                f"XOXD_TOKEN_{n}"
            )
        accounts.append(
            Config(
                xoxc_token=xoxc,
                xoxd_token=xoxd,
                slack_host=env(f"SLACK_HOST_{n}") or base_host,
                media_region=env(f"CHIME_MEDIA_REGION_{n}") or base_region,
            )
        )

    seen: dict[str, int] = {}
    for index, account in enumerate(accounts, start=1):
        key = account.xoxc_token
        if key in seen:
            raise ConfigError(
                f"accounts {seen[key]} and {index} use the same XOXC token. "
                "They would be the same Slack attendee and so share one "
                "drawing line in one hue; each colour needs a distinct account."
            )
        seen[key] = index
    return accounts

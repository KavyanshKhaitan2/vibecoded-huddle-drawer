# huddle-drawings

Draw on Slack huddle screen shares from Python — no browser, no headless
Chromium, no Slack app. Joins a huddle as a real user over plain HTTP, one
websocket and a pure-Python WebRTC stack, then sends the `PeerMessage.draw`
protobufs that every Slack client renders over the shared video.

Implemented from [`huddles-peer-messages.md`](docs/huddles-peer-messages.md)
and [`huddles-and-chime.md`](docs/huddles-and-chime.md), both vendored into
`docs/`. Those pages mark each claim as **[live]** (observed in a real huddle)
or **[source]** (read from Slack's shipped JS only); this code follows that
distinction, and every docstring that encodes a non-obvious protocol rule cites
where it came from.

> **This joins a real call as a real user and is visible to everyone in it.**
> Drawing, and especially `muteRequest` / `kickNotification`, act on other
> people's clients. Use a dedicated account, in a huddle whose participants are
> in on it.

## Setup

```sh
uv sync
cp .env.example .env    # then fill in XOXC_TOKEN, XOXD_TOKEN and SLACK_HOST
uv run huddle-drawings whoami
```

Both tokens come from a logged-in Slack web session — `.env.example` says
where. On Enterprise Grid the **enterprise-level** xoxc token (the one whose
team is an `E…` id) is required; a workspace-level `T…` token gets
`team_is_restricted`.

## The single command

Start this, then invite the account to a huddle from any Slack client. It walks
in by itself, waits for someone to share their screen, and draws:

```sh
uv run huddle-drawings autojoin --then-draw --image logo.png --hue 200 --hold 30
```

No channel id needed: the `huddle_invite` event carries the Chime meeting *and*
this account's attendee credentials, so the invite is everything required to
join.

Other commands:

Anywhere a channel is taken you can pass a channel (`C…`), DM (`D…`) or group
(`G…`) id, **or a user id** (`U…`/`W…`) — a user is resolved to that person's DM
channel, so you can start a huddle with someone by id alone:

```sh
uv run huddle-drawings draw U0123456789 --shape heart --hold 20
```

```sh
uv run huddle-drawings whoami                       # check the session tokens
uv run huddle-drawings join    C0123456789 --seconds 60
uv run huddle-drawings shares  C0123456789 --wait 60   # who's sharing a screen
uv run huddle-drawings watch   C0123456789            # log peers' draw/reacji traffic
uv run huddle-drawings draw    C0123456789 --shape heart --hue 320 --hold 20
uv run huddle-drawings draw    C0123456789 --text "HELLO" --y 0.1
uv run huddle-drawings draw    C0123456789 --image sticker.png --max-segments 400
uv run huddle-drawings draw    C0123456789 --svg-path "M .1 .1 C .9 .1 .9 .9 .1 .9"
```

## Drawing an image

`--image` traces a file into cubic Béziers and draws the result. This is the
technique the doc recommends: a segment is a *full* cubic Bézier, so contours
traced as smooth curves cost a small fraction of the messages straight hatching
would.

```sh
uv run huddle-drawings draw C0123456789 --image logo.png \
    --trace-mode auto --max-segments 400 --x .3 --y .3 --width .4 --height .4
```

Measured segment counts (default settings), so you can judge cost before
sending:

| image | outline | threshold | fill |
|---|---|---|---|
| 128 px filled square | 13 | 5 | 39 |
| 200 px circle | 48 | 23 | 62 |
| ring with a hole (a letter "O") | 83 | 40 | 93 |
| left-to-right gradient | 0 | 5 | 78 |
| blank image | 0 | 0 | 0 |

`auto` tries `threshold` and falls back to edges — that is genuinely all it is,
plus a guard that rejects a threshold result whose ink fraction is under 0.2%
or over 70% (near-blank or near-solid). There is no content classifier.

What traces badly — worth reading before blaming the code:

- **Photographs**, in every mode. Nothing separates subject from background by
  luminance, so you get a few hundred messages of disconnected confetti.
  Logos, line art, icons and high-contrast silhouettes are what this is for.
- **Smooth gradients** have no edges at all: `outline` returns **zero
  segments** by design, and `threshold` draws one meaningless curve along the
  split.
- **Uniform images** (any single shade) return nothing, `--invert` included.
- **`fill` loses anything thinner than a band** (two pen widths): a 2 px line
  fills to nothing, and a thin ring keeps only its top and bottom arcs. Use
  `outline`/`threshold` for line art and `fill` only for solid silhouettes.
- **Corners get rounded.** Catmull-Rom fitting slightly pillows a traced
  square; that is the price of one message per curve instead of per pixel step.
- **`--invert` does nothing in `outline` mode** — a gradient magnitude ignores
  polarity.
- The output is a single-hue line drawing: no shading, no fill colour, no
  stroke-width control. Those are not available at the protocol level.
- `--max-segments` is a hard budget, because one segment is one Chime message
  and the whole meeting shares ~150 of those per second. 600 segments is about
  four seconds of solid sending, during which every other sender is competing
  for the same bucket.
- `--trace-mode outline` (edges), `threshold` (region boundaries), `fill`
  (hairpin bands) and `auto`.

## Drawing in colour

```sh
uv run huddle-drawings draw C0123456789 --image logo.png --colors 3 --hold 20
```

This traces the image into colour layers and draws each one in its own hue,
simultaneously. **One account is enough** — you don't need extra Slack users.

Slack's renderer keeps one line in one hue per *attendee*, so N colours needs N
attendees. The trick is that `rooms.join` with **`multidevice=true`** returns a
*fresh* `AttendeeId` on every call, so one account can be several attendees at
once. `--colors N` joins N times that way.

Note this contradicts `docs/huddles-and-chime.md`, which says the same user
gets the same AttendeeId "whatever the params" and lists what `multidevice`
does as an open question. Both are now answered — see
[Verified live](#verified-live-2026-11-08).

Several accounts are still supported (`XOXC_TOKEN_2`/`XOXD_TOKEN_2`, … or
`--accounts-file accounts.json`) and get cycled across the attendees; that's
the fallback if a workspace ever refuses `multidevice`. `--hues 0,120,240`
overrides the traced hues.

Two things worth knowing before you reach for many colours:

- **More accounts buy colours, not speed.** Chime's ~150 messages/second
  budget is per *meeting*, shared by every sender, so all the accounts draw
  against one shared rate limiter and total time is still
  `segments / 150` seconds.
- **Every layer is a pastel.** The renderer forces `hsla(hue, 100%, 75%)`, so
  a dark navy region and a pale sky-blue region come out the same colour. Hue
  is the only controllable channel.
- Each attendee is a visible participant, so a 4-colour drawing means four
  entries in the huddle (all the same user, if you use one account).

## What the protocol forces on you

These are not design choices in this library; they are what Slack's renderer
and Chime do. Each cost a reverse-engineer real debugging time, so they are
worth knowing before you file a bug:

- **One attendee draws one line in one hue.** `rooms.join` returns the same
  AttendeeId to the same user for the life of the room, however many
  connections it opens. The colour is always `hsla(hue, 100%, 75%)`, so
  everything is a pastel — nothing can be drawn dark, grey or white.
  **N colours at once needs N accounts.**
- **Lines cannot be erased, only waited out.** After `END` a line starts fading
  0.7 s later; with no `END`, 5 s after the last non-`POSITION` message; the
  fade takes 3 s. Nothing removes a line early — and any stroke sent before the
  fade finishes brings the *whole* old line back at full opacity. `--hold` sends
  keep-alives to hold a drawing up; `DrawClient.wait_for_clear()` waits the
  ~8.5 s that reliably avoids the "old frame came back" glitch.
- **The audio leg is mandatory.** Signaling alone can send data messages, but
  Slack clients only learn an attendee exists once Chime reports it in
  `AUDIO_STREAM_ID_INFO`, which only happens once real Opus is flowing. Without
  it you can send perfectly well-formed draws that nobody ever sees.
- **`itemId` must be the sharer's attendeeId**, taken from Chime's `INDEX`
  frame with the `#content` suffix stripped. Any other value and the drawing
  silently does not appear. A share's attendeeId changes every time the share
  restarts.
- **~150 data messages/second per meeting**, burst ~200, shared by all senders,
  excess dropped silently with no feedback. Sustained overload disconnects you
  (close 1006). The client rate-limits itself because nothing else will.
- **DTLS-SRTP must be `SRTP_AES128_CM_HMAC_SHA1_80`.** aiortc offers AES-GCM
  first; Chime's audio server cannot decrypt it, sends nothing back, and drops
  the attendee ~2.5 s after the first packet. `audio.force_srtp_profile()`
  pins it and raises loudly if aiortc's internals change shape, because the
  failure mode is otherwise a mysterious 1006.

## Layout

| Module | What it does |
|---|---|
| `config.py` | `.env` loading, protocol constants and timings |
| `types.py` | shared dataclasses; keeps `JoinToken` out of reprs |
| `slack.py` | Slack Web API over a browser session: `rooms.join`, room info, invites |
| `realtime.py` | Slack's realtime (flannel) gateway: `huddle_invite`, `sh_room_*` |
| `signaling.py` | the raw Chime signaling websocket (`0x05` + `SdkSignalFrame`) |
| `audio.py` | the required WebRTC leg (aiortc), SRTP/codec pinning, PLI |
| `peer_messages.py` | every `PeerMessage` builder and parser |
| `geometry.py` | Catmull-Rom fitting, RDP, hairpin fills, stroke font |
| `shapes.py` | named shapes, an SVG-path subset, text placement |
| `image_trace.py` | image → contours → Béziers, within a segment budget |
| `draw.py` | `DrawClient`: strokes, hue, keep-alives, fade handling |
| `session.py` | `HuddleSession`: join (or join-from-invite) and get present |
| `autojoin.py` | listen for invites and walk in |
| `cli.py` | the command line |

Protobuf schemas are vendored as `.proto` under `proto/`; `make proto`
regenerates the Python modules. `make test` runs the suite.

## Status and honesty

Everything is unit-tested against fakes: local websocket servers speaking the
real `SdkSignalFrame` framing, `httpx.MockTransport` for the Web API, a
DTLS loopback that asserts the SRTP profile actually negotiates, and
round-trips of every `PeerMessage` through the real protobuf.

### Verified live (2026-11-08)

The whole path works against a real huddle: join by user id, become present,
trace an image and draw it on someone's screen share.

```
present in the huddle as ccd8a83c-…
drawing on the share by 814718c2-…
prepared 325 segments (~2.2s of the meeting's message budget)
audio leg closed after 858 frames
```

Five things had to be discovered by doing it, **none of which is in either
vendored doc**. Each produced the same useless symptom — the signaling
websocket closing with 1006 — so they are listed with what actually
distinguishes them:

1. **Chime sends signaling frames prefixed `0x02`, not only `0x05`.** The
   payload is an `SdkSignalFrame` either way, and AWS's own
   `stripFrameTypeRTC()` accepts both with the comment "TODO: change server
   frame type to send 0x05". Dropping them discards `INDEX` and
   `AUDIO_STREAM_ID_INFO`, so the attendee never becomes present and its
   drawings are silently never rendered.
2. **Chime media only traverses Chime's own TURN relay.** The credentials
   arrive in `JOIN_ACK` and the real client pairs them with
   `iceTransportPolicy: 'relay'`. Without them, ICE offers host/STUN
   candidates that Chime will not accept.
3. **The bundle policy must be `max-bundle`.** With aiortc's default,
   audio and video get separate ICE transports, so it makes *two* TURN
   allocations, advertises a different relay candidate in each m-section and
   tears one down — and Chime hangs up in about a second.
4. **aiortc offers only `ECDHE-ECDSA` cipher suites, and Chime's media server
   has an RSA certificate.** We are the DTLS *client*, so the suite must suit
   *Chime's* certificate: with the stock list there is no overlap and Chime
   replies with a fatal `handshake_failure` (alert 40).
5. **aiortc advertises three DTLS fingerprints (sha-256/384/512); Chime
   accepts only one.** With the cipher list fixed, the handshake reached
   certificate verification and died with a fatal `unknown_ca` (alert 48).
   Trimming to the single sha-256 line connected the media leg immediately.

And one finding that answers an open question in the doc rather than fixing a
bug: **`rooms.join` with `multidevice=true` returns a fresh `AttendeeId` on
every call.** Three such joins for one user in one room gave three distinct
attendee ids, while the flag unset or `"false"` gave the same id every time.
Two `multidevice` sessions drawing `hue=0` and `hue=210` rendered as a red line
and a blue line at once — so one account can draw in many colours, which the
doc says is impossible ("N colours at once need N accounts").

Plus one aiortc limitation worked around rather than fixed: under
`max-bundle`, `setRemoteDescription` keys remote candidates by ICE transport
(`iceCandidates[iceTransport] = media`), so with one shared transport the last
m-section wins. Chime correctly puts candidates only on the *primary* section,
leaving the survivor empty and ICE stuck in `checking`. We copy the primary
section's candidates onto the others before handing the answer to aiortc.

Also corrected from the SDK source: a `recvonly` video section must carry a
real stream id, because Chime requires `receiveStreamIds` to line up
positionally with the video sections "with a zero for any video send or
inactive section". A client subscribing to nothing therefore sends an
**inactive** video section, not a `recvonly` one.

### Still unverified

1. `rooms.inviteResponse`'s **accept** value, which has never been observed
   (only `"decline"` has). This is why `--accept-invite` is opt-in and
   best-effort: joining the Chime meeting is what actually puts the account in
   the huddle, so accepting is cosmetic.
2. The speculative `rooms.sendHuddleInvite` / `rooms.request` /
   `rooms.getLink` parameter shapes, which the doc lists as "exact params not
   seen". Each is labelled as guesswork in its docstring.
3. Receiving a share's **video** (`receiveStreamIds` with a real stream id and
   a `recvonly` section). The drawing path does not need it.
4. `muteRequest`, `kickNotification`, `jukeboxRequest`, and the transcript and
   awareness flows — built from the schema, never sent.

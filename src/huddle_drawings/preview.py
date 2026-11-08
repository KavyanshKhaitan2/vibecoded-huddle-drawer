"""Render what a drawing will look like, locally, before sending it.

Every segment is a Chime message against a budget the whole meeting shares,
and a line cannot be erased once drawn -- it only fades. So "look at it first"
is worth a lot here: a trace that comes out hollow, speckled or the wrong
colour costs seconds of the meeting's budget and ~4 s of everyone's attention
to find out.

This reproduces the renderer's fixed properties as closely as they are known
from ``docs/huddles-peer-messages.md``:

* a **7 px round** pen, with round caps and joins; width is not controllable
* colour is always ``hsla(hue, 100%, 75%)`` -- a pastel, never dark, grey or
  white, and only the hue varies
* one line per attendee, so each layer is drawn in exactly one colour

It is *not* pixel-exact: Slack draws over the shared video with its own
anti-aliasing and a drop shadow, and the pen is 7 px of the viewer's
*displayed* video, so the apparent thickness depends on how large the share
appears on their screen. Pass ``width``/``height`` matching roughly how big
the share is displayed to get a representative result.
"""

from __future__ import annotations

import colorsys
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image

from .geometry import Segment, bezier_point

logger = logging.getLogger(__name__)

#: The renderer's pen, in pixels of the displayed video.
PEN_PX = 7

#: Samples per Bézier when flattening it for drawing. The renderer draws true
#: curves; this only needs to be fine enough that the polyline looks smooth.
SAMPLES_PER_SEGMENT = 24

#: A dark neutral, close to Slack's own dark-theme backdrop, so the pastels
#: read the way they will over a typical share.
DEFAULT_BACKGROUND = (40, 40, 44)


class HasHueAndSegments(Protocol):
    """Anything layer-shaped: a hue plus the segments drawn in it."""

    hue: int
    segments: Sequence[Segment]


def pastel_for_hue(hue: int) -> tuple[int, int, int]:
    """The exact colour Slack will draw a line of this hue in.

    ``hsla(hue, 100%, 75%)``. Saturation and lightness are fixed by the
    renderer, so this is the only colour a given hue can ever produce.
    """
    r, g, b = colorsys.hls_to_rgb((hue % 360) / 360.0, 0.75, 1.0)
    return (round(r * 255), round(g * 255), round(b * 255))


def _flatten(segment: Segment) -> list[tuple[float, float]]:
    points = [bezier_point(segment, i / SAMPLES_PER_SEGMENT)
              for i in range(SAMPLES_PER_SEGMENT + 1)]
    return [(p.x, p.y) for p in points]


def render_preview(
    layers: Iterable[HasHueAndSegments],
    path: str | Path | None = None,
    *,
    width: int = 960,
    height: int = 600,
    background: tuple[int, int, int] = DEFAULT_BACKGROUND,
    pen_px: int = PEN_PX,
) -> Image.Image:
    """Draw ``layers`` into an image the way Slack would, and optionally save.

    Coordinates are the normalized 0..1 the wire format uses, mapped onto
    ``width`` x ``height``. Returns the image; writes it to ``path`` when one
    is given.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    radius = pen_px / 2.0
    total = 0

    for layer in layers:
        colour = pastel_for_hue(layer.hue)
        for segment in layer.segments:
            total += 1
            points = [(x * width, y * height) for x, y in _flatten(segment)]
            # A degenerate segment (a=b=c=d) is the renderer's "first point of
            # a stroke" dot, and `line` draws nothing for coincident points --
            # so the caps below are what make it visible, exactly as the round
            # pen does on the real canvas.
            draw.line(points, fill=colour, width=pen_px, joint="curve")
            for x, y in points:
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius], fill=colour
                )

    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
        logger.info(
            "wrote a preview of %d segment(s) to %s (%dx%d, %d px pen)",
            total,
            destination,
            width,
            height,
            pen_px,
        )
    return image


def preview_segments(
    segments: Sequence[Segment],
    hue: int,
    path: str | Path | None = None,
    **kwargs: object,
) -> Image.Image:
    """:func:`render_preview` for a single-colour drawing."""

    class _Layer:
        def __init__(self, hue: int, segments: Sequence[Segment]) -> None:
            self.hue = hue
            self.segments = segments

    return render_preview([_Layer(hue, segments)], path, **kwargs)  # type: ignore[arg-type]

"""Turn an image file into cubic Bézier segments that can be drawn on a share.

What :func:`trace_image` produces: **a single-hue line drawing**. The renderer
gives every attendee one line, 7 px round pen, colour fixed at
``hsla(hue, 100%, 75%)`` -- so there is no shading, no stroke-width variation,
no second colour (N colours needs N *attendees*), and nothing can be drawn dark,
grey or white. A traced image is therefore an outline, a region boundary, or a
hairpin-hatched silhouette, and nothing else.

:func:`trace_image_colors` is the same machinery run once per colour: it
quantises the image into N colour clusters and returns one layer per cluster,
each tagged with the hue to draw it in, for N *separate attendees* to send
in parallel. That buys colour and nothing else -- still 7 px, still a pastel
per layer, and the ~150 messages/s bucket is still shared, so N layers cost N
times the messages rather than going N times faster.

**Why it is this stingy.** Chime delivers about **150 data messages/s per
meeting**, burst ~200, shared by every sender, and excess is dropped *silently*
(``docs/huddles-peer-messages.md``, "Throughput"). One message carries one cubic
Bézier, so a trace's cost is counted in segments and
:meth:`TraceResult.estimated_seconds` is the number that matters: a 2000-segment
trace is ~13 s of solid messages, during which every other sender in the huddle
is fighting you for the same bucket. And because lines **cannot be erased** --
they only fade (0.7 s after ``END``, or 5 s after the last stroke, then a 3 s
fade), and any stroke sent before the fade finishes brings the *whole* old
picture back at full opacity -- a trace that overruns its budget does not just
look worse, it collides with the next one. Hence the two techniques the doc
names under "Ways to get more picture per message", both already implemented in
:mod:`geometry` and both used here:

* **Curves** -- contour -> :func:`geometry.rdp` -> Catmull-Rom, via
  :func:`geometry.smooth_path`. One message buys a whole curve instead of a
  short straight dash.
* **Hairpins** -- :func:`geometry.hairpin_segment` fills a band two pens tall
  per message, half the messages of plain hatching. Used by :func:`trace_fill`.

**What traces well:** logos, line art, icons, text, high-contrast silhouettes,
anything that is already a few solid regions. **What traces badly:** photographs
(no threshold separates a face from a sofa, so you get a few hundred messages of
confetti), soft gradients (no edges at all -- see :func:`trace_edges`), fine
texture and anything whose detail is below one pen width
(:func:`geometry.pen_size_normalized`: 7 px is ~0.0065 of a 1080p share's
height, and nothing finer survives rendering). Catmull-Rom also *rounds sharp
corners*: a traced square comes out as a squarish blob with bulged corners,
because 5 points through a corner is what one message per curve buys.

Everything returned is already normalized 0..1 and fitted into ``rect`` with
:func:`geometry.fit_segments_into_rect`, so segments can go straight to
``DrawClient.stroke``.

Dependencies: Pillow for decoding only (EXIF orientation, RGBA compositing,
downscaling) and numpy for the pixel work. The contour tracer here is a
hand-written crack-following boundary walk -- no OpenCV, no scikit-image.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from . import config, geometry
from .geometry import Point, Rect, Segment

logger = logging.getLogger(__name__)

__all__ = [
    "AUTO_MAX_INK",
    "AUTO_MIN_INK",
    "COLOR_BINS_BITS",
    "DEFAULT_BACKGROUND_BORDER_SHARE",
    "DEFAULT_BACKGROUND_LEVEL",
    "DEFAULT_BLUR",
    "DEFAULT_FILL_ROW_HEIGHT",
    "DEFAULT_GREY_HUE",
    "DEFAULT_HUE_MERGE_DEGREES",
    "DEFAULT_LAYERS",
    "DEFAULT_MAX_DIMENSION",
    "DEFAULT_MAX_SEGMENTS",
    "DEFAULT_FRAME_COVERAGE",
    "DEFAULT_MIN_CHROMA",
    "DEFAULT_MIN_CONTOUR_LENGTH",
    "DEFAULT_MIN_HUE_SEPARATION",
    "DEFAULT_MIN_LAYER_CONTOUR",
    "DEFAULT_MIN_LAYER_FEATURE",
    "DEFAULT_MIN_LAYER_SHARE",
    "DEFAULT_SIMPLIFY",
    "EDGE_MIN_GRADIENT",
    "MAX_CONTOUR_POINTS",
    "MAX_FILL_ROW_HEIGHT",
    "MAX_SIMPLIFY",
    "MIN_LAYER_SEGMENTS",
    "ColorLayer",
    "TraceMode",
    "TraceResult",
    "contours_from_mask",
    "describe_layers",
    "pastel_rgb",
    "preview_ascii",
    "rgb_to_hue",
    "total_estimated_seconds",
    "trace_edges",
    "trace_fill",
    "trace_image",
    "trace_image_colors",
    "trace_threshold",
]

#: Modes :func:`trace_image` understands. ``"edges"`` is an alias for
#: ``"outline"``.
TraceMode: TypeAlias = Literal["outline", "edges", "threshold", "fill", "auto"]

ImageSource: TypeAlias = "str | Path | Image.Image"

#: Default segment budget.
#:
#: One segment is one Chime data message, and the whole *meeting* shares about
#: 150 of those per second -- so this number is a time cost as much as a
#: fidelity one: 4000 segments is roughly 27 s of solid sending, during which
#: every other sender in the huddle is competing for the same bucket.
#:
#: It is set high because the budget is also the simplification dial: the
#: tracer raises its RDP tolerance until the drawing fits, so a tight budget
#: silently rounds off detail (measured: epsilon 0.030 at 200 segments against
#: the 0.004 floor at 800+). Giving it room means the limit becomes the
#: picture's own complexity rather than an arbitrary cap -- most images
#: saturate well below this and cost only what they need.
DEFAULT_MAX_SEGMENTS = 4000

#: Longest side the tracer works on. A 4000 px photo is downscaled to this
#: first: contour cost grows with the *perimeter in pixels*, so tracing at full
#: resolution would produce a million-point contour that then has to be thrown
#: away anyway (the 7 px pen cannot show 4000 px of detail).
DEFAULT_MAX_DIMENSION = 512

#: Gaussian blur radius (in traced pixels) applied before thresholding or
#: differentiating. Kills single-pixel speckle, which would otherwise each
#: become their own little contour.
DEFAULT_BLUR = 1.0

#: Default RDP epsilon, in normalized units of the image's longer side. Same
#: value as :func:`geometry.smooth_path`'s default: roughly one 7 px pen width
#: on a 1080p share, i.e. the point at which dropping detail costs nothing
#: visible and saves a message.
DEFAULT_SIMPLIFY = 0.004

#: Hard ceiling on the epsilon the budget logic will raise to. Past ~5% of the
#: image, shapes stop being recognisable and dropping whole contours is the
#: better trade -- see :func:`trace_image`'s budget rules.
MAX_SIMPLIFY = 0.05

#: Multiplier per epsilon-raising round.
SIMPLIFY_GROWTH = 1.5

#: Floor on the simplification epsilon, in **traced pixels**. Contours run
#: along pixel edges, so anything below about a pixel is rasterization
#: staircase, not shape: simplifying less than this buys a wobbly circle for
#: ten times the messages. The effective epsilon is
#: ``max(simplify, MIN_PIXEL_EPSILON / longer_side)``.
MIN_PIXEL_EPSILON = 1.5

#: Contours shorter than this (perimeter, in normalized units of the longer
#: side) are noise and get dropped before anything else happens.
DEFAULT_MIN_CONTOUR_LENGTH = 0.04

#: Default hairpin row pitch: one 7 px pen on a 1080p share
#: (:func:`geometry.pen_size_normalized`), so the two rows of a hairpin touch
#: and the band reads as solid.
DEFAULT_FILL_ROW_HEIGHT = geometry.pen_size_normalized(1920, 1080)[1]

#: Ceiling on the row pitch the budget logic will coarsen to (in destination
#: units). Beyond this a "fill" is visibly a few stripes.
MAX_FILL_ROW_HEIGHT = 0.08

#: Multiplier per row-pitch coarsening round.
FILL_ROW_GROWTH = 1.5

#: A band is filled where at least this fraction of its pixel rows are ink.
#: Consequence: features thinner than a band (two pen widths) can vanish in
#: ``"fill"`` mode. Fill is for solid areas; use ``"outline"`` for line art.
FILL_COVERAGE = 0.5

#: Minimum gradient magnitude (grey levels per pixel) that counts as an edge,
#: whatever Otsu says. Without a floor, a smooth gradient -- whose magnitude is
#: constant and tiny -- would "edge" everywhere.
EDGE_MIN_GRADIENT = 12.0

#: Contours longer than this are decimated before simplification, to keep RDP
#: from doing quadratic work on a 100k-point outline.
MAX_CONTOUR_POINTS = 4000

#: ``"auto"`` only accepts a threshold trace when the ink fraction is in this
#: band; outside it the image is near-blank or near-solid and its "region
#: boundary" is just the frame.
AUTO_MIN_INK = 0.002
AUTO_MAX_INK = 0.70

#: A contour whose bounding box covers more than this fraction of **both** the
#: width and the height of the traced area, and which is near-rectangular (see
#: :data:`FRAME_FILL_RATIO`), is the image's own border rather than anything in
#: the picture, and is rejected. This is what stops a region that reaches the
#: edge of the image -- any background -- from being drawn as one enormous
#: rounded rectangle around the whole share, which is both the ugliest and the
#: most expensive thing this module can emit. Pass ``frame_coverage=1.0`` to
#: switch the check off: nothing can cover more than the whole area.
DEFAULT_FRAME_COVERAGE = 0.90

#: How much of its own bounding box a near-full-size contour has to enclose
#: before it counts as the frame. A border-hugging rectangle encloses ~1.0 of
#: its box; a circle inscribed in the image encloses pi/4 = 0.79 and is a real
#: feature, so this is what keeps "big" from meaning "frame".
FRAME_FILL_RATIO = 0.90


# --------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TraceResult:
    """The segments a trace produced, plus what it had to throw away.

    ``segments`` are :class:`geometry.Segment` values in normalized 0..1
    coordinates, already fitted into the requested ``rect``, in drawing order.
    One segment is one ``draw`` message.

    ``contours`` is how many source features survived into ``segments``: traced
    contours in ``"outline"``/``"threshold"`` mode, and in ``"fill"`` mode the
    number of hairpin **bands** that got at least one stroke (there are no
    contours in a fill).

    The ``*_dropped`` / ``*_filtered`` / ``truncated`` fields exist so a caller
    can tell an honest trace from a degraded one; see :func:`trace_image` for
    what degradation does, in what order.
    """

    #: One cubic Bézier per ``draw`` message, normalized 0..1, in draw order.
    segments: list[Segment]
    #: Source features represented in ``segments`` (contours, or fill bands).
    contours: int
    #: ``(width, height)`` of the file on disk, before any downscaling.
    source_size: tuple[int, int]
    #: Which tracer actually ran. In ``"auto"`` this is the one auto picked.
    mode: str = "outline"
    #: ``(width, height)`` actually traced, after the downscale.
    traced_size: tuple[int, int] = (0, 0)
    #: The rect the segments were fitted into.
    rect: Rect = field(default_factory=geometry.unit_square)
    #: The budget this trace was held to.
    max_segments: int = 0
    #: Contours the tracer found before any filtering.
    contours_found: int = 0
    #: Contours dropped for being shorter than ``min_contour_length``.
    contours_filtered: int = 0
    #: Contours (or fill runs) dropped because the budget ran out.
    contours_dropped: int = 0
    #: Contours rejected for being the **image frame** rather than a feature
    #: (see ``frame_coverage`` on :func:`trace_image`).
    frames_dropped: int = 0
    #: The RDP epsilon finally used, after any budget-driven raising.
    simplify_epsilon: float = 0.0
    #: The hairpin row pitch finally used, in destination units (fill only).
    fill_row_height: float | None = None
    #: The grey level (0..255) or gradient magnitude used as the threshold.
    threshold: float | None = None
    #: True only in the last-resort case: the budget was too small even for one
    #: simplified contour, so the segment list was sliced. The picture will
    #: look cut off; this is the one case the module cannot make graceful.
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.segments)

    def __bool__(self) -> bool:
        return bool(self.segments)

    def estimated_seconds(self, rate: float = config.CHIME_DATA_MESSAGE_RATE) -> float:
        """Wall-clock time to send these at the meeting's message budget.

        ``len(segments) / rate``, with ``rate`` defaulting to Chime's measured
        ~150 data messages/s *per meeting*
        (:data:`config.CHIME_DATA_MESSAGE_RATE`). This is the number users
        actually care about: 2000 segments is ~13 s of solid messages.

        Be honest about what it ignores:

        * the budget is **shared by every sender** in the meeting, so this is a
          floor, not a promise -- two drawers at full tilt get ~half each, and
          anything over the bucket is dropped *silently*;
        * the ~200-message burst allowance makes short traces slightly faster
          than this says;
        * it does not count the handful of non-segment messages a stroke needs
          (``CHANGE_HUE``, the degenerate first ``NEW``, two ``END``s), nor the
          keep-alives required to stop the line fading after 5 s, which a trace
          longer than ~5 s needs anyway.
        """
        if rate <= 0:
            raise ValueError("rate must be positive")
        return len(self.segments) / float(rate)

    def describe(self) -> str:
        """One line for logs: cost, time, and what was given up to get there."""
        parts = [
            f"{len(self.segments)} segments",
            f"{self.contours} contours",
            f"mode={self.mode}",
            f"{self.estimated_seconds():.1f}s at "
            f"{config.CHIME_DATA_MESSAGE_RATE:.0f}/s",
        ]
        if self.contours_filtered:
            parts.append(f"{self.contours_filtered} filtered as noise")
        if self.frames_dropped:
            parts.append(f"{self.frames_dropped} rejected as the image frame")
        if self.contours_dropped:
            parts.append(f"{self.contours_dropped} dropped for budget")
        if self.simplify_epsilon > 0:
            parts.append(f"epsilon={self.simplify_epsilon:.4f}")
        if self.fill_row_height is not None:
            parts.append(f"row={self.fill_row_height:.4f}")
        if self.truncated:
            parts.append("TRUNCATED")
        return ", ".join(parts)


# --------------------------------------------------------------------------
# loading (Pillow is used for decoding only)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Loaded:
    """A decoded, greyscaled, downscaled, blurred image as a float array."""

    #: Luminance, 0..255, shape ``(height, width)``.
    grey: np.ndarray
    #: ``(width, height)`` of the original file.
    source_size: tuple[int, int]

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` actually traced."""
        h, w = self.grey.shape
        return int(w), int(h)

    @property
    def scale(self) -> float:
        """Pixels per source unit: the longer side, in pixels."""
        h, w = self.grey.shape
        return float(max(w, h))

    @property
    def rect(self) -> Rect:
        """The image in *source* units.

        Pixel coordinates are divided by the **longer** side rather than per
        axis, so this space keeps the image's aspect ratio -- which is what
        makes ``preserve_aspect`` mean anything once
        :func:`geometry.fit_segments_into_rect` gets hold of it. (Normalizing
        each axis by its own length, as the wire format does, would turn every
        image into a square.)
        """
        h, w = self.grey.shape
        m = float(max(w, h))
        return Rect(0.0, 0.0, w / m, h / m)

    @property
    def is_uniform(self) -> bool:
        """True when every pixel is the same shade: nothing to trace."""
        if self.grey.size == 0:
            return True
        return float(self.grey.max() - self.grey.min()) < 1e-6


def _composite_on_white(img: Image.Image) -> Image.Image:
    """Flatten any alpha onto **white**, leaving the mode otherwise alone.

    Compositing onto white matters: a transparent PNG converted straight to
    ``L`` gives transparent pixels luminance 0, i.e. "ink", and the trace comes
    out as the negative of what the user sees on a light background. For the
    colour tracer it matters for the same reason -- transparent would otherwise
    cluster as a big black region.
    """
    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)
    return img


def _flatten_to_grey(img: Image.Image) -> Image.Image:
    """Greyscale, compositing any alpha onto white first."""
    return _composite_on_white(img).convert("L")


def _decode(
    img: Image.Image, *, max_dimension: int, blur: float, mode: str
) -> tuple[Image.Image, tuple[int, int]]:
    """Orient, flatten onto white, downscale and blur, in ``mode`` (``L``/``RGB``).

    Shared by the luminance tracers and :func:`trace_image_colors` so both see
    exactly the same pixels; returns the prepared image and the **original**
    file's ``(width, height)``.
    """
    oriented = ImageOps.exif_transpose(img) or img
    source_size = (int(oriented.width), int(oriented.height))
    flat = _composite_on_white(oriented).convert(mode)
    if max_dimension and max(flat.size) > max_dimension:
        factor = max_dimension / float(max(flat.size))
        target = (
            max(1, round(flat.width * factor)),
            max(1, round(flat.height * factor)),
        )
        logger.debug(
            "downscaling %dx%d -> %dx%d before tracing (max_dimension=%d)",
            *source_size,
            *target,
            max_dimension,
        )
        flat = flat.resize(target, Image.Resampling.LANCZOS)
    if blur and blur > 0 and min(flat.size) > 1:
        flat = flat.filter(ImageFilter.GaussianBlur(radius=float(blur)))
    return flat, source_size


def _prepare(
    img: Image.Image, *, max_dimension: int, blur: float
) -> _Loaded:
    grey, source_size = _decode(
        img, max_dimension=max_dimension, blur=blur, mode="L"
    )
    return _Loaded(np.asarray(grey, dtype=np.float32), source_size)


def _prepare_rgb(
    img: Image.Image, *, max_dimension: int, blur: float
) -> tuple[np.ndarray, _Loaded]:
    """``(h, w, 3)`` float RGB plus the matching :class:`_Loaded` luminance.

    The luminance is derived from the *same* downscaled, blurred RGB, so the
    two agree pixel for pixel and every shape/scale/placement helper in this
    module works unchanged on a colour trace.
    """
    rgb, source_size = _decode(
        img, max_dimension=max_dimension, blur=blur, mode="RGB"
    )
    array = np.asarray(rgb, dtype=np.float32)
    grey = np.asarray(rgb.convert("L"), dtype=np.float32)
    return array, _Loaded(grey, source_size)


def _load(
    source: ImageSource,
    *,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    blur: float = DEFAULT_BLUR,
) -> _Loaded:
    if isinstance(source, Image.Image):
        return _prepare(source, max_dimension=max_dimension, blur=blur)
    with Image.open(Path(source)) as img:
        return _prepare(img, max_dimension=max_dimension, blur=blur)


def _load_rgb(
    source: ImageSource,
    *,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    blur: float = DEFAULT_BLUR,
) -> tuple[np.ndarray, _Loaded]:
    if isinstance(source, Image.Image):
        return _prepare_rgb(source, max_dimension=max_dimension, blur=blur)
    with Image.open(Path(source)) as img:
        return _prepare_rgb(img, max_dimension=max_dimension, blur=blur)


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------


def _otsu(values: np.ndarray, *, bins: int = 256) -> float:
    """Otsu's threshold over ``values``' own range.

    Returns the single value when the data is constant, so a uniform image
    thresholds to an empty mask rather than to "everything".
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.0
    lo, hi = float(flat.min()), float(flat.max())
    if hi - lo < 1e-9:
        return lo
    hist, edges = np.histogram(flat, bins=bins, range=(lo, hi))
    centres = (edges[:-1] + edges[1:]) / 2.0
    p = hist / float(flat.size)
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = np.where(denom > 0, (mu_total * omega - mu) ** 2 / denom, -1.0)
    return float(centres[int(np.argmax(between))])


def _resolve_threshold(threshold: float | None) -> float | None:
    """Turn a user ``threshold`` (0..1) into a 0..255 level, or ``None``."""
    if threshold is None:
        return None
    value = float(threshold)
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"threshold must be a fraction in 0..1, got {threshold!r}"
        )
    return value * 255.0


def _threshold_mask(
    loaded: _Loaded, threshold: float | None, invert: bool
) -> tuple[np.ndarray, float]:
    """Ink mask from a luminance threshold: dark is ink unless ``invert``.

    A uniform image returns an empty mask whatever ``invert`` says -- there is
    no region in it, and "invert" of nothing is still nothing.
    """
    grey = loaded.grey
    if loaded.is_uniform:
        logger.debug("uniform image: nothing to threshold")
        return np.zeros(grey.shape, dtype=bool), float(
            grey.reshape(-1)[0] if grey.size else 0.0
        )
    level = _resolve_threshold(threshold)
    if level is None:
        level = _otsu(grey)
    mask = grey < level
    if invert:
        mask = ~mask
    return mask, float(level)


def _convolve3(a: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """3x3 convolution with edge padding, in pure numpy."""
    padded = np.pad(a, 1, mode="edge")
    h, w = a.shape
    out = np.zeros_like(a, dtype=np.float32)
    for i in range(3):
        for j in range(3):
            k = float(kernel[i, j])
            if k:
                out += k * padded[i : i + h, j : j + w]
    return out


def _sobel_magnitude(grey: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude, scaled to **grey levels per pixel**.

    The kernels are divided by 8, so a black-to-white step edge gives ~255 and
    a ramp of one level per pixel gives ~1. That makes
    :data:`EDGE_MIN_GRADIENT` a meaningful absolute floor -- at the cost of
    being resolution-dependent, since downscaling an image steepens its
    gradients.
    """
    kx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]) / 8.0
    gx = _convolve3(grey, kx)
    gy = _convolve3(grey, kx.T)
    return np.hypot(gx, gy)


def _edge_mask(
    loaded: _Loaded, threshold: float | None
) -> tuple[np.ndarray, float]:
    """Ink mask from gradient magnitude: the Canny-ish half of Canny.

    No non-maximum suppression and no hysteresis -- just magnitude over a
    threshold, because what follows traces the *boundary* of the mask anyway.
    Note the consequence: an edge band has two sides, so edge mode traces both
    and costs roughly twice what ``"threshold"`` does for the same picture.
    """
    if loaded.grey.size == 0 or loaded.is_uniform:
        return np.zeros(loaded.grey.shape, dtype=bool), 0.0
    magnitude = _sobel_magnitude(loaded.grey)
    level = _resolve_threshold(threshold)
    if level is None:
        level = max(_otsu(magnitude), EDGE_MIN_GRADIENT)
    mask = magnitude >= level
    return mask, float(level)


# --------------------------------------------------------------------------
# contour tracing (hand-written; no OpenCV, no scikit-image)
# --------------------------------------------------------------------------

#: Directed boundary edges of one ink pixel, walked clockwise in screen
#: coordinates (y down). Each entry is ``(neighbour offset, start corner, end
#: corner)``: the edge is emitted when that neighbour is background.
_PIXEL_EDGES: tuple[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], ...] = (
    ((-1, 0), (0, 0), (1, 0)),  # north neighbour empty -> top edge, left to right
    ((0, 1), (1, 0), (1, 1)),  # east  -> right edge, top to bottom
    ((1, 0), (1, 1), (0, 1)),  # south -> bottom edge, right to left
    ((0, -1), (0, 1), (0, 0)),  # west  -> left edge, bottom to top
)


def contours_from_mask(mask: np.ndarray) -> list[list[Point]]:
    """Trace every boundary of a boolean mask as a closed loop of points.

    This is "crack following": every edge between an ink pixel and a background
    pixel is emitted as a *directed* segment along pixel corners, oriented with
    ink on the inside, and those edges are then stitched into closed loops.
    Compared with Moore-neighbour tracing it needs no visited bookkeeping and
    gets holes right for free -- the outside of a region and the inside of its
    holes are both just loops -- which is what makes a letter "O" trace as two
    circles rather than one.

    The loops are rectilinear (they run along pixel edges), so they look like
    staircases until :func:`geometry.rdp` flattens them; an epsilon of about a
    pixel removes the staircase and nothing else.

    Coordinates are normalized by the mask's **longer** side -- ``x`` spans
    ``0..width/longer``, ``y`` spans ``0..height/longer`` -- so the result keeps
    the mask's aspect ratio and is ready for
    :func:`geometry.fit_segments_into_rect` with the matching ``source`` rect.
    Points are pixel *corners*, so an all-ink 4x4 mask traces as the square
    ``(0,0)-(1,1)``, not ``(0,0)-(0.75,0.75)``.

    Each loop is returned open (the first point is not repeated at the end);
    feed it to :func:`geometry.smooth_path` with ``closed=True``.
    """
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {m.shape!r}")
    m = m.astype(bool, copy=False)
    if m.size == 0 or not m.any():
        return []

    height, width = m.shape
    scale = float(max(width, height))
    padded = np.pad(m, 1)
    inner = padded[1:-1, 1:-1]

    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for (dr, dc), (sx, sy), (ex, ey) in _PIXEL_EDGES:
        # Shifted view of the padded mask = the neighbour of every pixel.
        neighbour = padded[1 + dr : 1 + dr + height, 1 + dc : 1 + dc + width]
        rows, cols = np.nonzero(inner & ~neighbour)
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            edges.setdefault((c + sx, r + sy), []).append((c + ex, r + ey))

    loops = _stitch_loops(edges)
    logger.debug(
        "traced %d contour(s) from a %dx%d mask (%d boundary edges)",
        len(loops),
        width,
        height,
        sum(len(v) for v in edges.values()),
    )
    return [[Point(x / scale, y / scale) for x, y in loop] for loop in loops]


def _stitch_loops(
    edges: dict[tuple[int, int], list[tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    """Walk the directed boundary edges into closed loops.

    Every corner has as many outgoing as incoming edges, so a walk started
    anywhere can only run out of edges back at its start. Where a corner has
    two choices (the diagonal "checkerboard" case) the walk prefers to continue
    straight, which keeps staircases intact instead of splitting them into
    1-pixel loops; either choice yields valid closed loops.
    """
    loops: list[list[tuple[int, int]]] = []
    while edges:
        start = next(iter(edges))
        loop = [start]
        current = start
        heading: tuple[int, int] | None = None
        while True:
            outgoing = edges.get(current)
            if not outgoing:
                break  # defensive: unbalanced edges should be impossible
            index = 0
            if heading is not None and len(outgoing) > 1:
                for i, candidate in enumerate(outgoing):
                    step = (candidate[0] - current[0], candidate[1] - current[1])
                    if step == heading:
                        index = i
                        break
            nxt = outgoing.pop(index)
            if not outgoing:
                del edges[current]
            heading = (nxt[0] - current[0], nxt[1] - current[1])
            current = nxt
            if current == start:
                break
            loop.append(current)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _contour_length(points: Sequence[Point]) -> float:
    """Closed perimeter of a contour, in the contour's own units."""
    if len(points) < 2:
        return 0.0
    total = sum(
        a.distance_to(b) for a, b in zip(points, points[1:], strict=False)
    )
    return total + points[-1].distance_to(points[0])


def _decimate(points: list[Point], limit: int = MAX_CONTOUR_POINTS) -> list[Point]:
    """Thin an absurdly long contour before RDP sees it.

    RDP is cheap on well-behaved input and quadratic on pathological input; a
    photo's edge mask produces plenty of the latter. Keeping every ``n``-th
    point loses nothing the pen could show.
    """
    if len(points) <= limit:
        return points
    step = math.ceil(len(points) / limit)
    thinned = points[::step]
    if thinned[-1] != points[-1]:
        thinned.append(points[-1])
    logger.debug("decimated a %d-point contour to %d", len(points), len(thinned))
    return thinned


# --------------------------------------------------------------------------
# the segment budget
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Fit:
    segments: list[Segment]
    kept: int
    dropped: int
    epsilon: float
    truncated: bool


def _simplify_to_count(
    points: Sequence[Point], max_points: int, start_epsilon: float
) -> list[Point]:
    """RDP with the epsilon raised until at most ``max_points`` survive."""
    epsilon = max(start_epsilon, 1e-6)
    simplified = geometry.rdp(points, epsilon)
    while len(simplified) > max_points and epsilon < 10.0:
        epsilon *= 2.0
        simplified = geometry.rdp(points, epsilon)
    return simplified


def _fit_contours_to_budget(
    contours: Sequence[Sequence[Point]], *, budget: int, epsilon: float
) -> _Fit:
    """Simplify and curve-fit contours so the total segment count fits.

    Degradation order, which is the whole point of this module:

    1. **Raise the epsilon.** Every contour gets coarser together, so the
       picture stays whole and just gets smoother. Up to :data:`MAX_SIMPLIFY`.
    2. **Drop whole contours, least significant first** (shortest perimeter),
       which is why contours are ranked by length rather than by position --
       dropping by position would give you the top-left corner of a good
       picture instead of a rough version of all of it.
    3. **Last resort:** if the budget is too small for even the longest contour
       (budgets in the single digits), that contour is simplified down to the
       budget's worth of points and the list is sliced. This is the only path
       that sets ``truncated``.
    """
    if budget <= 0:
        if contours:
            logger.warning(
                "max_segments=%d: dropping all %d contours", budget, len(contours)
            )
        return _Fit([], 0, len(contours), epsilon, False)
    ranked = sorted(contours, key=_contour_length, reverse=True)
    if not ranked:
        return _Fit([], 0, 0, epsilon, False)

    eps = max(float(epsilon), 0.0)
    while True:
        fitted = [
            geometry.smooth_path(c, epsilon=eps, closed=True) for c in ranked
        ]
        total = sum(len(f) for f in fitted)
        if total <= budget or eps >= MAX_SIMPLIFY:
            break
        eps = min(max(eps, 1e-4) * SIMPLIFY_GROWTH, MAX_SIMPLIFY)

    if total <= budget:
        return _Fit(
            [s for f in fitted for s in f], len(ranked), 0, eps, False
        )

    logger.info(
        "trace over budget at epsilon %.4f (%d > %d segments): dropping the "
        "least significant contours",
        eps,
        total,
        budget,
    )
    out: list[Segment] = []
    kept = 0
    for f in fitted:
        if len(out) + len(f) > budget:
            break
        out.extend(f)
        kept += 1

    if out:
        dropped = len(ranked) - kept
        logger.warning(
            "trace dropped %d of %d contours to fit %d segments "
            "(kept the %d longest, epsilon %.4f)",
            dropped,
            len(ranked),
            budget,
            kept,
            eps,
        )
        return _Fit(out, kept, dropped, eps, False)

    # Not even one contour fits: spend the whole budget on the longest one.
    points = _simplify_to_count(ranked[0], max(budget, 2), eps)
    out = geometry.catmull_rom_to_beziers(points, closed=True)
    truncated = len(out) > budget
    if truncated:
        out = out[:budget]
    logger.warning(
        "budget of %d segment(s) is below one contour; drew the longest "
        "contour with %d point(s)%s",
        budget,
        len(points),
        " and sliced the result (the picture is cut off)" if truncated else "",
    )
    return _Fit(out, 1 if out else 0, len(ranked) - 1, eps, truncated)


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------


def _union(a: Rect, b: Rect) -> Rect:
    a, b = a.normalized(), b.normalized()
    return Rect(
        min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1)
    )


def _destination(rect: Rect | None) -> Rect:
    return (rect or geometry.unit_square()).normalized()


def _vertical_scale(loaded: _Loaded, rect: Rect, preserve_aspect: bool) -> float:
    """Destination units per source unit, vertically.

    Needed before generating hairpins: the row pitch has to come out at about
    one pen on screen *after* the fit, so it is divided by this first.
    """
    src, dst = loaded.rect, rect.normalized()
    sx = dst.width / src.width if src.width else 0.0
    sy = dst.height / src.height if src.height else 0.0
    if preserve_aspect:
        scale = min(sx, sy) if (sx and sy) else (sx or sy)
        return float(scale or 1.0)
    return float(sy or 1.0)


def _source_rect(loaded: _Loaded, segments: Sequence[Segment]) -> Rect:
    """The image's rect **unioned with the segments' control points**.

    The union is not cosmetic: a Catmull-Rom control point near the image
    border, and a hairpin's 4/3 overshoot, both sit slightly outside the image,
    and without the union they would land outside ``rect`` (and, for the
    default unit square, outside the visible 0..1 area). Unioning costs a
    sliver of scale and keeps the guarantee that every coordinate returned is
    inside ``rect``.
    """
    if not segments:
        return loaded.rect
    return _union(
        loaded.rect, Rect.bounding([p for s in segments for p in s.points])
    )


def _place(
    segments: Sequence[Segment],
    loaded: _Loaded,
    rect: Rect,
    preserve_aspect: bool,
    *,
    source: Rect | None = None,
) -> list[Segment]:
    """Fit segments into ``rect`` via :func:`geometry.fit_segments_into_rect`.

    ``source`` defaults to :func:`_source_rect` for these segments alone. Pass
    one explicitly to place several traces with the **same** transform --
    :func:`trace_image_colors` has to, or its layers would each be scaled to
    their own bounding box and the colours would not register with each other.
    """
    if not segments:
        return []
    return geometry.fit_segments_into_rect(
        segments,
        rect,
        preserve_aspect=preserve_aspect,
        source=source if source is not None else _source_rect(loaded, segments),
    )


# --------------------------------------------------------------------------
# the tracers
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Traced:
    """Contour segments in **source** units -- not yet fitted into a rect."""

    segments: list[Segment]
    contours: int
    contours_found: int
    contours_filtered: int
    contours_dropped: int
    frames_dropped: int
    epsilon: float
    truncated: bool
    #: Perimeter of the biggest non-frame contour, in source units. A layer
    #: whose biggest feature is tiny is speckle -- see
    #: :func:`trace_image_colors`'s ``min_layer_feature``.
    longest_contour: float = 0.0


def _polygon_area(points: Sequence[Point]) -> float:
    """Absolute area enclosed by a closed polygon (shoelace formula)."""
    if len(points) < 3:
        return 0.0
    total = 0.0
    previous = points[-1]
    for point in points:
        total += previous.x * point.y - point.x * previous.y
        previous = point
    return abs(total) / 2.0


def _is_frame(
    points: Sequence[Point], area: Rect, coverage: float
) -> bool:
    """Whether a contour is the image's own border rather than a feature.

    Any mask that reaches the edge of the image -- which is what a background
    *is* -- has the image border as one of its boundaries, and that boundary
    traces as a box around the whole share. It is never what the user asked
    for, and on a busy background it is also the most expensive contour in the
    picture, so it is rejected: near-full size on **both** axes
    (:data:`DEFAULT_FRAME_COVERAGE`) *and* near-rectangular
    (:data:`FRAME_FILL_RATIO`), which spares a big circle or blob that happens
    to fill the frame.
    """
    if len(points) < 3:
        return False
    box = Rect.bounding(points)
    if box.width <= coverage * area.width or box.height <= coverage * area.height:
        return False
    box_area = box.width * box.height
    if box_area <= 0.0:
        return False
    return _polygon_area(points) >= FRAME_FILL_RATIO * box_area


@dataclass(frozen=True, slots=True)
class _Contours:
    """A mask's contours after noise and frame rejection, before any budget."""

    contours: list[list[Point]]
    found: int
    filtered: int
    frames: int
    longest: float


def _prepare_contours(
    loaded: _Loaded,
    mask: np.ndarray,
    *,
    min_contour_length: float,
    frame_coverage: float = DEFAULT_FRAME_COVERAGE,
) -> _Contours:
    """Trace a mask's contours and throw away the frame and the noise.

    Budget-independent, so :func:`trace_image_colors` can do this once per
    colour layer, decide which layers are worth an attendee at all, and only
    then divide the segment budget between the survivors.
    """
    raw = contours_from_mask(mask)
    features: list[list[Point]] = []
    frames = 0
    for contour in raw:
        if _is_frame(contour, loaded.rect, frame_coverage):
            frames += 1
            continue
        features.append(contour)
    if frames:
        logger.info(
            "rejected %d contour(s) as the image frame: a region that reaches "
            "the edge of the image would otherwise be drawn as a box around "
            "the whole share",
            frames,
        )
    longest = max((_contour_length(c) for c in features), default=0.0)
    kept = [
        _decimate(c)
        for c in features
        if _contour_length(c) >= min_contour_length
    ]
    filtered = len(features) - len(kept)
    if filtered:
        logger.debug(
            "dropped %d of %d contours shorter than %.4f",
            filtered,
            len(raw),
            min_contour_length,
        )
    return _Contours(
        contours=kept,
        found=len(raw),
        filtered=filtered,
        frames=frames,
        longest=longest,
    )


def _fit_prepared(
    loaded: _Loaded,
    prepared: _Contours,
    *,
    max_segments: int,
    simplify: float,
) -> _Traced:
    """Simplify and curve-fit prepared contours into a segment budget."""
    floor = MIN_PIXEL_EPSILON / loaded.scale
    epsilon = max(simplify, floor)
    if epsilon > simplify:
        logger.debug(
            "raising epsilon %.5f to the %.1f-pixel floor %.5f", simplify,
            MIN_PIXEL_EPSILON, floor,
        )
    fit = _fit_contours_to_budget(
        prepared.contours, budget=max_segments, epsilon=epsilon
    )
    return _Traced(
        segments=fit.segments,
        contours=fit.kept,
        contours_found=prepared.found,
        contours_filtered=prepared.filtered,
        contours_dropped=fit.dropped,
        frames_dropped=prepared.frames,
        epsilon=fit.epsilon,
        truncated=fit.truncated,
        longest_contour=prepared.longest,
    )


def _contour_segments(
    loaded: _Loaded,
    mask: np.ndarray,
    *,
    max_segments: int,
    min_contour_length: float,
    simplify: float,
    frame_coverage: float = DEFAULT_FRAME_COVERAGE,
) -> _Traced:
    """Mask -> contours -> filtering -> budget, stopping short of placement.

    Split out of :func:`_trace_contours` so :func:`trace_image_colors` can
    trace every colour layer and only then work out the one shared transform
    that fits all of them into the destination rect.
    """
    return _fit_prepared(
        loaded,
        _prepare_contours(
            loaded,
            mask,
            min_contour_length=min_contour_length,
            frame_coverage=frame_coverage,
        ),
        max_segments=max_segments,
        simplify=simplify,
    )


def _trace_contours(
    loaded: _Loaded,
    mask: np.ndarray,
    *,
    mode: str,
    rect: Rect,
    max_segments: int,
    min_contour_length: float,
    simplify: float,
    preserve_aspect: bool,
    threshold: float | None,
    frame_coverage: float = DEFAULT_FRAME_COVERAGE,
) -> TraceResult:
    """Shared tail of the contour-based modes."""
    traced = _contour_segments(
        loaded,
        mask,
        max_segments=max_segments,
        min_contour_length=min_contour_length,
        simplify=simplify,
        frame_coverage=frame_coverage,
    )
    result = TraceResult(
        segments=_place(traced.segments, loaded, rect, preserve_aspect),
        contours=traced.contours,
        source_size=loaded.source_size,
        mode=mode,
        traced_size=loaded.size,
        rect=rect,
        max_segments=max_segments,
        contours_found=traced.contours_found,
        contours_filtered=traced.contours_filtered,
        contours_dropped=traced.contours_dropped,
        frames_dropped=traced.frames_dropped,
        simplify_epsilon=traced.epsilon,
        threshold=threshold,
        truncated=traced.truncated,
    )
    logger.info("traced image: %s", result.describe())
    return result


def trace_threshold(
    path: ImageSource,
    rect: Rect | None = None,
    *,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    threshold: float | None = None,
    blur: float = DEFAULT_BLUR,
    min_contour_length: float = DEFAULT_MIN_CONTOUR_LENGTH,
    invert: bool = False,
    simplify: float = DEFAULT_SIMPLIFY,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    preserve_aspect: bool = True,
    frame_coverage: float = DEFAULT_FRAME_COVERAGE,
) -> TraceResult:
    """Trace the **boundaries of dark regions**: luminance threshold -> contours.

    The cheapest and best-looking mode for logos, text, icons and silhouettes:
    one closed curve per region, and holes come out as their own curves (so an
    "O" is two circles). ``threshold`` is a fraction of full black-to-white
    contrast, 0..1; by default Otsu picks it from the image's own histogram.
    ``invert=True`` traces the boundaries of the *light* regions instead, which
    on a light background adds the image frame as a contour.

    A uniform image traces to nothing, whatever ``invert`` says.

    Each contour costs one message per curve after simplification -- typically
    tens for a logo. See :func:`trace_image` for the budget rules.
    """
    loaded = _load(path, max_dimension=max_dimension, blur=blur)
    mask, level = _threshold_mask(loaded, threshold, invert)
    return _trace_contours(
        loaded,
        mask,
        mode="threshold",
        rect=_destination(rect),
        max_segments=max_segments,
        min_contour_length=min_contour_length,
        simplify=simplify,
        preserve_aspect=preserve_aspect,
        threshold=level,
        frame_coverage=frame_coverage,
    )


def trace_edges(
    path: ImageSource,
    rect: Rect | None = None,
    *,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    threshold: float | None = None,
    blur: float = DEFAULT_BLUR,
    min_contour_length: float = DEFAULT_MIN_CONTOUR_LENGTH,
    invert: bool = False,
    simplify: float = DEFAULT_SIMPLIFY,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    preserve_aspect: bool = True,
    frame_coverage: float = DEFAULT_FRAME_COVERAGE,
) -> TraceResult:
    """Trace **edges**: Sobel gradient magnitude, thresholded, then contoured.

    This is the mode for images that have no single luminance that separates
    subject from background -- but be clear about what it gives you. There is
    no non-maximum suppression and no hysteresis (so this is "Canny-ish", not
    Canny): edges come out as *bands*, and tracing a band's boundary draws both
    of its sides, so edge mode costs roughly twice what :func:`trace_threshold`
    does for the same picture and tends to look like a double outline.

    ``threshold`` is a fraction 0..1 of full contrast, read as a gradient of
    that many grey levels per pixel; by default it is Otsu's over the magnitude
    with a floor of :data:`EDGE_MIN_GRADIENT`. The floor is why a smooth
    gradient traces to **nothing**: a constant, shallow slope is not an edge.
    Because the image is downscaled first, the same threshold means different
    things at different ``max_dimension``.

    ``invert`` is accepted for signature symmetry and has **no effect** here:
    gradient magnitude is the same for an image and its negative.
    """
    del invert  # documented no-op: |∇I| == |∇(255-I)|
    loaded = _load(path, max_dimension=max_dimension, blur=blur)
    mask, level = _edge_mask(loaded, threshold)
    return _trace_contours(
        loaded,
        mask,
        mode="outline",
        rect=_destination(rect),
        max_segments=max_segments,
        min_contour_length=min_contour_length,
        simplify=simplify,
        preserve_aspect=preserve_aspect,
        threshold=level,
        frame_coverage=frame_coverage,
    )


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` column spans of consecutive True values."""
    if not flags.any():
        return []
    padded = np.concatenate(([False], flags.astype(bool), [False]))
    deltas = np.diff(padded.astype(np.int8))
    starts = np.nonzero(deltas == 1)[0]
    ends = np.nonzero(deltas == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _hairpin_bands(
    mask: np.ndarray, row_height: float, min_run: float
) -> list[tuple[int, Segment, float]]:
    """Hairpin strokes covering ``mask``, one per run of ink per band.

    Band ``k`` covers source ``y`` in ``[2k·h, 2(k+1)·h)`` and is drawn by one
    :func:`geometry.hairpin_segment` whose two pen rows sit at ``+0.5h`` and
    ``+1.5h``, so consecutive bands tile exactly with a pen ``h`` tall. A band
    is filled where at least :data:`FILL_COVERAGE` of its pixel rows are ink.

    A hairpin's control points overshoot its tip by 4/3
    (:data:`geometry.HAIRPIN_OVERSHOOT`), and :func:`_place` has to fit those
    inside the destination rect, so a tip that pokes out of the image shrinks
    the whole picture. Each run is therefore drawn in whichever direction keeps
    its tip inside the image; where both work, alternate bands go right-to-left
    (the serpentine of :func:`geometry.hairpin_fill`, which makes no difference
    to the picture but stops the pencil cursor and name tag from strobing
    across the share), and where neither works -- a run spanning the full width
    -- all the overshoot is put on the same side, which costs less scale than
    alternating would.

    Returns ``(band index, segment, run width)`` in drawing order.
    """
    height, width = mask.shape
    scale = float(max(width, height))
    span = width / scale
    reach = geometry.HAIRPIN_OVERSHOOT
    bands = max(1, math.ceil((height / scale) / (2.0 * row_height)))
    out: list[tuple[int, Segment, float]] = []
    for k in range(bands):
        top = k * 2.0 * row_height
        r0 = int(math.floor(top * scale))
        r1 = int(math.ceil((top + 2.0 * row_height) * scale))
        rows = mask[max(0, r0) : min(height, r1)]
        if rows.size == 0:
            continue
        covered = rows.mean(axis=0) >= FILL_COVERAGE
        for c0, c1 in _runs(covered):
            x0 = c0 / scale
            x1 = (c1 + 1) / scale
            if x1 - x0 < min_run:
                continue
            rightward = x0 + (x1 - x0) * reach <= span
            leftward = x1 + (x0 - x1) * reach >= 0.0
            if rightward and leftward:
                left, right = (x1, x0) if k % 2 else (x0, x1)
            elif leftward:
                left, right = x1, x0
            else:
                left, right = x0, x1
            out.append(
                (
                    k,
                    geometry.hairpin_segment(
                        left, right, top + 0.5 * row_height, row_height
                    ),
                    x1 - x0,
                )
            )
    return out


def trace_fill(
    path: ImageSource,
    rect: Rect | None = None,
    *,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    threshold: float | None = None,
    blur: float = DEFAULT_BLUR,
    min_contour_length: float = DEFAULT_MIN_CONTOUR_LENGTH,
    invert: bool = False,
    fill_row_height: float | None = None,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    preserve_aspect: bool = True,
) -> TraceResult:
    """Fill dark regions with **hairpins**: one message per band two pens tall.

    The doc's second message-saving trick: a cubic
    ``a=(x0,y), b=c=(x1',y)…(x1',y+1), d=(x0,y+1)`` runs out along one row and
    back along the next, so one message covers two pen rows -- half the
    messages of plain hatching. Use it for solid silhouettes; it is the only
    mode that produces something that reads as *filled* rather than outlined.

    It is also by far the most expensive mode: a shape covering half of a
    1080p share costs roughly ``0.5 / (2 · 0.0065) ≈ 38`` messages per full
    width of coverage, and a complex mask multiplies that by the number of ink
    runs per band. Check :meth:`TraceResult.estimated_seconds` before sending.

    ``fill_row_height`` is the row pitch in **destination** units and defaults
    to one 7 px pen on a 1080p share (:data:`DEFAULT_FILL_ROW_HEIGHT`); it is
    converted into source units before the bands are generated, so the bands
    still touch after the fit into ``rect``. ``min_contour_length`` is reused
    here as the minimum run *width*: narrower runs are noise and get dropped.
    ``invert=True`` fills the light regions instead.

    Two things to expect from it. A band is filled where at least
    :data:`FILL_COVERAGE` of its pixel rows are ink, so **features thinner than
    a band disappear** -- a 4 px ring traces to nothing in fill mode. And a
    hairpin's control points overshoot its tip by 4/3, which has to fit inside
    ``rect`` like everything else, so a fill whose runs span the whole image is
    scaled to about three quarters of the rect.

    Degradation when over budget: coarsen the row pitch first (the fill gets
    stripy but stays whole, up to :data:`MAX_FILL_ROW_HEIGHT`), then drop the
    narrowest runs, then -- only for absurd budgets -- slice.
    """
    loaded = _load(path, max_dimension=max_dimension, blur=blur)
    mask, level = _threshold_mask(loaded, threshold, invert)
    destination = _destination(rect)
    scale = _vertical_scale(loaded, destination, preserve_aspect)
    filled = _fill_segments(
        mask,
        max_segments=max_segments,
        min_run=min_contour_length,
        fill_row_height=fill_row_height,
        scale=scale,
    )
    result = TraceResult(
        segments=_place(filled.segments, loaded, destination, preserve_aspect),
        contours=filled.bands,
        source_size=loaded.source_size,
        mode="fill",
        traced_size=loaded.size,
        rect=destination,
        max_segments=max_segments,
        contours_found=filled.found,
        contours_filtered=0,
        contours_dropped=filled.dropped,
        simplify_epsilon=0.0,
        fill_row_height=filled.row_height * scale,
        threshold=level,
        truncated=filled.truncated,
    )
    logger.info("traced image: %s", result.describe())
    return result


@dataclass(frozen=True, slots=True)
class _Filled:
    """Hairpin segments in **source** units -- not yet fitted into a rect."""

    segments: list[Segment]
    #: Distinct bands that got at least one stroke.
    bands: int
    #: Hairpin runs generated before the budget dropped any.
    found: int
    dropped: int
    #: The row pitch finally used, in **source** units.
    row_height: float
    truncated: bool


def _fill_segments(
    mask: np.ndarray,
    *,
    max_segments: int,
    min_run: float,
    fill_row_height: float | None,
    scale: float,
) -> _Filled:
    """Hairpin-fill a mask within a budget, stopping short of placement.

    ``scale`` is destination units per source unit (:func:`_vertical_scale`):
    the row pitch is given and reported in destination units but the bands are
    generated in source units, so they still touch after the fit into the rect.
    Split out of :func:`trace_fill` for :func:`trace_image_colors`, which fills
    one mask per colour and places them all with one shared transform.
    """
    target_row = float(fill_row_height or DEFAULT_FILL_ROW_HEIGHT)
    if target_row <= 0:
        raise ValueError("fill_row_height must be positive")

    row = target_row / scale
    ceiling = MAX_FILL_ROW_HEIGHT / scale
    bands: list[tuple[int, Segment, float]] = []
    if mask.any() and max_segments > 0:
        while True:
            bands = _hairpin_bands(mask, row, min_run)
            if len(bands) <= max_segments or row >= ceiling:
                break
            row = min(row * FILL_ROW_GROWTH, ceiling)
            logger.info(
                "fill over budget: coarsening the row pitch to %.4f", row * scale
            )

    dropped = 0
    truncated = False
    if len(bands) > max_segments:
        # Keep the widest runs: they are the ones that read as "filled".
        widest = sorted(
            range(len(bands)), key=lambda i: bands[i][2], reverse=True
        )[:max_segments]
        dropped = len(bands) - len(widest)
        bands = [bands[i] for i in sorted(widest)]
        logger.warning(
            "fill dropped %d of %d hairpin runs to fit %d segments",
            dropped,
            dropped + len(bands),
            max_segments,
        )
        if len(bands) > max_segments:  # pragma: no cover - defensive
            bands = bands[:max_segments]
            truncated = True

    return _Filled(
        segments=[segment for _, segment, _ in bands],
        bands=len({k for k, _, _ in bands}),
        found=len(bands) + dropped,
        dropped=dropped,
        row_height=row,
        truncated=truncated,
    )


def _trace_auto(
    loaded: _Loaded,
    *,
    rect: Rect,
    max_segments: int,
    threshold: float | None,
    min_contour_length: float,
    invert: bool,
    simplify: float,
    preserve_aspect: bool,
    frame_coverage: float = DEFAULT_FRAME_COVERAGE,
) -> TraceResult:
    """``"auto"``: try threshold, fall back to edges. That is the whole trick.

    Plainly: there is no content classifier here. It thresholds the image, and
    if that gives at least one contour and an ink fraction between
    :data:`AUTO_MIN_INK` and :data:`AUTO_MAX_INK` (outside that band the image
    is near-blank or near-solid and its only "region boundary" is the frame),
    it keeps that. Otherwise it runs the edge tracer. If both come back empty,
    the empty threshold result is returned.
    """
    mask, level = _threshold_mask(loaded, threshold, invert)
    ink = float(mask.mean()) if mask.size else 0.0
    threshold_result: TraceResult | None = None
    if mask.any() and AUTO_MIN_INK <= ink <= AUTO_MAX_INK:
        threshold_result = _trace_contours(
            loaded,
            mask,
            mode="threshold",
            rect=rect,
            max_segments=max_segments,
            min_contour_length=min_contour_length,
            simplify=simplify,
            preserve_aspect=preserve_aspect,
            threshold=level,
            frame_coverage=frame_coverage,
        )
        if threshold_result.segments:
            logger.info("auto chose threshold (ink fraction %.3f)", ink)
            return threshold_result
    logger.info(
        "auto falling back to edges (ink fraction %.3f, threshold trace %s)",
        ink,
        "empty" if threshold_result is not None else "skipped",
    )
    edge_mask, edge_level = _edge_mask(loaded, threshold)
    edges = _trace_contours(
        loaded,
        edge_mask,
        mode="outline",
        rect=rect,
        max_segments=max_segments,
        min_contour_length=min_contour_length,
        simplify=simplify,
        preserve_aspect=preserve_aspect,
        threshold=edge_level,
        frame_coverage=frame_coverage,
    )
    if edges.segments or threshold_result is None:
        return edges
    return threshold_result


def trace_image(
    path: ImageSource,
    rect: Rect | None = None,
    *,
    mode: TraceMode | str = "outline",
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    threshold: float | None = None,
    blur: float = DEFAULT_BLUR,
    min_contour_length: float = DEFAULT_MIN_CONTOUR_LENGTH,
    invert: bool = False,
    simplify: float = DEFAULT_SIMPLIFY,
    fill_row_height: float | None = None,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    preserve_aspect: bool = True,
    frame_coverage: float = DEFAULT_FRAME_COVERAGE,
) -> TraceResult:
    """Trace an image into cubic Bézier segments, within a segment budget.

    Args:
        path: image file (anything Pillow decodes), or an open
            :class:`PIL.Image.Image`. EXIF orientation is honoured, alpha is
            composited onto **white**, and the image is downscaled to
            ``max_dimension`` before tracing.
        rect: where to put the drawing, in normalized 0..1 share coordinates.
            Defaults to the whole visible area. Every returned coordinate --
            control points included -- is inside this rect.
        mode: ``"outline"`` (edges, alias ``"edges"``), ``"threshold"`` (region
            boundaries), ``"fill"`` (hairpin bands), or ``"auto"``. See
            :func:`trace_edges`, :func:`trace_threshold`, :func:`trace_fill`
            and :func:`_trace_auto` for what each one is good for.
        max_segments: **a hard budget, not a hint** -- see below.
        threshold: 0..1 fraction of full contrast; ``None`` means Otsu.
        blur: Gaussian blur radius in traced pixels, applied before
            thresholding; suppresses speckle that would become its own contour.
        min_contour_length: contours shorter than this (perimeter, normalized
            to the image's longer side) are dropped as noise. In ``"fill"``
            mode it is the minimum run width instead.
        invert: trace the light regions rather than the dark ones. No effect in
            ``"outline"`` mode (gradient magnitude ignores polarity).
        simplify: starting RDP epsilon, in normalized units.
        fill_row_height: hairpin row pitch for ``"fill"`` mode, in destination
            units; ignored by the other modes.
        max_dimension: longest side actually traced.
        frame_coverage: reject any contour that covers more than this fraction
            of both axes of the traced area *and* is near-rectangular -- that
            is the image's own border, not anything in the picture, and drawing
            it puts a box around the whole share (``invert=True`` on a light
            background is the classic way to produce one). ``1.0`` keeps it.
            Ignored in ``"fill"`` mode, which has no contours.
        preserve_aspect: keep the image's aspect ratio when fitting into
            ``rect`` (centred on the slack axis). Note "aspect" is aspect in
            *normalized* space, which is on-screen aspect only if the share is
            square -- the wire format normalizes each axis independently.

    Returns:
        A :class:`TraceResult`. ``estimated_seconds()`` is the number to look
        at before sending: at Chime's ~150 messages/s *per meeting* (shared
        with every other sender, excess dropped silently), 600 segments is ~4 s
        of solid traffic and 2000 is ~13 s.

    **The budget.** ``max_segments`` is never exceeded. Hitting it degrades in
    a fixed order, so the result is predictable: first the simplification
    epsilon is raised for *all* contours together (the picture stays whole and
    gets smoother), then whole contours are dropped **shortest first**, and
    only for budgets too small for a single contour is the list sliced
    (``TraceResult.truncated``). Nothing is ever cut off mid-drawing to make it
    fit: ask for 600 segments and you get a rough version of the whole picture,
    not a crisp version of its top-left corner. What was given up is reported
    in ``contours_filtered`` / ``contours_dropped`` / ``truncated`` and logged.

    **Honest limitations.** The output is a *single-hue line drawing*: the pen
    is 7 px round and fixed, the colour is always ``hsla(hue, 100%, 75%)``, and
    one attendee owns one line in one hue -- so there is no shading, no line
    weight and no second colour. Logos, line art, text and high-contrast
    silhouettes trace well. **Photographs trace badly** in every mode: there is
    rarely a luminance that separates subject from background, so you get a few
    hundred messages of disconnected confetti. Soft gradients have no edges at
    all and trace to nothing. Detail finer than one pen width
    (:func:`geometry.pen_size_normalized`) cannot survive, and Catmull-Rom
    rounds sharp corners, so a square comes out slightly pillowed.
    """
    chosen = "outline" if mode == "edges" else str(mode)
    if chosen not in ("outline", "threshold", "fill", "auto"):
        raise ValueError(
            f"unknown mode {mode!r}; expected outline/edges, threshold, fill "
            "or auto"
        )
    if max_segments < 0:
        raise ValueError("max_segments must be >= 0")
    if simplify < 0:
        raise ValueError("simplify must be >= 0")
    if min_contour_length < 0:
        raise ValueError("min_contour_length must be >= 0")
    if max_dimension < 1:
        raise ValueError("max_dimension must be >= 1")
    if fill_row_height is not None and chosen != "fill":
        logger.debug("fill_row_height is ignored in %s mode", chosen)

    if chosen == "fill":
        return trace_fill(
            path,
            rect,
            max_segments=max_segments,
            threshold=threshold,
            blur=blur,
            min_contour_length=min_contour_length,
            invert=invert,
            fill_row_height=fill_row_height,
            max_dimension=max_dimension,
            preserve_aspect=preserve_aspect,
        )
    if chosen == "threshold":
        return trace_threshold(
            path,
            rect,
            max_segments=max_segments,
            threshold=threshold,
            blur=blur,
            min_contour_length=min_contour_length,
            invert=invert,
            simplify=simplify,
            max_dimension=max_dimension,
            preserve_aspect=preserve_aspect,
            frame_coverage=frame_coverage,
        )
    if chosen == "outline":
        return trace_edges(
            path,
            rect,
            max_segments=max_segments,
            threshold=threshold,
            blur=blur,
            min_contour_length=min_contour_length,
            invert=invert,
            simplify=simplify,
            max_dimension=max_dimension,
            preserve_aspect=preserve_aspect,
            frame_coverage=frame_coverage,
        )
    return _trace_auto(
        _load(path, max_dimension=max_dimension, blur=blur),
        rect=_destination(rect),
        max_segments=max_segments,
        threshold=threshold,
        min_contour_length=min_contour_length,
        invert=invert,
        simplify=simplify,
        preserve_aspect=preserve_aspect,
        frame_coverage=frame_coverage,
    )


# --------------------------------------------------------------------------
# colour layers: one hue per attendee (a `multidevice` join per layer)
# --------------------------------------------------------------------------

#: Default number of colour layers (i.e. attendees) to decompose into.
DEFAULT_LAYERS = 3

#: Layers covering less than this fraction of the traced ink are dropped: a
#: 1%-of-the-image layer is not worth a whole attendee.
DEFAULT_MIN_LAYER_SHARE = 0.02

#: Hues closer than this (degrees on the circle) render as the same pastel at
#: ``hsla(hue, 100%, 75%)``, so such layers are merged instead of spending a
#: second attendee on a colour nobody can tell apart.
DEFAULT_HUE_MERGE_DEGREES = 12.0

#: Mean chroma -- ``(max(R,G,B) - min(R,G,B)) / 255``, averaged over a
#: cluster's pixels -- below which the cluster has no *usable* hue. Black,
#: white and every grey sit at chroma 0, but the threshold is this high
#: (rather than near zero) for a harder reason: the renderer forces
#: ``hsla(hue, 100%, 75%)``, so it takes a barely-present tint and amplifies it
#: into a vivid pastel. Measured on a real pastel cartoon, a 0.16-chroma pale
#: cream came out as vivid salmon -- numerically correct, perceptually
#: arbitrary. Anything under this is therefore treated as neutral and folded
#: into the one greyscale layer instead of being given a confident colour.
#: Chroma rather than HSL saturation on purpose: ``rgb(10, 0, 0)`` is
#: HSL-saturation 1.0 but is visually black.
DEFAULT_MIN_CHROMA = 0.18

#: Minimum separation, in degrees, between the hues finally handed out, when
#: ``spread_hues`` is on. Pastel source art lands every cluster in a narrow
#: arc of near-neutrals, and the forced saturation then renders them as
#: near-identical vivid colours; spreading trades absolute fidelity (already
#: impossible) for layers a viewer can actually tell apart.
DEFAULT_MIN_HUE_SEPARATION = 40.0

#: Default ``min_contour_length`` for **colour** layers, as a perimeter
#: fraction of the traced area's longer side. Higher than
#: :data:`DEFAULT_MIN_CONTOUR_LENGTH` because a colour layer costs an attendee
#: and a pile of dots is not worth one: on a cartoon with a repeating
#: background motif, the motif traced as dozens of specks and read as noise.
DEFAULT_MIN_LAYER_CONTOUR = 0.10

#: A colour layer whose **biggest** contour is shorter than this (perimeter, as
#: a fraction of the traced area's longer side) is speckle, however much ink it
#: adds up to, and is dropped: 0.30 is roughly a 7%-of-frame box. Set to 0 to
#: keep speckle layers.
DEFAULT_MIN_LAYER_FEATURE = 0.30

#: Fraction of the image's **border ring** a cluster has to own before it is
#: called the background. Structural, not colour-based: whatever reaches all
#: the way around the edge of the picture is the page behind it.
DEFAULT_BACKGROUND_BORDER_SHARE = 0.5

#: How many pixels deep the border ring sampled for background detection is.
BORDER_RING_PX = 2

#: How much a cluster's coloured pixels have to **agree** on a hue before that
#: hue is used (the circular resultant length, 0..1). A cluster straddling two
#: hues -- the dark fringe blur leaves around a red shape and a green one, say
#: -- averages to a confident third hue that is nowhere in the picture. Below
#: this the cluster is treated as having no usable hue, exactly like a neutral.
MIN_HUE_CONCENTRATION = 0.6

#: Luminance fraction above which an **unsaturated** pixel counts as the page
#: rather than as ink. White cannot be drawn at all (every colour is a pastel),
#: so near-white is the one thing that must not become a layer. Lower it to
#: treat mid-greys as background too; ``invert=True`` flips it to near-black.
DEFAULT_BACKGROUND_LEVEL = 0.90

#: The hue given to the near-greyscale layer. Arbitrary by construction --
#: grey has no hue, and the renderer cannot draw grey -- so pick your own.
#: Fallback hue for a near-neutral layer when there are no chromatic layers to
#: stay away from (a greyscale image). Any value is arbitrary -- the renderer
#: forces S=100% L=75%, so grey, white and black cannot be drawn at all -- but
#: a cool blue is the least assertive pastel available, whereas hue 0 is red
#: and reads as a confident claim about the picture. When there *are* chromatic
#: layers, `place_neutral_hue` ignores this and uses the widest gap instead.
DEFAULT_GREY_HUE = 210

#: Smallest segment budget worth giving a layer: below about this a closed
#: contour cannot be drawn at all, so fewer layers with usable budgets beat
#: more layers with none.
MIN_LAYER_SEGMENTS = 4

#: Bits per channel used to bin colours before clustering (32^3 bins).
COLOR_BINS_BITS = 5

#: Lloyd iterations for the colour k-means. It runs on a weighted colour
#: histogram, not on pixels, so this converges in a handful of rounds.
KMEANS_ITERATIONS = 32


def rgb_to_hue(rgb: Sequence[float]) -> int:
    """The HSL/HSV hue of an ``(r, g, b)`` colour, as degrees 0..359.

    This is the **only** thing about a colour that survives to the screen:
    ``draw.hue`` is an int32 of degrees and the renderer draws
    ``hsla(hue, 100%, 75%)``, with saturation, lightness, width and caps all
    fixed (``docs/huddles-peer-messages.md``, "Lines, colours and accounts").
    Returns ``0`` for any grey, which has no hue -- see
    :data:`DEFAULT_MIN_CHROMA` for how :func:`trace_image_colors` tells the
    difference.
    """
    r, g, b = (float(c) / 255.0 for c in rgb)
    high, low = max(r, g, b), min(r, g, b)
    delta = high - low
    if delta <= 0.0:
        return 0
    if high == r:
        hue = 60.0 * (((g - b) / delta) % 6.0)
    elif high == g:
        hue = 60.0 * (((b - r) / delta) + 2.0)
    else:
        hue = 60.0 * (((r - g) / delta) + 4.0)
    return int(round(hue)) % 360


def pastel_rgb(hue: int) -> tuple[int, int, int]:
    """What ``hsla(hue, 100%, 75%)`` actually looks like, as 8-bit RGB.

    Every channel comes out in 128..255, which is the whole limitation in one
    line: **nothing can be drawn dark, grey or white**, only one of 360
    pastels. Useful for previewing a decomposition honestly rather than in the
    image's own colours.
    """
    c = 0.5  # (1 - |2*0.75 - 1|) * 1.00
    h = (float(hue) % 360.0) / 60.0
    x = c * (1.0 - abs((h % 2.0) - 1.0))
    sector = int(h) % 6
    rgb = (
        (c, x, 0.0),
        (x, c, 0.0),
        (0.0, c, x),
        (0.0, x, c),
        (x, 0.0, c),
        (c, 0.0, x),
    )[sector]
    m = 0.75 - c / 2.0
    return tuple(int(round((v + m) * 255.0)) for v in rgb)  # type: ignore[return-value]


def _hue_distance(a: int, b: int) -> float:
    """Shortest distance between two hues, in degrees (0..180)."""
    d = abs(int(a) - int(b)) % 360
    return float(min(d, 360 - d))


# --------------------------------------------------------------------------
# perceptual colour space
# --------------------------------------------------------------------------

#: Linear sRGB -> LMS-ish cone responses, and the cube roots of those -> OkLab.
#: Constants from Björn Ottosson's OkLab definition.
_OKLAB_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float64,
)
_OKLAB_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)


def _srgb_to_linear(channels: np.ndarray) -> np.ndarray:
    """Undo the sRGB transfer function; input and output in 0..1."""
    x = np.clip(np.asarray(channels, dtype=np.float64), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """8-bit sRGB ``(..., 3)`` -> OkLab ``(..., 3)``.

    Clustering happens here rather than in raw RGB because Euclidean distance
    in RGB does not match what the eye groups: a step between two dark colours
    looks far smaller than the same step between two light ones, so k-means in
    RGB splits bright regions and lumps shadows together, and the resulting
    "colours" feel arbitrary. OkLab is perceptually uniform and closed-form
    from linear sRGB, so it costs one matrix multiply and a cube root -- no
    scipy, no colour-science dependency.
    """
    linear = _srgb_to_linear(np.asarray(rgb, dtype=np.float64) / 255.0)
    lms = linear @ _OKLAB_M1.T
    return np.cbrt(lms) @ _OKLAB_M2.T


def _rgb_chroma(rgb: np.ndarray) -> np.ndarray:
    """``(max - min) / 255`` per colour: how far it is from neutral, 0..1.

    The measure :data:`DEFAULT_MIN_CHROMA` is expressed in, and the weight used
    when a cluster's hue is averaged -- a cluster's colour should be decided by
    the pixels that actually *have* a colour, not washed toward grey by the
    neutral majority.
    """
    values = np.asarray(rgb, dtype=np.float64)
    return (values.max(axis=-1) - values.min(axis=-1)) / 255.0


def _weighted_hue(hues: np.ndarray, weights: np.ndarray) -> tuple[int, float]:
    """Circular mean of ``hues`` (degrees) weighted by ``weights``.

    Circular because hue wraps: the plain mean of 350 and 10 is 180 (cyan),
    which is the opposite of the right answer (0, red).

    Also returns the **concentration** 0..1 (the resultant length): how much
    the weighted hues agree. 1.0 is one hue, and a cluster holding equal parts
    red and green comes out at 0.5 with a mean of yellow -- a hue that is in
    the arithmetic but not in the picture. See
    :data:`MIN_HUE_CONCENTRATION`.
    """
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum()) if w.size else 0.0
    if total <= 0.0:
        return 0, 0.0
    radians = np.radians(np.asarray(hues, dtype=np.float64))
    x = float((w * np.cos(radians)).sum())
    y = float((w * np.sin(radians)).sum())
    concentration = math.hypot(x, y) / total
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return 0, 0.0  # the hues cancel out exactly: no mean direction
    return int(round(math.degrees(math.atan2(y, x)))) % 360, concentration


@dataclass(frozen=True, slots=True)
class ColorLayer:
    """One colour's worth of a decomposed image: **one attendee's line**.

    Slack gives every attendee exactly one line in one hue, so a drawing in N
    colours needs N attendees, each sending the segments of one of these
    (``docs/huddles-peer-messages.md``, "Lines, colours and accounts"). One
    account can supply several attendees when it joins with
    ``multidevice=true`` (each join then returns a fresh AttendeeId), so what a
    layer really costs is an attendee, not necessarily a login.

    ``segments`` are :class:`geometry.Segment` values in normalized 0..1
    coordinates, already fitted into the requested rect with the **same**
    transform as every other layer -- so the layers register with each other
    and can be sent concurrently by different attendees. One segment is one
    ``draw`` message against the meeting's shared ~150/s bucket.
    """

    #: Hue in degrees 0..359 to send as ``CHANGE_HUE`` before drawing.
    hue: int
    #: This layer's cubic Béziers, normalized 0..1, in draw order.
    segments: list[Segment]
    #: Fraction of the image's traced ink this layer covers, 0..1. Not
    #: renormalized after filtering, so the returned shares sum to less than 1
    #: exactly when layers were dropped.
    pixel_share: float
    #: The cluster's representative colour -- the colour in it that carries the
    #: most chroma-weighted pixels, before the renderer flattens it to a
    #: pastel. Compare with :attr:`rendered_rgb` to see what is lost.
    sample_rgb: tuple[int, int, int]
    #: Full trace detail for this layer: budget, contours, what was dropped.
    trace: TraceResult | None = None
    #: True when this layer came from near-neutral pixels and its :attr:`hue`
    #: is therefore arbitrary (see :func:`trace_image_colors`).
    is_greyscale: bool = False
    #: How many colour clusters were merged into this layer (1 = none were).
    merged_clusters: int = 1
    #: True for the layer identified as the image's background. Only ever set
    #: on a returned layer when ``drop_background=False``.
    is_background: bool = False
    #: The hue the pixels actually had, when :attr:`hue` was moved off it to
    #: keep the layers distinguishable (``spread_hues``). ``None`` when the
    #: hue is the measured one.
    original_hue: int | None = None
    #: Mean chroma of the cluster's pixels, 0..1: how much colour there was to
    #: work with. Low values are why a hue can look arbitrary.
    chroma: float = 0.0

    def __len__(self) -> int:
        return len(self.segments)

    def __bool__(self) -> bool:
        return bool(self.segments)

    @property
    def rendered_rgb(self) -> tuple[int, int, int]:
        """What this layer will actually look like: :func:`pastel_rgb`."""
        return pastel_rgb(self.hue)

    def css_color(self, opacity: float = 1.0) -> str:
        """The exact colour string the renderer uses for this layer."""
        return f"hsla({self.hue}, 100%, 75%, {opacity:g})"

    def estimated_seconds(
        self, rate: float = config.CHIME_DATA_MESSAGE_RATE
    ) -> float:
        """Wall-clock time to send this layer, as :meth:`TraceResult.estimated_seconds`.

        Same caveats, and one more that matters here: the ~150 messages/s is
        **per meeting**, not per sender, so N layers drawing at once share it.
        Use :func:`total_estimated_seconds` for the number that predicts how
        long a multi-attendee drawing really takes.
        """
        if rate <= 0:
            raise ValueError("rate must be positive")
        return len(self.segments) / float(rate)

    def describe(self) -> str:
        """One line for logs: hue, cost, share, and source colour -> drawn colour.

        The ``->`` is the point of it: the source colour is what the image has
        and the rendered colour is what the share will show, and on pastel art
        those are startlingly different (pale cream comes out salmon). Seeing
        both before spending messages is the difference between "the colours
        look random" and "of course, S is forced to 100%".
        """
        parts = [
            f"hue={self.hue:3d}",
            f"{len(self.segments)} segments",
            f"{self.pixel_share * 100:.1f}% of ink",
            "rgb({},{},{}) -> rgb({},{},{})".format(
                *self.sample_rgb, *self.rendered_rgb
            ),
            f"chroma={self.chroma:.2f}",
        ]
        if self.original_hue is not None:
            parts.append(f"hue moved from {self.original_hue}")
        if self.is_greyscale:
            parts.append("near-neutral (hue is arbitrary)")
        if self.is_background:
            parts.append("BACKGROUND")
        if self.merged_clusters > 1:
            parts.append(f"{self.merged_clusters} clusters merged")
        if self.trace is not None and self.trace.truncated:
            parts.append("TRUNCATED")
        return ", ".join(parts)


def total_estimated_seconds(
    layers: Sequence[ColorLayer], rate: float = config.CHIME_DATA_MESSAGE_RATE
) -> float:
    """Wall-clock time to send **all** the layers, consistent with per-layer.

    The sum, not the max: Chime's ~150 data messages/s is a per-*meeting* token
    bucket shared by every sender, so drawing six layers from six attendees at
    once does not go six times faster -- it just means the bucket drains six
    times as fast and the excess is dropped silently.
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    return sum(len(layer.segments) for layer in layers) / float(rate)


def describe_layers(layers: Sequence[ColorLayer]) -> str:
    """Multi-line summary of a decomposition: one line per layer, plus a total."""
    lines = [layer.describe() for layer in layers]
    lines.append(
        f"{len(layers)} layer(s) = {len(layers)} attendee(s), "
        f"{sum(len(x) for x in layers)} segments, "
        f"~{total_estimated_seconds(layers):.1f}s at "
        f"{config.CHIME_DATA_MESSAGE_RATE:.0f}/s"
    )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _Cluster:
    """One colour cluster: the pixels that belong to it and what colour it is."""

    mask: np.ndarray
    rgb: tuple[int, int, int]
    hue: int
    share: float
    is_grey: bool
    #: Mean chroma over the cluster's pixels, 0..1.
    chroma: float = 0.0
    merged: int = 1
    original_hue: int | None = None
    is_background: bool = False

    def merged_with(self, other: _Cluster) -> _Cluster:
        """This cluster plus ``other``'s pixels, keeping *this* one's colour."""
        total = self.share + other.share
        weights = (self.share, other.share) if total > 0 else (1.0, 1.0)
        return _Cluster(
            mask=self.mask | other.mask,
            rgb=self.rgb,
            hue=self.hue,
            share=total,
            is_grey=self.is_grey or other.is_grey,
            chroma=(
                self.chroma * weights[0] + other.chroma * weights[1]
            ) / sum(weights),
            merged=self.merged + other.merged,
            original_hue=self.original_hue,
            is_background=self.is_background or other.is_background,
        )

    def with_hue(self, hue: int, *, record: bool = False) -> _Cluster:
        """The same cluster drawn in a different hue.

        ``record=True`` remembers the hue the pixels actually had, so a caller
        can see that this layer's colour was moved and why.
        """
        new = int(hue) % 360
        return _Cluster(
            mask=self.mask,
            rgb=self.rgb,
            hue=new,
            share=self.share,
            is_grey=self.is_grey,
            chroma=self.chroma,
            merged=self.merged,
            original_hue=(
                self.original_hue
                if self.original_hue is not None
                else (self.hue if record and new != self.hue else None)
            ),
            is_background=self.is_background,
        )

    def as_background(self) -> _Cluster:
        return _Cluster(
            mask=self.mask,
            rgb=self.rgb,
            hue=self.hue,
            share=self.share,
            is_grey=self.is_grey,
            chroma=self.chroma,
            merged=self.merged,
            original_hue=self.original_hue,
            is_background=True,
        )


def _init_centres(
    colors: np.ndarray, weights: np.ndarray, k: int
) -> list[int]:
    """Deterministic k-means++: the heaviest colour, then farthest-by-weight.

    Plain k-means++ samples the next centre with probability proportional to
    ``d^2``; taking the argmax of ``d^2 * weight`` instead makes the result
    reproducible (tests depend on that) and stops a single stray pixel from
    becoming a whole attendee's worth of layer.
    """
    chosen = [int(np.argmax(weights))]
    if k <= 1:
        return chosen
    d2 = ((colors - colors[chosen[0]]) ** 2).sum(axis=1)
    while len(chosen) < k:
        score = d2 * weights
        index = int(np.argmax(score))
        if score[index] <= 0.0:
            break  # every remaining colour already sits on a centre
        chosen.append(index)
        d2 = np.minimum(d2, ((colors - colors[index]) ** 2).sum(axis=1))
    return chosen


def _weighted_kmeans(
    colors: np.ndarray,
    weights: np.ndarray,
    k: int,
    *,
    iterations: int = KMEANS_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted Lloyd's algorithm, in plain numpy (no sklearn/scipy).

    ``colors`` is whatever space the caller wants distances measured in --
    :func:`_color_clusters` passes OkLab, because Euclidean distance in RGB
    does not group colours the way an eye does.

    Runs on a **colour histogram** -- one row per occupied colour bin, weighted
    by pixel count -- rather than on pixels, so the cost does not depend on the
    image size and the weighting keeps outliers out of the centres. Returns
    ``(labels, centres)`` with empty clusters removed, so the number of centres
    can come back smaller than ``k``.
    """
    m = colors.shape[0]
    k = max(1, min(int(k), m))
    centres = colors[_init_centres(colors, weights, k)].astype(np.float64)
    labels = np.full(m, -1, dtype=np.int64)
    for _ in range(max(1, iterations)):
        d2 = ((colors[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argmin(d2, axis=1).astype(np.int64)
        if np.array_equal(nearest, labels):
            break
        labels = nearest
        for j in range(centres.shape[0]):
            selected = labels == j
            total = float(weights[selected].sum())
            if total > 0.0:
                centres[j] = (
                    colors[selected] * weights[selected, None]
                ).sum(axis=0) / total
    used = np.nonzero(np.bincount(labels, minlength=centres.shape[0]))[0]
    lookup = np.full(centres.shape[0], -1, dtype=np.int64)
    lookup[used] = np.arange(len(used), dtype=np.int64)
    return lookup[labels], centres[used]


def _ink_mask(
    rgb: np.ndarray,
    loaded: _Loaded,
    *,
    background_level: float,
    min_chroma: float,
    invert: bool,
) -> np.ndarray:
    """Which pixels are drawable ink rather than the page.

    Near-white unsaturated pixels are the page, because white is the one thing
    the renderer *cannot* produce -- every colour is ``hsla(h, 100%, 75%)``.
    The chroma condition is what keeps pale yellows and baby blues as ink: they
    are light but they do have a hue. ``invert=True`` treats near-**black** as
    the page instead, for art on a dark background.
    """
    chroma = (rgb.max(axis=2) - rgb.min(axis=2)) / 255.0
    if invert:
        page = (loaded.grey <= (1.0 - background_level) * 255.0) & (
            chroma < min_chroma
        )
    else:
        page = (loaded.grey >= background_level * 255.0) & (chroma < min_chroma)
    return ~page


def _color_clusters(
    rgb: np.ndarray,
    loaded: _Loaded,
    *,
    layers: int,
    background_level: float,
    min_chroma: float,
    invert: bool,
    grey_hue: int,
) -> list[_Cluster]:
    """Quantise the image into at most ``layers`` colour clusters.

    Bins colours to :data:`COLOR_BINS_BITS` bits per channel, runs
    :func:`_weighted_kmeans` on that histogram **in OkLab**, then turns each
    cluster into a pixel mask with its share of the ink, its mean chroma, and
    the hue it will be drawn in.

    Two deliberate choices about a cluster's colour, both because the renderer
    keeps only the hue:

    * the hue is the **chroma-weighted circular mean** of the cluster's bins,
      so a cluster of pale skin with a saturated blush in it takes the blush's
      hue rather than a washed-out average that is one rounding away from a
      different colour entirely;
    * ``rgb`` is the cluster's highest chroma-weight *bin*, i.e. a colour that
      really is in the image, rather than a mean that may not be.

    Near-neutral clusters are folded into **one** layer (see
    :func:`trace_image_colors`). The result is sorted by share, largest first.
    """
    height, width, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    if flat.shape[0] == 0 or bool(np.all(flat == flat[0])):
        logger.debug("uniform image: no colours to separate")
        return []

    ink = _ink_mask(
        rgb,
        loaded,
        background_level=background_level,
        min_chroma=min_chroma,
        invert=invert,
    )
    total = int(ink.sum())
    if total == 0:
        logger.info(
            "colour trace: every pixel reads as page at background_level=%.2f",
            background_level,
        )
        return []

    samples = rgb[ink]
    shift = 8 - COLOR_BINS_BITS
    span = 1 << COLOR_BINS_BITS
    binned = (samples.astype(np.int32) >> shift).clip(0, span - 1)
    keys = (binned[:, 0] * span + binned[:, 1]) * span + binned[:, 2]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    bins = counts.shape[0]
    means = (
        np.stack(
            [
                np.bincount(inverse, weights=samples[:, c], minlength=bins)
                for c in range(3)
            ],
            axis=1,
        )
        / counts[:, None]
    )
    # Cluster perceptually, not in raw RGB -- but keep the sRGB means around,
    # because hue and chroma are defined on them.
    weights = counts.astype(np.float64)
    labels, _ = _weighted_kmeans(_rgb_to_oklab(means), weights, layers)
    cluster_count = int(labels.max()) + 1 if labels.size else 0
    logger.debug(
        "quantised %d ink pixel(s) in %d colour bin(s) into %d cluster(s) "
        "(k-means in OkLab)",
        total,
        bins,
        cluster_count,
    )

    bin_chroma = _rgb_chroma(means)
    bin_hues = np.array([rgb_to_hue(c) for c in means], dtype=np.float64)
    chroma_weight = weights * bin_chroma

    pixel_labels = np.full((height, width), -1, dtype=np.int64)
    pixel_labels[ink] = labels[inverse]

    clusters: list[_Cluster] = []
    for j in range(cluster_count):
        mask = pixel_labels == j
        count = int(mask.sum())
        if count == 0:  # pragma: no cover - defensive
            continue
        members = labels == j
        pixels = weights[members]
        # The cluster's colour comes from the pixels that *have* colour: a
        # chroma-weighted circular mean, not the arithmetic mean (which washes
        # a cluster toward grey and makes its hue unstable), and a
        # representative colour that is a real colour out of the image.
        chroma = float((pixels * bin_chroma[members]).sum() / pixels.sum())
        colour_weight = chroma_weight[members]
        if float(colour_weight.sum()) > 0.0:
            pick = int(np.argmax(colour_weight))
        else:
            pick = int(np.argmax(pixels))
        colour = tuple(
            int(v)
            for v in np.clip(np.round(means[members][pick]), 0, 255).astype(int)
        )
        hue, concentration = _weighted_hue(bin_hues[members], colour_weight)
        is_grey = chroma < min_chroma or concentration < MIN_HUE_CONCENTRATION
        if is_grey and chroma >= min_chroma:
            logger.info(
                "cluster rgb%s has chroma %.2f but its hues only agree to "
                "%.2f: treating it as having no usable hue rather than "
                "drawing it in the %d degrees the average happens to land on",
                colour,
                chroma,
                concentration,
                hue,
            )
        clusters.append(
            _Cluster(
                mask=mask,
                rgb=colour,  # type: ignore[arg-type]
                hue=grey_hue % 360 if is_grey else hue,
                share=count / total,
                is_grey=is_grey,
                chroma=chroma,
            )
        )

    clusters.sort(key=lambda c: c.share, reverse=True)
    return _fold_greys(clusters, grey_hue=grey_hue)


def place_neutral_hue(neutral_count: int, chromatic: Sequence[int]) -> list[int]:
    """Hues for near-neutral layers: the emptiest part of the colour wheel.

    A near-neutral region has no hue worth the name, but the renderer forces
    ``S=100% L=75%`` so it *must* be drawn in some vivid pastel -- grey, white
    and black are not available. The choice is therefore arbitrary, and the
    only question is which arbitrary value does least harm.

    A fixed value is the wrong answer, and ``0`` is the worst fixed value:
    hue 0 is **red**, which reads as a deliberate, confidently wrong colour.
    Observed live on a pastel illustration whose cream skin tones fell just
    under the chroma threshold: 147 segments of face drawn in vivid red.

    So neutrals are placed in the **largest gap between the chromatic layers'
    hues**, which maximises the distance from every colour that *is* meaningful
    and makes the neutral read as "the other one" rather than as a claim about
    the picture. With no chromatic layers at all (a greyscale image) this falls
    back to :data:`DEFAULT_GREY_HUE`.
    """
    if neutral_count <= 0:
        return []
    known = sorted({h % 360 for h in chromatic})
    if not known:
        # Nothing to avoid: spread the neutrals evenly from the default.
        return [
            (DEFAULT_GREY_HUE + round(i * 360 / neutral_count)) % 360
            for i in range(neutral_count)
        ]
    if len(known) == 1:
        gaps = [(360.0, known[0])]
    else:
        gaps = [
            (float(b - a), a) for a, b in zip(known, known[1:], strict=False)
        ]
        gaps.append((float(360 - known[-1] + known[0]), known[-1]))
    gaps.sort(reverse=True)
    width, start = gaps[0]
    # Space the neutrals inside the widest gap, away from both its edges.
    return [
        round(start + width * (i + 1) / (neutral_count + 1)) % 360
        for i in range(neutral_count)
    ]

def _reposition_neutral_layers(
    layers: list[ColorLayer], *, explicit_grey_hue: bool
) -> list[ColorLayer]:
    """Move near-neutral layers into the emptiest part of the colour wheel.

    See :func:`place_neutral_hue`. Skipped when the caller named a
    ``grey_hue`` explicitly -- that is a deliberate choice to respect.
    """
    if explicit_grey_hue:
        return layers
    neutrals = [i for i, layer in enumerate(layers) if layer.is_greyscale]
    if not neutrals:
        return layers
    chromatic = [layer.hue for layer in layers if not layer.is_greyscale]
    placed = place_neutral_hue(len(neutrals), chromatic)
    for index, hue in zip(neutrals, placed, strict=True):
        layer = layers[index]
        if layer.hue == hue:
            continue
        logger.info(
            "moved the near-neutral layer from hue %d to %d: a neutral has no "
            "hue, so it goes in the widest gap between the real colours "
            "(%s) rather than landing on an arbitrary one",
            layer.hue,
            hue,
            ", ".join(str(h) for h in sorted(chromatic)) or "none",
        )
        layers[index] = replace(
            layer, hue=hue, original_hue=layer.original_hue or layer.hue
        )
    return layers

def _fold_greys(clusters: list[_Cluster], *, grey_hue: int) -> list[_Cluster]:
    """Collapse every near-neutral cluster into a single layer.

    A near-neutral cluster has no hue worth the name, and the renderer's forced
    ``S=100% L=75%`` would turn whatever faint tint it has into a confident
    vivid colour -- so splitting several of them across attendees buys nothing
    but a handful of arbitrary pastels that look unrelated to the picture.
    They are merged into one layer which keeps the **largest** one's colour and
    gets ``grey_hue``; the merged layer is deliberately kept out of the
    hue-merging pass, since its hue is arbitrary and must not swallow a real
    colour that happens to sit near it.
    """
    greys = [c for c in clusters if c.is_grey]
    if len(greys) <= 1:
        return clusters
    merged = greys[0]
    for other in greys[1:]:
        merged = merged.merged_with(other)
    logger.info(
        "merged %d near-neutral cluster(s) into one layer at the arbitrary "
        "hue %d (a near-neutral has no hue, and the renderer cannot draw grey)",
        len(greys),
        grey_hue % 360,
    )
    out = [c for c in clusters if not c.is_grey]
    out.append(merged)
    out.sort(key=lambda c: c.share, reverse=True)
    return out


def _background_cluster(
    clusters: Sequence[_Cluster], *, ring_px: int, min_share: float
) -> int | None:
    """Index of the cluster that owns the image's border, if one does.

    Structural rather than colour-based, because "background" is not a colour:
    it is whatever lies behind everything, and what that means on a raster is
    *it reaches all the way around the edge of the picture*. Sampling the outer
    ring and asking which cluster owns most of it finds a lavender background
    as easily as a white one.

    Returns ``None`` when the border is all page (a logo on white -- there is
    no background cluster to begin with), when no single cluster owns
    ``min_share`` of the inked ring (a photo, where everything touches the
    edge, so dropping "the" background would be a guess), or when there is only
    one cluster (dropping it would leave nothing to draw).
    """
    if len(clusters) < 2:
        return None
    height, width = clusters[0].mask.shape
    depth = max(1, min(int(ring_px), (min(height, width) + 1) // 2))
    ring = np.zeros((height, width), dtype=bool)
    ring[:depth, :] = True
    ring[-depth:, :] = True
    ring[:, :depth] = True
    ring[:, -depth:] = True
    owned = [int((c.mask & ring).sum()) for c in clusters]
    total = sum(owned)
    if total == 0:
        logger.debug("no cluster owns the image border: nothing to call page")
        return None
    best = max(range(len(clusters)), key=lambda i: owned[i])
    share = owned[best] / total
    if share < min_share:
        logger.info(
            "no cluster owns more than %.0f%% of the image border (best is "
            "%.0f%%): keeping every layer rather than guessing at a background",
            min_share * 100,
            share * 100,
        )
        return None
    return best


def _spread_hues(hues: Sequence[int], anchor: int, minimum: float) -> list[int]:
    """Push crowded hues apart, keeping their order around the wheel.

    Pastel source art puts every cluster in a narrow arc of near-neutrals, and
    the renderer's forced ``S=100% L=75%`` then draws them as near-identical
    vivid colours -- the layers become indistinguishable, which is the one
    thing a colour decomposition must not be. This is the converse of
    :func:`_merge_near_hues`: where the gaps between neighbouring hues are too
    small, each gap is grown to at least ``minimum`` and the leftover circle is
    shared out **in proportion to the original gaps**, so a hue that sat
    between two others still does.

    ``anchor`` (an index into ``hues``) keeps its hue, so the most important
    layer's colour is the one that does not move. Returns new hues in the input
    order; when the hues are already far enough apart, the input is returned
    unchanged.
    """
    n = len(hues)
    if n < 2 or minimum <= 0:
        return list(hues)
    order = sorted(range(n), key=lambda i: (int(hues[i]) % 360, i))
    ring = [int(hues[i]) % 360 for i in order]
    # gaps[k] is the arc from ring[k] forward to its neighbour; these sum to
    # 360 for any n >= 2, including n == 2 (d and 360 - d).
    gaps = [float((ring[(k + 1) % n] - ring[k]) % 360) for k in range(n)]
    if min(gaps) >= minimum:
        return list(hues)
    slack = 360.0 - n * minimum
    if slack <= 0:
        logger.warning(
            "%d layers cannot be %.0f degrees apart on a 360-degree wheel; "
            "spacing them evenly instead",
            n,
            minimum,
        )
        new_gaps = [360.0 / n] * n
    else:
        total = sum(gaps) or 360.0
        new_gaps = [minimum + slack * (g / total) for g in gaps]
    position = order.index(anchor) if 0 <= anchor < n else 0
    spread = [0] * n
    spread[order[position]] = ring[position]
    current = float(ring[position])
    for step in range(1, n):
        current += new_gaps[(position + step - 1) % n]
        spread[order[(position + step) % n]] = int(round(current)) % 360
    logger.info(
        "spread %d crowded hue(s) apart to at least %.0f degrees: %s -> %s "
        "(the source colours cannot be reproduced anyway -- only the hue "
        "survives -- so distinguishable beats faithful)",
        n,
        minimum,
        list(hues),
        spread,
    )
    return spread


def _free_hue(taken: Sequence[int], preferred: int, tolerance: float) -> int:
    """``preferred``, or -- if that clashes with a real colour -- a free hue.

    The greyscale layer's hue is arbitrary, but it must not *duplicate* a
    chromatic layer's: two attendees drawing the same pastel is exactly the
    waste :func:`_merge_near_hues` exists to avoid. When ``preferred`` lands
    within ``tolerance`` of a layer that really is that colour, this returns
    the middle of the largest empty arc of the hue circle instead.
    """
    wanted = int(preferred) % 360
    if not taken or all(_hue_distance(wanted, t) > tolerance for t in taken):
        return wanted
    ring = sorted({int(t) % 360 for t in taken})
    best, widest = wanted, -1.0
    for low, high in zip(ring, [*ring[1:], ring[0] + 360], strict=True):
        gap = float(high - low)
        if gap > widest:
            widest, best = gap, int(round(low + gap / 2.0)) % 360
    logger.info(
        "the greyscale layer's preferred hue %d is within %.0f degrees of a "
        "real colour; moving it to %d (the emptiest part of the hue circle)",
        wanted,
        tolerance,
        best,
    )
    return best


def _merge_near_hues(
    clusters: Sequence[_Cluster], tolerance: float
) -> list[_Cluster]:
    """Merge clusters whose hues render as the same pastel.

    Walks the clusters largest-share first and merges each into the first
    accepted layer within ``tolerance`` degrees, keeping the larger layer's hue
    and sample colour so ``hue`` and ``sample_rgb`` always agree. The
    greyscale layer never merges (its hue is arbitrary).
    """
    if tolerance <= 0:
        return list(clusters)
    accepted: list[_Cluster] = []
    for cluster in sorted(clusters, key=lambda c: c.share, reverse=True):
        if cluster.is_grey:
            accepted.append(cluster)
            continue
        for i, kept in enumerate(accepted):
            if kept.is_grey or _hue_distance(kept.hue, cluster.hue) > tolerance:
                continue
            logger.info(
                "merging a hue-%d layer (%.1f%% of ink) into the hue-%d one: "
                "%.0f degrees apart renders as the same hsla(h, 100%%, 75%%) "
                "pastel and would cost a second attendee for nothing",
                cluster.hue,
                cluster.share * 100,
                kept.hue,
                _hue_distance(kept.hue, cluster.hue),
            )
            accepted[i] = kept.merged_with(cluster)
            break
        else:
            accepted.append(cluster)
    accepted.sort(key=lambda c: c.share, reverse=True)
    return accepted


def _split_budget(
    shares: Sequence[float], total: int, minimum: int
) -> list[int]:
    """Divide ``total`` segments among layers in proportion to their ink.

    ``max_segments`` is a budget for the **whole picture**, not per layer:
    every segment is one Chime message and the meeting shares ~150/s however
    many attendees are drawing. So each layer gets ``minimum`` segments as a
    floor plus a share-weighted slice of the rest (largest remainder, so the
    parts sum to exactly ``total``). ``shares`` must already be sorted largest
    first: when the budget cannot give every layer the floor, the smallest
    layers get **nothing** and are dropped, because a layer with two segments
    is a wasted attendee.
    """
    n = len(shares)
    if n == 0:
        return []
    if total <= 0:
        return [0] * n
    floor_each = max(1, min(int(minimum), total))
    keep = max(1, min(n, total // floor_each))
    if keep < n:
        logger.warning(
            "a budget of %d segment(s) cannot give %d layers %d each: keeping "
            "the %d largest",
            total,
            n,
            floor_each,
            keep,
        )
    allocation = [0] * n
    remaining = total - floor_each * keep
    weights = [max(float(shares[i]), 0.0) for i in range(keep)]
    weight_total = sum(weights) or 1.0
    exact = [w / weight_total * remaining for w in weights]
    extra = [int(math.floor(e)) for e in exact]
    for i in sorted(
        range(keep), key=lambda i: exact[i] - extra[i], reverse=True
    )[: remaining - sum(extra)]:
        extra[i] += 1
    for i in range(keep):
        allocation[i] = floor_each + extra[i]
    return allocation


def _trace_layer(
    loaded: _Loaded,
    mask: np.ndarray,
    *,
    fill: bool,
    budget: int,
    min_contour_length: float,
    simplify: float,
    fill_row_height: float | None,
    scale: float,
) -> _Traced | _Filled:
    """Trace one colour layer's mask, unplaced, within ``budget`` segments."""
    if fill:
        return _fill_segments(
            mask,
            max_segments=budget,
            min_run=min_contour_length,
            fill_row_height=fill_row_height,
            scale=scale,
        )
    return _contour_segments(
        loaded,
        mask,
        max_segments=budget,
        min_contour_length=min_contour_length,
        simplify=simplify,
    )


def _hit_budget(
    raw: _Traced | _Filled, budget: int, target_row: float
) -> bool:
    """Whether this layer would have used more segments if it had them."""
    if len(raw.segments) >= budget:
        return True
    if isinstance(raw, _Filled):
        return (
            raw.dropped > 0
            or raw.truncated
            or raw.row_height > target_row * 1.0001
        )
    return raw.contours_dropped > 0 or raw.truncated


def _layer_trace_result(
    raw: _Traced | _Filled,
    placed: list[Segment],
    loaded: _Loaded,
    *,
    rect: Rect,
    budget: int,
    scale: float,
) -> TraceResult:
    """Wrap one layer's unplaced trace plus its placed segments in a result."""
    if isinstance(raw, _Filled):
        return TraceResult(
            segments=placed,
            contours=raw.bands,
            source_size=loaded.source_size,
            mode="fill",
            traced_size=loaded.size,
            rect=rect,
            max_segments=budget,
            contours_found=raw.found,
            contours_filtered=0,
            contours_dropped=raw.dropped,
            simplify_epsilon=0.0,
            fill_row_height=raw.row_height * scale,
            truncated=raw.truncated,
        )
    return TraceResult(
        segments=placed,
        contours=raw.contours,
        source_size=loaded.source_size,
        mode="threshold",
        traced_size=loaded.size,
        rect=rect,
        max_segments=budget,
        contours_found=raw.contours_found,
        contours_filtered=raw.contours_filtered,
        contours_dropped=raw.contours_dropped,
        frames_dropped=raw.frames_dropped,
        simplify_epsilon=raw.epsilon,
        truncated=raw.truncated,
    )


#: Minimum run width to use when filling. In fill mode this parameter is the
#: narrowest band the tracer lays down, so the outline-oriented default (tuned
#: to suppress speckle) drops most runs and leaves a detailed picture as a few
#: sparse blobs. Measured on an illustration at a 4000-segment budget:
#: 120 segments and unrecognisable at 0.01, against 1094 and a clear picture
#: at 0.0015.
FILL_MIN_RUN = 0.0015

#: Band height for filling, in normalized units. One hairpin covers two bands,
#: and the pen is 7 px of the *displayed* video -- roughly 0.008 of the height
#: on a typical share -- so bands this tall overlap and read as solid rather
#: than striped. Smaller is denser and costs proportionally more messages.
FILL_ROW_HEIGHT = 0.004


def trace_options(
    *,
    fill: bool,
    mode: str = "auto",
    min_contour_length: float | None = None,
    fill_row_height: float | None = None,
    max_segments: int | None = None,
    invert: bool = False,
) -> dict[str, Any]:
    """Tracer keyword arguments for "filled" or "outlined", in one place.

    Both the command line and the interactive console need this decision, and
    having it in two places is how the console ended up still drawing hollow
    outlines after the command line was fixed. An explicit ``mode`` other than
    ``"auto"`` always wins.
    """
    resolved = mode if mode != "auto" else ("fill" if fill else "auto")
    options: dict[str, Any] = {"mode": resolved, "invert": invert}
    if max_segments is not None:
        options["max_segments"] = max_segments
    if fill_row_height is not None:
        options["fill_row_height"] = fill_row_height
    elif resolved == "fill":
        options["fill_row_height"] = FILL_ROW_HEIGHT
    if min_contour_length is not None:
        options["min_contour_length"] = min_contour_length
    elif resolved == "fill":
        options["min_contour_length"] = FILL_MIN_RUN
    return options


def trace_image_colors(
    path: ImageSource,
    rect: Rect | None = None,
    *,
    layers: int = DEFAULT_LAYERS,
    mode: TraceMode | str = "auto",
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    min_layer_share: float = DEFAULT_MIN_LAYER_SHARE,
    min_layer_feature: float = DEFAULT_MIN_LAYER_FEATURE,
    hue_merge_degrees: float = DEFAULT_HUE_MERGE_DEGREES,
    spread_hues: bool = True,
    min_hue_separation: float = DEFAULT_MIN_HUE_SEPARATION,
    min_chroma: float = DEFAULT_MIN_CHROMA,
    drop_background: bool = True,
    background_border_share: float = DEFAULT_BACKGROUND_BORDER_SHARE,
    background_level: float = DEFAULT_BACKGROUND_LEVEL,
    grey_hue: int | None = None,
    min_layer_segments: int = MIN_LAYER_SEGMENTS,
    blur: float = DEFAULT_BLUR,
    min_contour_length: float = DEFAULT_MIN_LAYER_CONTOUR,
    invert: bool = False,
    simplify: float = DEFAULT_SIMPLIFY,
    fill_row_height: float | None = None,
    frame_coverage: float = DEFAULT_FRAME_COVERAGE,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
    preserve_aspect: bool = True,
) -> list[ColorLayer]:
    """Decompose an image into colour layers, **one attendee per layer**.

    Slack gives every attendee exactly one line in one hue, so *N colours need
    N attendees* (``docs/huddles-peer-messages.md``, "Lines, colours and
    accounts"; one account can hold several attendees by joining with
    ``multidevice=true``, which is what makes this affordable). This
    quantises the image into at most ``layers`` colours, traces each colour's
    region separately, and returns one :class:`ColorLayer` per surviving
    colour, **largest ink share first** -- so a caller with fewer attendees than
    layers can take the first K and still get the colours that matter.

    Every layer is fitted into ``rect`` with the *same* transform, so the
    layers register with each other and can be drawn concurrently.

    Clusters are found with k-means in **OkLab**, not RGB: Euclidean distance
    in RGB does not group colours the way an eye does, which on its own makes a
    decomposition feel arbitrary.

    **Only the hue survives, and that is brutal.** ``draw.hue`` is degrees
    0..359 and the renderer always draws ``hsla(hue, 100%, 75%)`` with a fixed
    7 px round pen. So dark navy and pale sky blue map to the *same* pastel
    blue; nothing can be drawn dark, grey or white; and -- least obviously --
    **a barely-present tint is amplified into a vivid colour**: a 0.16-chroma
    pale cream renders as salmon. Four consequences are handled here rather
    than hidden:

    * **A layer's hue comes from its coloured pixels.** Not the cluster's mean
      colour, which washes toward grey and lands a rounding error away from a
      different hue, but the chroma-weighted circular mean of its pixels. A
      cluster of pale skin containing a saturated blush takes the blush's hue.
    * **Near-duplicate hues are merged.** Clusters whose hues are within
      ``hue_merge_degrees`` render identically, so they become one layer (the
      larger one's hue and ``sample_rgb`` are kept) instead of burning a second
      attendee on a colour nobody can distinguish.
    * **Crowded hues are spread apart** (``spread_hues``, on by default). When
      the surviving layers' hues are closer together than
      ``min_hue_separation`` -- the normal case for pastel art, where every
      cluster is a near-neutral -- they are pushed apart to at least that, in
      proportion to their original gaps, **preserving their order around the
      wheel** and keeping the largest layer's hue fixed. This *deliberately
      departs from the image's colours*: fidelity is impossible when
      saturation and lightness are not controllable, so the choice is between
      layers that are distinguishable and layers that are all the same vivid
      pastel. ``ColorLayer.original_hue`` records what was measured, and
      ``spread_hues=False`` keeps the measured hues.
    * **Near-neutrals get no confident colour.** Any cluster whose mean chroma
      is below ``min_chroma`` is near-neutral -- black, white, every grey, a
      dark colour whose tint is meaningless, and pale washes. **All** of them
      are folded into a single layer which keeps the largest one's
      ``sample_rgb``, is flagged ``is_greyscale``, and is drawn at the
      arbitrary hue ``grey_hue`` (moved out of the way if a real colour is
      already there). So a greyscale -- or uniformly pale -- image gives
      **one** layer, not ``layers`` vivid meaningless ones.

    :meth:`ColorLayer.describe` and :func:`describe_layers` print the source
    colour next to the colour that will actually be drawn, which is the fastest
    way to see what a decomposition is about to cost you.

    **Backgrounds, twice over.** Near-white unsaturated pixels are treated as
    the page, not as ink, because white is exactly what the renderer cannot
    draw; everything else is ink and ``pixel_share`` is a fraction of that ink.
    ``background_level`` moves that line (lower it to make mid-greys page too)
    and ``invert=True`` moves it to near-**black**, for art on a dark
    background. But a *coloured* background is still ink, and it is the worst
    thing that can happen to a trace: its only boundary is the edge of the
    image, so it draws as one enormous box around the whole share and eats the
    budget. Two independent defences, both on by default:

    * ``drop_background`` discards the cluster that owns most of the image's
      **border ring** -- structural, so it catches a lavender background as
      easily as a white one. It is skipped when the border is all page, when no
      cluster owns ``background_border_share`` of it (a photo: everything
      touches the edge, so there is no obvious background to drop), or when
      dropping would leave nothing. The surviving layers are then budgeted
      against the **whole** of ``max_segments``, not the share they had while
      the background was still in the running.
    * ``frame_coverage`` rejects any contour that is the image frame, in every
      mode including :func:`trace_image`'s.

    A uniform image, and an image that is entirely page, both give **zero**
    layers.

    **The budget.** ``max_segments`` is a ceiling on the sum over *all* layers,
    not per layer, because one segment is one Chime message and the whole
    meeting shares ~150/s however many attendees draw. It is divided in
    proportion to each layer's ink, with ``min_layer_segments`` as a floor; if
    the budget cannot afford that floor for every layer, the smallest layers
    are dropped rather than given a useless two segments. Any budget a layer
    cannot spend (a simple shape needs few) is handed back to the layers that
    hit their limit, once. Within a layer, degradation is exactly
    :func:`trace_image`'s: raise the epsilon for the whole layer, then drop its
    shortest contours, then -- only for absurd budgets -- slice. Per-layer
    detail is in :attr:`ColorLayer.trace`; :func:`describe_layers` prints it.

    Args:
        path: image file or open :class:`PIL.Image.Image`, as
            :func:`trace_image` (EXIF honoured, alpha composited onto white,
            downscaled to ``max_dimension``).
        rect: where to put the drawing, in normalized 0..1 share coordinates.
            Every returned coordinate of every layer is inside it.
        layers: an **upper bound** on the number of colours, i.e. on the
            attendees needed. Fewer come back when clusters merge, fall below
            ``min_layer_share``, trace to nothing or cannot be afforded.
        mode: ``"fill"`` fills each colour region with hairpins; ``"outline"``,
            ``"edges"``, ``"threshold"`` and ``"auto"`` all trace the region's
            boundary. There is no luminance threshold and no Sobel step here --
            the colour cluster *is* the region -- so unlike
            :func:`trace_image` the only real choice is contours vs fill.
        max_segments: hard total budget across all layers.
        min_layer_share: drop layers covering less of the ink than this.
        min_layer_feature: drop layers whose biggest contour is shorter than
            this (perimeter, relative to the traced area's longer side) --
            speckle, however much of it there is. ``0`` keeps them.
        hue_merge_degrees: merge layers whose hues are closer than this.
        spread_hues: push crowded hues apart so the layers stay
            distinguishable, at the cost of their absolute values.
        min_hue_separation: how far apart ``spread_hues`` pushes them.
        min_chroma: below this **mean chroma** a cluster is near-neutral and
            joins the single greyscale layer instead of getting a hue of its
            own. Raise it for pastel art (every faint tint renders vivid),
            lower it to let faint tints keep their own attendee.
        drop_background: discard the cluster that owns the image's border.
        background_border_share: how much of the border ring a cluster has to
            own to count as the background.
        background_level: luminance fraction at which an unsaturated pixel
            becomes page rather than ink.
        grey_hue: the arbitrary hue for the near-neutral layer.
        min_layer_segments: floor on a layer's budget; layers that cannot be
            given this many are dropped.
        min_contour_length: contours shorter than this are noise. The default
            is higher than :func:`trace_image`'s, because a colour layer costs
            an attendee. Note that in ``"fill"`` mode this is the minimum run
            *width* instead, where the colour default is aggressive -- pass
            :data:`DEFAULT_MIN_CONTOUR_LENGTH` to fill narrow shapes.
        blur, invert, simplify, fill_row_height, frame_coverage, max_dimension,
            preserve_aspect: as :func:`trace_image`.

    Returns:
        Layers sorted by ``pixel_share`` descending, each with at least one
        segment. ``pixel_share`` is never renormalized, so the shares sum to
        1.0 only when nothing was dropped.

    **What decomposes badly.** Flat-colour art (flags, logos, icons, charts)
    decomposes well. Photographs do not: k-means on a photo's colours gives
    layers whose masks are confetti, and each layer then spends its whole
    budget on fragments. Soft gradients get banded into arbitrary stripes whose
    boundaries are not in the picture. Adjacent flat regions share a border and
    each layer traces its own side of it, so every internal edge is drawn twice
    in two hues (which looks fine, but costs twice). And any region whose
    colour is near-white is simply not drawable, so it reads as a hole.
    """
    chosen = "outline" if mode == "edges" else str(mode)
    if chosen not in ("outline", "threshold", "fill", "auto"):
        raise ValueError(
            f"unknown mode {mode!r}; expected outline/edges, threshold, fill "
            "or auto"
        )
    if layers < 1:
        raise ValueError("layers must be >= 1")
    if max_segments < 0:
        raise ValueError("max_segments must be >= 0")
    if simplify < 0:
        raise ValueError("simplify must be >= 0")
    if min_contour_length < 0:
        raise ValueError("min_contour_length must be >= 0")
    if max_dimension < 1:
        raise ValueError("max_dimension must be >= 1")
    if not 0.0 <= min_layer_share <= 1.0:
        raise ValueError("min_layer_share must be a fraction in 0..1")
    if min_layer_feature < 0:
        raise ValueError("min_layer_feature must be >= 0")
    if hue_merge_degrees < 0:
        raise ValueError("hue_merge_degrees must be >= 0")
    if min_hue_separation < 0:
        raise ValueError("min_hue_separation must be >= 0")
    if not 0.0 <= min_chroma <= 1.0:
        raise ValueError("min_chroma must be a fraction in 0..1")
    if not 0.0 <= background_border_share <= 1.0:
        raise ValueError("background_border_share must be a fraction in 0..1")
    if not 0.0 <= background_level <= 1.0:
        raise ValueError("background_level must be a fraction in 0..1")
    if min_layer_segments < 1:
        raise ValueError("min_layer_segments must be >= 1")

    rgb, loaded = _load_rgb(path, max_dimension=max_dimension, blur=blur)
    destination = _destination(rect)
    clusters = _merge_near_hues(
        _color_clusters(
            rgb,
            loaded,
            layers=layers,
            background_level=background_level,
            min_chroma=min_chroma,
            invert=invert,
            grey_hue=DEFAULT_GREY_HUE if grey_hue is None else grey_hue,
        ),
        hue_merge_degrees,
    )
    if not clusters:
        logger.info("colour trace: nothing drawable in this image")
        return []

    # The background first, so everything after it -- the share filter and
    # above all the budget split -- sees only layers that will be drawn.
    background = _background_cluster(
        clusters,
        ring_px=BORDER_RING_PX,
        min_share=background_border_share,
    )
    if background is not None and not drop_background:
        clusters = [
            c.as_background() if i == background else c
            for i, c in enumerate(clusters)
        ]
    elif background is not None:
        marked = clusters[background]
        logger.info(
            "dropping the background layer: hue %d, rgb%s, %.1f%% of the ink, "
            "owns the image border. Its only boundary is the edge of the "
            "image, so drawing it would spend the budget on a box around the "
            "whole share (pass drop_background=False to keep it)",
            marked.hue,
            marked.rgb,
            marked.share * 100,
        )
        clusters = [c for i, c in enumerate(clusters) if i != background]

    kept = [c for c in clusters if c.share >= min_layer_share]
    if len(kept) != len(clusters):
        logger.info(
            "dropped %d colour layer(s) below %.1f%% of the ink (not worth a "
            "whole attendee): %s",
            len(clusters) - len(kept),
            min_layer_share * 100,
            ", ".join(
                f"hue {c.hue} at {c.share * 100:.2f}%"
                for c in clusters
                if c.share < min_layer_share
            ),
        )
    if not kept:
        logger.warning(
            "every colour layer was below min_layer_share=%.3f", min_layer_share
        )
        return []

    fill = chosen == "fill"
    scale = _vertical_scale(loaded, destination, preserve_aspect)
    target_row = float(fill_row_height or DEFAULT_FILL_ROW_HEIGHT) / scale

    # Contours are budget-independent, so they can be traced once, up front,
    # and used to decide which layers are worth an attendee at all. A fill
    # needs them only for that decision, so it skips them when it is off.
    prepared = [
        _prepare_contours(
            loaded,
            cluster.mask,
            min_contour_length=min_contour_length,
            frame_coverage=frame_coverage,
        )
        if not fill or min_layer_feature > 0
        else _Contours(contours=[], found=0, filtered=0, frames=0, longest=0.0)
        for cluster in kept
    ]
    if min_layer_feature > 0:
        solid = [
            i
            for i, p in enumerate(prepared)
            if p.longest >= min_layer_feature
        ]
        if not solid:
            logger.warning(
                "every colour layer is speckle (no contour longer than %.2f); "
                "keeping the largest rather than returning nothing",
                min_layer_feature,
            )
            solid = [max(range(len(prepared)), key=lambda i: prepared[i].longest)]
        if len(solid) != len(kept):
            logger.info(
                "dropped %d speckle layer(s) whose biggest feature was under "
                "%.2f: a layer made of dots is worth neither an attendee nor "
                "the messages (%s)",
                len(kept) - len(solid),
                min_layer_feature,
                ", ".join(
                    f"hue {kept[i].hue} at {kept[i].share * 100:.1f}% of ink, "
                    f"biggest contour {prepared[i].longest:.3f}"
                    for i in range(len(kept))
                    if i not in set(solid)
                ),
            )
            kept = [kept[i] for i in solid]
            prepared = [prepared[i] for i in solid]

    # Hues last, so spreading only has to separate layers that will be drawn.
    if spread_hues:
        chromatic = [i for i, c in enumerate(kept) if not c.is_grey]
        if len(chromatic) >= 2:
            anchor = max(
                range(len(chromatic)), key=lambda k: kept[chromatic[k]].share
            )
            moved = _spread_hues(
                [kept[i].hue for i in chromatic], anchor, min_hue_separation
            )
            for position, i in enumerate(chromatic):
                kept[i] = kept[i].with_hue(moved[position], record=True)
    taken = [c.hue for c in kept if not c.is_grey]
    kept = [
        c.with_hue(
            _free_hue(
                taken,
                c.hue,
                max(hue_merge_degrees, min_hue_separation if spread_hues else 0.0),
            )
        )
        if c.is_grey
        else c
        for c in kept
    ]

    shares = [c.share for c in kept]
    allocation = _split_budget(shares, max_segments, min_layer_segments)

    def trace(index: int) -> _Traced | _Filled:
        if fill:
            return _fill_segments(
                kept[index].mask,
                max_segments=allocation[index],
                min_run=min_contour_length,
                fill_row_height=fill_row_height,
                scale=scale,
            )
        return _fit_prepared(
            loaded,
            prepared[index],
            max_segments=allocation[index],
            simplify=simplify,
        )

    raws = [trace(i) for i in range(len(kept))]

    # Hand unspent budget to the layers that ran out of it. Only the slack of
    # layers that did *not* hit their limit is redistributed, so the total
    # still cannot exceed max_segments.
    hungry = [
        i
        for i, raw in enumerate(raws)
        if allocation[i] > 0 and _hit_budget(raw, allocation[i], target_row)
    ]
    free = sum(
        allocation[i] - len(raws[i].segments)
        for i in range(len(raws))
        if i not in set(hungry)
    )
    if free > 0 and hungry:
        logger.info(
            "redistributing %d unspent segment(s) to %d layer(s) that hit "
            "their budget",
            free,
            len(hungry),
        )
        extra = _split_budget([shares[i] for i in hungry], free, 1)
        for position, i in enumerate(hungry):
            if extra[position] > 0:
                allocation[i] += extra[position]
                raws[i] = trace(i)

    # One shared transform for every layer, or the colours would not register.
    source = _source_rect(loaded, [s for raw in raws for s in raw.segments])
    out: list[ColorLayer] = []
    for cluster, raw, budget in zip(kept, raws, allocation, strict=True):
        placed = _place(
            raw.segments,
            loaded,
            destination,
            preserve_aspect,
            source=source,
        )
        if not placed:
            logger.info(
                "colour layer hue=%d (%.1f%% of ink) traced to nothing at a "
                "budget of %d segment(s)",
                cluster.hue,
                cluster.share * 100,
                budget,
            )
            continue
        out.append(
            ColorLayer(
                hue=cluster.hue,
                segments=placed,
                pixel_share=cluster.share,
                sample_rgb=cluster.rgb,
                trace=_layer_trace_result(
                    raw,
                    placed,
                    loaded,
                    rect=destination,
                    budget=budget,
                    scale=scale,
                ),
                is_greyscale=cluster.is_grey,
                merged_clusters=cluster.merged,
                is_background=cluster.is_background,
                original_hue=cluster.original_hue,
                chroma=cluster.chroma,
            )
        )

    out = _reposition_neutral_layers(out, explicit_grey_hue=grey_hue is not None)
    out.sort(key=lambda layer: layer.pixel_share, reverse=True)
    if out:
        logger.info("traced image in colour:\n%s", describe_layers(out))
    return out


# --------------------------------------------------------------------------
# preview
# --------------------------------------------------------------------------


def preview_ascii(
    result: TraceResult | Sequence[Segment],
    width: int = 64,
    *,
    ink: str = "#",
    empty: str = ".",
) -> str:
    """Render a trace as ASCII so it can be sanity-checked in a terminal.

    The grid spans the **whole visible 0..1 area**, not the trace's bounding
    box, so it shows placement inside the share as well as shape. Rows are
    ``width // 2`` because terminal cells are about twice as tall as they are
    wide; the result is ``width`` characters per line, newline-separated, with
    no trailing newline.

    This rasterizes the Béziers, so it shows what the *curves* do -- including
    the corner rounding Catmull-Rom introduces. It does not model the 7 px pen,
    so a preview looks thinner than the real drawing.
    """
    if width < 2:
        raise ValueError("width must be >= 2")
    segments = (
        result.segments if isinstance(result, TraceResult) else list(result)
    )
    rows = max(1, width // 2)
    grid = [[empty] * width for _ in range(rows)]

    def plot(point: Point) -> None:
        col = min(width - 1, max(0, int(geometry.clamp01(point.x) * width)))
        row = min(rows - 1, max(0, int(geometry.clamp01(point.y) * rows)))
        grid[row][col] = ink

    for segment in segments:
        if segment.is_degenerate:
            plot(segment.a)
            continue
        hull = sum(
            a.distance_to(b)
            for a, b in zip(segment.points, segment.points[1:], strict=False)
        )
        steps = max(2, min(512, int(hull * max(width, rows) * 2) + 2))
        for i in range(steps + 1):
            plot(geometry.bezier_point(segment, i / steps))
    return "\n".join("".join(row) for row in grid)

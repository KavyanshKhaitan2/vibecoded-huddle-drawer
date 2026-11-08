"""Pure geometry for huddle drawings: curves, simplification, fills, shapes.

No I/O, no asyncio, no protobuf import at module level -- this module is
importable on its own, and the protobuf conversion helpers
(:func:`to_proto_segment`, :func:`to_proto_segments`) import the generated
module lazily.

**Coordinates are normalized 0..1 over the shared video's bounding rect**, which
is exactly what Slack's ``ProgressiveCurveTransformer.add`` produces:
``[(x - left) / width, (y - top) / height]``. ``y`` grows *downward*, as on
screen. Values outside 0..1 are legal on the wire (``-5, -5`` is the documented
way to park a cursor off-screen) but will not be visible.

Why this module exists: Chime delivers only ~150 data messages/s per *meeting*,
shared by every sender, and one message carries one cubic Bézier. The doc's
"Ways to get more picture per message" names the two tricks implemented here:

* **Curves.** A segment is a full cubic Bézier, so an outline traced as smooth
  curves (contour -> :func:`rdp` -> :func:`catmull_rom_to_beziers`) needs a
  small fraction of the messages that straight hatching does.
* **Hairpins.** :func:`hairpin_segment` fills a band two pens tall with one
  message -- half the messages of plain hatching.

The pen is fixed at 7 px, round caps, and the stroke width is not controllable,
so there is no point modelling anything finer than about a pen width: see
:func:`pen_size_normalized` to convert 7 px into this coordinate space.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .proto import peer_message_pb2 as pb

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PEN_PX",
    "HAIRPIN_OVERSHOOT",
    "TEXT_GLYPHS",
    "Point",
    "Rect",
    "Segment",
    "bezier_point",
    "catmull_rom_to_beziers",
    "circle_segments",
    "clamp01",
    "ellipse_segments",
    "fit_points_into_rect",
    "fit_segments_into_rect",
    "from_proto_segment",
    "hairpin_fill",
    "hairpin_segment",
    "line_segments",
    "pen_size_normalized",
    "polyline_segments",
    "rdp",
    "rect_segments",
    "smooth_path",
    "text_coverage",
    "text_segments",
    "to_proto_segment",
    "to_proto_segments",
    "unit_square",
]

#: The renderer's fixed pen: 7 px, round caps, width not controllable.
DEFAULT_PEN_PX = 7.0

#: A cubic with x-controls ``(x0, x1', x1', x0)`` reaches only
#: ``x0 + 3/4 * (x1' - x0)``, so to make a hairpin's tip land on ``x1`` the
#: control points go 4/3 of the way out. See :func:`hairpin_segment`.
HAIRPIN_OVERSHOOT = 4.0 / 3.0


def clamp01(value: float) -> float:
    """Clamp a single coordinate into the visible 0..1 range."""
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else float(value)


@dataclass(frozen=True, slots=True)
class Point:
    """A point in normalized 0..1 video coordinates (``y`` grows downward)."""

    x: float
    y: float

    def __iter__(self):  # so ``x, y = point`` and ``tuple(point)`` work
        yield self.x
        yield self.y

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, k: float) -> Point:
        return Point(self.x * k, self.y * k)

    __rmul__ = __mul__

    def clamped(self) -> Point:
        """This point, clamped into 0..1."""
        return Point(clamp01(self.x), clamp01(self.y))

    def distance_to(self, other: Point) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    @classmethod
    def of(cls, value: PointLike) -> Point:
        """Coerce a ``Point``, a protobuf ``Point`` or an ``(x, y)`` pair."""
        if isinstance(value, Point):
            return value
        if hasattr(value, "x") and hasattr(value, "y"):
            return cls(float(value.x), float(value.y))  # type: ignore[attr-defined]
        px, py = value  # type: ignore[misc]
        return cls(float(px), float(py))


PointLike: TypeAlias = "Point | tuple[float, float] | Sequence[float] | object"


@dataclass(frozen=True, slots=True)
class Segment:
    """One cubic Bézier: ``a`` start, ``b``/``c`` controls, ``d`` end.

    This is exactly ``PeerMessage.Segment``, and one of these is the payload of
    one ``draw`` message. The renderer does its own ``moveTo(a)`` for every
    segment, so consecutive segments of a line do **not** have to connect.
    """

    a: Point
    b: Point
    c: Point
    d: Point

    def __iter__(self):
        yield from (self.a, self.b, self.c, self.d)

    @property
    def points(self) -> tuple[Point, Point, Point, Point]:
        return (self.a, self.b, self.c, self.d)

    @classmethod
    def degenerate(cls, point: PointLike) -> Segment:
        """The ``a=b=c=d`` segment a stroke's first ``NEW`` message carries.

        The doc: "first point of a stroke: ``NEW`` with a degenerate segment
        (``a=b=c=d``)". With the 7 px round pen this renders as a single dot.
        """
        p = Point.of(point)
        return cls(p, p, p, p)

    @classmethod
    def line(cls, start: PointLike, end: PointLike) -> Segment:
        """A straight cubic from ``start`` to ``end`` (controls at 1/3 and 2/3)."""
        p0, p1 = Point.of(start), Point.of(end)
        delta = p1 - p0
        return cls(p0, p0 + delta * (1 / 3), p0 + delta * (2 / 3), p1)

    @property
    def is_degenerate(self) -> bool:
        return all(p == self.a for p in self.points)

    def reversed(self) -> Segment:
        """The same curve traced the other way."""
        return Segment(self.d, self.c, self.b, self.a)

    def clamped(self) -> Segment:
        """Every control point clamped into 0..1.

        Note this changes the curve's shape, it does not clip it; it is a
        cheap guard against sending coordinates that cannot be seen.
        """
        return Segment(*(p.clamped() for p in self.points))

    def translated(self, dx: float, dy: float) -> Segment:
        offset = Point(dx, dy)
        return Segment(*(p + offset for p in self.points))

    def scaled(self, sx: float, sy: float | None = None) -> Segment:
        sy = sx if sy is None else sy
        return Segment(*(Point(p.x * sx, p.y * sy) for p in self.points))


@dataclass(frozen=True, slots=True)
class Rect:
    """An axis-aligned rectangle in normalized coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> Point:
        return Point((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def normalized(self) -> Rect:
        """This rect with ``x0 <= x1`` and ``y0 <= y1``."""
        return Rect(
            min(self.x0, self.x1),
            min(self.y0, self.y1),
            max(self.x0, self.x1),
            max(self.y0, self.y1),
        )

    def contains(self, point: PointLike) -> bool:
        p = Point.of(point)
        r = self.normalized()
        return r.x0 <= p.x <= r.x1 and r.y0 <= p.y <= r.y1

    @classmethod
    def bounding(cls, points: Iterable[PointLike]) -> Rect:
        pts = [Point.of(p) for p in points]
        if not pts:
            raise ValueError("no points to bound")
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        return cls(min(xs), min(ys), max(xs), max(ys))


def unit_square() -> Rect:
    """The whole visible area: ``Rect(0, 0, 1, 1)``."""
    return Rect(0.0, 0.0, 1.0, 1.0)


def pen_size_normalized(
    video_width_px: float, video_height_px: float, pen_px: float = DEFAULT_PEN_PX
) -> tuple[float, float]:
    """The 7 px pen expressed in normalized units for a given video size.

    Use the height component as the row pitch for :func:`hairpin_fill`, and as
    the floor on any detail worth drawing: the stroke width is fixed at 7 px
    with round caps, so nothing finer survives rendering.
    """
    if video_width_px <= 0 or video_height_px <= 0:
        raise ValueError("video size must be positive")
    return pen_px / video_width_px, pen_px / video_height_px


# --------------------------------------------------------------------------
# protobuf conversion (lazy import, so geometry stands alone)
# --------------------------------------------------------------------------


def to_proto_segment(segment: Segment) -> pb.PeerMessage.Segment:
    """Convert to the protobuf ``Segment`` a ``draw`` message carries."""
    from .proto import peer_message_pb2 as pb_mod

    out = pb_mod.PeerMessage.Segment()
    for name, point in zip("abcd", segment.points, strict=True):
        target = getattr(out, name)
        target.x = float(point.x)
        target.y = float(point.y)
    return out


def to_proto_segments(segments: Iterable[Segment]) -> list[pb.PeerMessage.Segment]:
    """Convert many segments; one protobuf ``Segment`` per ``draw`` message."""
    return [to_proto_segment(s) for s in segments]


def from_proto_segment(segment: object) -> Segment:
    """Convert a protobuf ``Segment`` (or anything a/b/c/d-shaped) back."""
    return Segment(*(Point.of(getattr(segment, name)) for name in "abcd"))


# --------------------------------------------------------------------------
# curves
# --------------------------------------------------------------------------


def bezier_point(segment: Segment, t: float) -> Point:
    """Evaluate the cubic at ``t`` in 0..1. ``t=0`` is ``a``, ``t=1`` is ``d``."""
    u = 1.0 - t
    w0, w1, w2, w3 = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
    a, b, c, d = segment.points
    return Point(
        w0 * a.x + w1 * b.x + w2 * c.x + w3 * d.x,
        w0 * a.y + w1 * b.y + w2 * c.y + w3 * d.y,
    )


def _dedupe(points: Sequence[PointLike], *, eps: float = 1e-12) -> list[Point]:
    out: list[Point] = []
    for raw in points:
        p = Point.of(raw)
        if not out or out[-1].distance_to(p) > eps:
            out.append(p)
    return out


def catmull_rom_to_beziers(
    points: Sequence[PointLike],
    *,
    closed: bool = False,
    tension: float = 1.0,
) -> list[Segment]:
    """Fit a Catmull-Rom spline through ``points`` as cubic Bézier segments.

    One segment per gap between consecutive input points, with ``a`` and ``d``
    set to those two points -- so the curve **passes through every input
    point** exactly, and the control points are derived from the neighbours:
    ``b = p1 + (p2 - p0) * tension / 6``, ``c = p2 - (p3 - p1) * tension / 6``.

    For an open path the ends are extrapolated by reflection; with
    ``closed=True`` the neighbours wrap and a closing segment is added.
    ``tension=0`` degenerates to straight lines.

    This is the message-cheap way to draw: feed it a simplified contour (see
    :func:`rdp`) and each message buys a whole curve instead of a short
    straight dash.
    """
    pts = _dedupe(points)
    if len(pts) < 2:
        return []
    if len(pts) == 2 and not closed:
        return [Segment.line(pts[0], pts[1])]

    k = tension / 6.0
    out: list[Segment] = []
    n = len(pts)
    last = n if closed else n - 1
    for i in range(last):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        if closed:
            p0 = pts[(i - 1) % n]
            p3 = pts[(i + 2) % n]
        else:
            p0 = pts[i - 1] if i > 0 else pts[0] + (pts[0] - pts[1])
            p3 = pts[i + 2] if i + 2 < n else pts[-1] + (pts[-1] - pts[-2])
        out.append(Segment(p1, p1 + (p2 - p0) * k, p2 - (p3 - p1) * k, p2))
    return out


def _perpendicular_distance(point: Point, start: Point, end: Point) -> float:
    dx, dy = end.x - start.x, end.y - start.y
    if dx == 0.0 and dy == 0.0:
        return point.distance_to(start)
    # |cross product| / |segment|
    return abs(dy * (point.x - start.x) - dx * (point.y - start.y)) / math.hypot(dx, dy)


def rdp(points: Sequence[PointLike], epsilon: float) -> list[Point]:
    """Ramer-Douglas-Peucker simplification.

    Drops every point that is within ``epsilon`` (perpendicular distance, in
    normalized units) of the polyline kept around it. Endpoints are always
    kept. ``epsilon`` around one pen height (:func:`pen_size_normalized`) throws
    away detail the 7 px pen could not show anyway, and each point dropped is
    one fewer ``draw`` message against the ~150/s meeting budget.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    pts = [Point.of(p) for p in points]
    if len(pts) < 3:
        return pts

    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack: list[tuple[int, int]] = [(0, len(pts) - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        worst = -1.0
        index = first
        for i in range(first + 1, last):
            dist = _perpendicular_distance(pts[i], pts[first], pts[last])
            if dist > worst:
                worst, index = dist, i
        if worst > epsilon:
            keep[index] = True
            stack.append((first, index))
            stack.append((index, last))
    return [p for p, k in zip(pts, keep, strict=True) if k]


def smooth_path(
    points: Sequence[PointLike],
    *,
    epsilon: float = 0.004,
    closed: bool = False,
    tension: float = 1.0,
) -> list[Segment]:
    """The whole message-saving pipeline: simplify, then fit curves.

    ``contour -> rdp -> catmull_rom_to_beziers``, which is the order the doc
    recommends ("contours -> simplification -> Catmull-Rom"). The default
    ``epsilon`` is roughly a pen width on a 1080p share.
    """
    return catmull_rom_to_beziers(
        rdp(points, epsilon), closed=closed, tension=tension
    )


# --------------------------------------------------------------------------
# the hairpin fill
# --------------------------------------------------------------------------


def hairpin_segment(
    x0: float,
    x1: float,
    y: float,
    row_height: float,
    *,
    overshoot: float = HAIRPIN_OVERSHOOT,
) -> Segment:
    """One hairpin: out along row ``y`` and back along row ``y + row_height``.

    Verbatim from the doc's "Ways to get more picture per message":

        **Hairpins.** For filled areas, one Bézier ``a=(x0,y),
        b=c=(x1',y)…(x1',y+1), d=(x0,y+1)`` runs out along one row and back
        along the next, filling a band two pens tall per message: half the
        messages of plain hatching.

    So the control points sit at the far end ``x1'``: ``b = (x1', y)`` on the
    outbound row and ``c = (x1', y + row_height)`` on the return row, and the
    curve ends back under where it started. ``y + 1`` in the doc means the next
    pen row, one ``row_height`` down -- not ``y + 1.0``, which would be off the
    bottom of a 0..1 space.

    ``x1'`` is an **overshoot** of the requested ``x1``: a cubic whose x
    controls are ``(x0, x1', x1', x0)`` only reaches ``x0 + 3/4 (x1' - x0)``, so
    the controls go 4/3 of the way out (:data:`HAIRPIN_OVERSHOOT`) to make the
    tip of the hairpin land on ``x1``. Pass ``overshoot=1.0`` for the literal
    control points with a short tip.

    ``row_height`` should be about one pen height
    (:func:`pen_size_normalized`): the two rows then touch and the band reads
    as solid fill.
    """
    tip = x0 + (x1 - x0) * overshoot
    return Segment(
        Point(x0, y),
        Point(tip, y),
        Point(tip, y + row_height),
        Point(x0, y + row_height),
    )


def hairpin_fill(
    rect: Rect,
    row_height: float,
    *,
    overshoot: float = HAIRPIN_OVERSHOOT,
    serpentine: bool = True,
) -> list[Segment]:
    """Fill ``rect`` with hairpins: one message per band two pens tall.

    Walks down the rect in steps of ``2 * row_height``, emitting one
    :func:`hairpin_segment` per band. The last band is shortened so it does not
    run past ``rect.y1`` (the 7 px round pen still spills about half a pen in
    every direction -- there is no way to stop that).

    With ``serpentine=True`` alternate bands start from the right, which keeps
    the pencil cursor from jumping the full width between messages. It makes no
    difference to the picture: the renderer does its own ``moveTo`` per segment,
    so segments need not connect.
    """
    if row_height <= 0:
        raise ValueError("row_height must be positive")
    r = rect.normalized()
    out: list[Segment] = []
    y = r.y0
    flip = False
    while y < r.y1:
        dy = min(row_height, max(r.y1 - y, 0.0))
        if dy <= 0:
            break
        left, right = (r.x1, r.x0) if (serpentine and flip) else (r.x0, r.x1)
        out.append(hairpin_segment(left, right, y, dy, overshoot=overshoot))
        y += 2 * row_height
        flip = not flip
    return out


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------


def line_segments(start: PointLike, end: PointLike) -> list[Segment]:
    """A straight line as one segment (one ``draw`` message)."""
    return [Segment.line(start, end)]


def polyline_segments(points: Sequence[PointLike], *, closed: bool = False) -> list[Segment]:
    """A polyline as one straight segment per edge.

    This is the expensive way to draw -- one message per edge. Prefer
    :func:`smooth_path` when the shape is a curve.
    """
    pts = _dedupe(points)
    if len(pts) < 2:
        return []
    edges = list(zip(pts, pts[1:], strict=False))
    if closed:
        edges.append((pts[-1], pts[0]))
    return [Segment.line(p, q) for p, q in edges]


def rect_segments(
    rect: Rect,
    *,
    fill_row_height: float | None = None,
    overshoot: float = HAIRPIN_OVERSHOOT,
) -> list[Segment]:
    """A rectangle: 4 segments as an outline, or hairpins when filled.

    Pass ``fill_row_height`` (about one pen height) to fill it with
    :func:`hairpin_fill` instead of outlining it.
    """
    r = rect.normalized()
    if fill_row_height is not None:
        return hairpin_fill(r, fill_row_height, overshoot=overshoot)
    corners = [
        Point(r.x0, r.y0),
        Point(r.x1, r.y0),
        Point(r.x1, r.y1),
        Point(r.x0, r.y1),
    ]
    return polyline_segments(corners, closed=True)


#: Bézier circle constant: 4/3 * (sqrt(2) - 1).
_KAPPA = 4.0 / 3.0 * (math.sqrt(2.0) - 1.0)


def ellipse_segments(
    center: PointLike, rx: float, ry: float, *, quarters: int = 4
) -> list[Segment]:
    """An ellipse as cubic Béziers -- 4 messages for a whole circle.

    Uses the standard kappa approximation per quarter (max radial error ~0.027%,
    far below one 7 px pen). ``quarters`` may be 1..4 to draw only part of it,
    starting at the rightmost point and going clockwise in screen coordinates
    (``y`` down).
    """
    if not 1 <= quarters <= 4:
        raise ValueError("quarters must be 1..4")
    c = Point.of(center)
    kx, ky = _KAPPA * rx, _KAPPA * ry
    # Right, bottom, left, top in screen coordinates.
    anchors = [
        Point(c.x + rx, c.y),
        Point(c.x, c.y + ry),
        Point(c.x - rx, c.y),
        Point(c.x, c.y - ry),
    ]
    controls = [
        (Point(c.x + rx, c.y + ky), Point(c.x + kx, c.y + ry)),
        (Point(c.x - kx, c.y + ry), Point(c.x - rx, c.y + ky)),
        (Point(c.x - rx, c.y - ky), Point(c.x - kx, c.y - ry)),
        (Point(c.x + kx, c.y - ry), Point(c.x + rx, c.y - ky)),
    ]
    out: list[Segment] = []
    for i in range(quarters):
        b, cc = controls[i]
        out.append(Segment(anchors[i], b, cc, anchors[(i + 1) % 4]))
    return out


def circle_segments(center: PointLike, radius: float, *, quarters: int = 4) -> list[Segment]:
    """A circle as 4 cubic Béziers.

    Beware: the coordinate space is normalized per axis, so a circle in these
    coordinates is only round on screen if the share is square. Scale ``rx``
    by the video's aspect ratio with :func:`ellipse_segments` if roundness
    matters.
    """
    return ellipse_segments(center, radius, radius, quarters=quarters)


# --------------------------------------------------------------------------
# a minimal stroked vector font
# --------------------------------------------------------------------------

# Glyphs are polylines in a unit cell: x and y both run 0..1, y downward,
# origin at the cell's top-left. Deliberately crude -- the pen is 7 px with
# round caps, so hairlines and serifs would not survive anyway.
TEXT_GLYPHS: dict[str, tuple[tuple[tuple[float, float], ...], ...]] = {
    " ": (),
    "A": (((0, 1), (0.5, 0), (1, 1)), ((0.25, 0.55), (0.75, 0.55))),
    "B": (
        ((0, 0), (0, 1)),
        ((0, 0), (0.8, 0.12), (0.8, 0.38), (0, 0.5)),
        ((0, 0.5), (0.9, 0.64), (0.9, 0.88), (0, 1)),
    ),
    "C": (((1, 0.2), (0.5, 0), (0, 0.3), (0, 0.7), (0.5, 1), (1, 0.8)),),
    "D": (((0, 0), (0, 1)), ((0, 0), (0.75, 0.25), (0.75, 0.75), (0, 1))),
    "E": (((1, 0), (0, 0), (0, 1), (1, 1)), ((0, 0.5), (0.7, 0.5))),
    "F": (((1, 0), (0, 0), (0, 1)), ((0, 0.5), (0.7, 0.5))),
    "G": (
        ((1, 0.2), (0.5, 0), (0, 0.3), (0, 0.7), (0.5, 1), (1, 0.8), (1, 0.55)),
        ((0.55, 0.55), (1, 0.55)),
    ),
    "H": (((0, 0), (0, 1)), ((1, 0), (1, 1)), ((0, 0.5), (1, 0.5))),
    "I": (((0.5, 0), (0.5, 1)), ((0.2, 0), (0.8, 0)), ((0.2, 1), (0.8, 1))),
    "J": (((0.8, 0), (0.8, 0.8), (0.4, 1), (0, 0.8)),),
    "K": (((0, 0), (0, 1)), ((1, 0), (0, 0.5), (1, 1))),
    "L": (((0, 0), (0, 1), (1, 1)),),
    "M": (((0, 1), (0, 0), (0.5, 0.55), (1, 0), (1, 1)),),
    "N": (((0, 1), (0, 0), (1, 1), (1, 0)),),
    "O": (((0.5, 0), (1, 0.3), (1, 0.7), (0.5, 1), (0, 0.7), (0, 0.3), (0.5, 0)),),
    "P": (((0, 1), (0, 0), (0.85, 0.15), (0.85, 0.4), (0, 0.55)),),
    "Q": (
        ((0.5, 0), (1, 0.3), (1, 0.7), (0.5, 1), (0, 0.7), (0, 0.3), (0.5, 0)),
        ((0.6, 0.72), (1, 1)),
    ),
    "R": (((0, 1), (0, 0), (0.85, 0.15), (0.85, 0.4), (0, 0.55)), ((0.35, 0.55), (1, 1))),
    "S": (((1, 0.15), (0.4, 0), (0, 0.25), (0.6, 0.5), (1, 0.7), (0.6, 1), (0, 0.85)),),
    "T": (((0, 0), (1, 0)), ((0.5, 0), (0.5, 1))),
    "U": (((0, 0), (0, 0.7), (0.5, 1), (1, 0.7), (1, 0)),),
    "V": (((0, 0), (0.5, 1), (1, 0)),),
    "W": (((0, 0), (0.25, 1), (0.5, 0.4), (0.75, 1), (1, 0)),),
    "X": (((0, 0), (1, 1)), ((1, 0), (0, 1))),
    "Y": (((0, 0), (0.5, 0.5), (1, 0)), ((0.5, 0.5), (0.5, 1))),
    "Z": (((0, 0), (1, 0), (0, 1), (1, 1)),),
    "0": (
        ((0.5, 0), (1, 0.3), (1, 0.7), (0.5, 1), (0, 0.7), (0, 0.3), (0.5, 0)),
        ((0.2, 0.8), (0.8, 0.2)),
    ),
    "1": (((0.2, 0.2), (0.5, 0), (0.5, 1)), ((0.2, 1), (0.8, 1))),
    "2": (((0, 0.2), (0.5, 0), (1, 0.25), (0, 1), (1, 1)),),
    "3": (((0, 0.1), (0.6, 0), (1, 0.25), (0.4, 0.5), (1, 0.75), (0.6, 1), (0, 0.9)),),
    "4": (((0.7, 1), (0.7, 0), (0, 0.7), (1, 0.7)),),
    "5": (((1, 0), (0, 0), (0, 0.45), (0.6, 0.45), (1, 0.7), (0.5, 1), (0, 0.9)),),
    "6": (
        ((0.9, 0.1), (0.4, 0), (0, 0.4), (0, 0.8), (0.5, 1), (1, 0.8), (0.9, 0.55), (0.3, 0.5), (0, 0.65)),
    ),
    "7": (((0, 0), (1, 0), (0.35, 1)),),
    "8": (
        ((0.5, 0.5), (1, 0.3), (0.9, 0.05), (0.5, 0), (0.1, 0.05), (0, 0.3), (0.5, 0.5)),
        ((0.5, 0.5), (1, 0.72), (0.9, 0.96), (0.5, 1), (0.1, 0.96), (0, 0.72), (0.5, 0.5)),
    ),
    "9": (
        ((0.1, 0.9), (0.6, 1), (1, 0.6), (1, 0.2), (0.5, 0), (0, 0.2), (0.1, 0.45), (0.7, 0.5), (1, 0.35)),
    ),
    ".": (((0.45, 0.97), (0.55, 1)),),
    ",": (((0.55, 0.88), (0.35, 1.1)),),
    "-": (((0.15, 0.5), (0.85, 0.5)),),
    "_": (((0, 1), (1, 1)),),
    ":": (((0.5, 0.3), (0.5, 0.38)), ((0.5, 0.78), (0.5, 0.86))),
    "!": (((0.5, 0), (0.5, 0.7)), ((0.5, 0.95), (0.5, 1))),
    "?": (((0, 0.2), (0.5, 0), (1, 0.25), (0.5, 0.55), (0.5, 0.7)), ((0.5, 0.95), (0.5, 1))),
    "/": (((1, 0), (0, 1)),),
    "'": (((0.5, 0), (0.5, 0.25)),),
    "+": (((0.15, 0.5), (0.85, 0.5)), ((0.5, 0.15), (0.5, 0.85))),
    "=": (((0.15, 0.4), (0.85, 0.4)), ((0.15, 0.65), (0.85, 0.65))),
    "(": (((0.7, 0), (0.3, 0.5), (0.7, 1)),),
    ")": (((0.3, 0), (0.7, 0.5), (0.3, 1)),),
}

#: What :func:`text_segments` can draw. Anything else becomes a box glyph.
_UNKNOWN_GLYPH: tuple[tuple[tuple[float, float], ...], ...] = (
    ((0.1, 0.05), (0.9, 0.05), (0.9, 0.95), (0.1, 0.95), (0.1, 0.05)),
)


def text_coverage() -> str:
    """The characters :func:`text_segments` has glyphs for.

    Coverage is deliberately minimal and self-contained: ``A``-``Z``, ``0``-``9``,
    space and ``. , - _ : ! ? / ' + = ( )``. Lowercase input is upper-cased;
    ``\\n`` starts a new line; anything else is drawn as an empty box so a
    missing glyph is visible rather than silent.
    """
    return "".join(sorted(TEXT_GLYPHS))


def text_segments(
    text: str,
    origin: PointLike = (0.0, 0.0),
    *,
    height: float = 0.1,
    aspect: float = 0.6,
    tracking: float = 0.25,
    line_spacing: float = 1.4,
    curved: bool = False,
) -> list[Segment]:
    """Stroke ``text`` as segments, one per glyph edge (or per curve).

    ``origin`` is the top-left of the first glyph cell. ``height`` is the cell
    height in normalized units; a glyph is ``height * aspect`` wide and glyphs
    advance by ``height * aspect * (1 + tracking)``. ``\\n`` moves down by
    ``height * line_spacing``.

    ``curved=True`` runs each glyph polyline through :func:`catmull_rom_to_beziers`
    instead of emitting straight edges: fewer messages for the round glyphs,
    but noticeably wobbly letters. Coverage: see :func:`text_coverage`.

    Budget check: ~3 segments per character, so a 20-character caption is ~60
    messages -- fine against the ~150/s meeting budget, but not something to
    redraw every frame.
    """
    start = Point.of(origin)
    cell_w = height * aspect
    advance = cell_w * (1.0 + tracking)
    out: list[Segment] = []
    cursor_x, cursor_y = start.x, start.y
    for raw_char in text:
        if raw_char == "\n":
            cursor_x = start.x
            cursor_y += height * line_spacing
            continue
        char = raw_char.upper()
        glyph = TEXT_GLYPHS.get(char)
        if glyph is None:
            logger.debug("no glyph for %r; drawing a box", raw_char)
            glyph = _UNKNOWN_GLYPH
        for polyline in glyph:
            pts = [
                Point(cursor_x + px * cell_w, cursor_y + py * height)
                for px, py in polyline
            ]
            if curved and len(pts) > 2:
                out.extend(catmull_rom_to_beziers(pts))
            else:
                out.extend(polyline_segments(pts))
        cursor_x += advance
    return out


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------


def fit_points_into_rect(
    points: Sequence[PointLike],
    rect: Rect,
    *,
    preserve_aspect: bool = True,
    source: Rect | None = None,
) -> list[Point]:
    """Scale/translate ``points`` to fit inside ``rect``.

    Lets a caller build a drawing in whatever units are convenient and then
    park it in, say, the bottom-right corner of the share. ``source`` overrides
    the inferred bounding box (pass :func:`unit_square` to place a drawing that
    is already normalized). With ``preserve_aspect`` the result is centred in
    ``rect`` along the slack axis.

    Note that "aspect" here is aspect *in normalized space*: because each axis
    is normalized independently, preserving it does not preserve the on-screen
    aspect unless the share is square.
    """
    pts = [Point.of(p) for p in points]
    if not pts:
        return []
    src = (source or Rect.bounding(pts)).normalized()
    dst = rect.normalized()
    sx = dst.width / src.width if src.width else 0.0
    sy = dst.height / src.height if src.height else 0.0
    ox, oy = dst.x0, dst.y0
    if preserve_aspect:
        scale = min(sx, sy) if (sx and sy) else (sx or sy)
        sx = sy = scale
        ox += (dst.width - src.width * scale) / 2
        oy += (dst.height - src.height * scale) / 2
    return [Point(ox + (p.x - src.x0) * sx, oy + (p.y - src.y0) * sy) for p in pts]


def fit_segments_into_rect(
    segments: Sequence[Segment],
    rect: Rect,
    *,
    preserve_aspect: bool = True,
    source: Rect | None = None,
) -> list[Segment]:
    """:func:`fit_points_into_rect` for whole segments.

    The transform is derived from the segments' control points, so the fitted
    curve is guaranteed inside ``rect`` (a Bézier stays within its control
    points' convex hull).
    """
    if not segments:
        return []
    flat = [p for s in segments for p in s.points]
    moved = fit_points_into_rect(
        flat, rect, preserve_aspect=preserve_aspect, source=source
    )
    return [Segment(*moved[i : i + 4]) for i in range(0, len(moved), 4)]

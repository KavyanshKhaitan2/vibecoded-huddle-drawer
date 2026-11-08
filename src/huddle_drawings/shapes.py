"""Ready-made drawings: named shapes and SVG paths, as Bézier segments.

Everything here returns :class:`~huddle_drawings.geometry.Segment` lists in
normalized 0..1 coordinates, fitted into a target rect, ready for
:meth:`DrawClient.stroke`.

Segment count is the currency: one segment is one Chime data message, and the
whole meeting shares ~150 of those per second. Each shape below says what it
costs, and curves are used in preference to polylines wherever possible for
exactly the reason the doc gives -- a segment is a full cubic Bézier, so a
traced curve needs a small fraction of the messages that straight hatching
does.
"""

from __future__ import annotations

import math
import re

from .geometry import (
    Point,
    Rect,
    Segment,
    circle_segments,
    fit_points_into_rect,
    fit_segments_into_rect,
    polyline_segments,
    rect_segments,
    smooth_path,
    text_segments,
)

_UNIT = Rect(0.0, 0.0, 1.0, 1.0)


def _to_unit(points: list[Point]) -> list[Point]:
    """Rescale a point list into the unit square.

    Shapes are defined in whatever coordinate space is natural for their
    formula (the parametric heart spans roughly +/-16). Simplification
    tolerances in ``geometry`` are expressed in normalized units, so a shape
    must be brought into 0..1 *before* smoothing or the epsilon is effectively
    zero and no points are removed -- which silently costs one Chime message
    per sample instead of one per curve.
    """
    return fit_points_into_rect(points, _UNIT, preserve_aspect=True)

#: Named shapes available to ``--shape``.
SHAPES = ("heart", "star", "circle", "square", "box", "spiral", "arrow", "cross")


def _heart(samples: int = 64) -> list[Point]:
    """The classic parametric heart, sampled then smoothed into curves."""
    pts: list[Point] = []
    for i in range(samples + 1):
        t = 2 * math.pi * i / samples
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append(Point(x, y))
    return pts


def _star(points: int = 5, inner: float = 0.382) -> list[Point]:
    """A closed star polygon. ``inner`` 0.382 is the pentagram ratio."""
    pts: list[Point] = []
    for i in range(points * 2 + 1):
        angle = -math.pi / 2 + i * math.pi / points
        r = 1.0 if i % 2 == 0 else inner
        pts.append(Point(r * math.cos(angle), r * math.sin(angle)))
    return pts


def _spiral(turns: float = 3.0, samples: int = 96) -> list[Point]:
    pts: list[Point] = []
    for i in range(samples + 1):
        t = i / samples
        angle = 2 * math.pi * turns * t
        r = t
        pts.append(Point(r * math.cos(angle), r * math.sin(angle)))
    return pts


def _arrow() -> list[Point]:
    """A closed block arrow pointing right."""
    return [
        Point(0.0, 0.35),
        Point(0.55, 0.35),
        Point(0.55, 0.1),
        Point(1.0, 0.5),
        Point(0.55, 0.9),
        Point(0.55, 0.65),
        Point(0.0, 0.65),
        Point(0.0, 0.35),
    ]


def _cross() -> list[Point]:
    return [
        Point(0.35, 0.0),
        Point(0.65, 0.0),
        Point(0.65, 0.35),
        Point(1.0, 0.35),
        Point(1.0, 0.65),
        Point(0.65, 0.65),
        Point(0.65, 1.0),
        Point(0.35, 1.0),
        Point(0.35, 0.65),
        Point(0.0, 0.65),
        Point(0.0, 0.35),
        Point(0.35, 0.35),
        Point(0.35, 0.0),
    ]


def shape_segments(
    name: str,
    rect: Rect | None = None,
    *,
    filled: bool = False,
    fill_row_height: float | None = None,
    simplify: float = 0.01,
) -> list[Segment]:
    """Segments for a named shape, fitted into ``rect`` (default: whole frame).

    Curved shapes go through :func:`geometry.smooth_path`, so they cost a
    fraction of what the sampled polyline behind them would.

    ``simplify`` is the RDP tolerance in normalized units, and is the direct
    cost/fidelity dial: raising it drops segments (and therefore Chime
    messages) at the expense of detail. Measured on the heart: 0.002 -> 51
    segments, 0.01 (the default) -> 29, 0.05 -> 13. At the default that is
    about a fifth of a second of the meeting's whole message budget.
    """
    rect = rect or _UNIT
    name = name.lower()

    if name == "circle":
        segs = circle_segments(Point(0.5, 0.5), 0.5)
    elif name in ("square", "box"):
        segs = rect_segments(
            _UNIT, fill_row_height=fill_row_height if filled else None
        )
    elif name == "heart":
        segs = smooth_path(_to_unit(_heart()), epsilon=simplify, closed=True)
    elif name == "star":
        # A star's corners are the point of it -- keep them sharp.
        segs = polyline_segments(_to_unit(_star()), closed=True)
    elif name == "spiral":
        segs = smooth_path(_to_unit(_spiral()), epsilon=simplify, closed=False)
    elif name == "arrow":
        segs = polyline_segments(_arrow(), closed=True)
    elif name == "cross":
        segs = polyline_segments(_cross(), closed=True)
    else:
        raise ValueError(f"unknown shape {name!r}; available: {', '.join(SHAPES)}")

    return fit_segments_into_rect(segs, rect)


def text_in(
    text: str,
    rect: Rect | None = None,
    *,
    height: float = 0.12,
    curved: bool = False,
) -> list[Segment]:
    """Stroked text fitted into ``rect``.

    The input is upper-cased, because the built-in font covers only uppercase
    letters, digits and some punctuation -- see :func:`geometry.text_coverage`
    for the exact set.

    Anything outside that set renders as a **placeholder box**, not as nothing:
    the geometry layer substitutes an "unknown glyph" costing 4 segments per
    character. So accented or non-Latin text draws a row of boxes rather than
    failing, and is best checked with :func:`geometry.text_coverage` first.
    """
    rect = rect or Rect(0.0, 0.0, 1.0, 1.0)
    segs = text_segments(text.upper(), (0.0, 0.0), height=height, curved=curved)
    if not segs:
        return []
    return fit_segments_into_rect(segs, rect)


# -- a very small SVG path subset -----------------------------------------

# Arcs are matched on purpose, so an arc raises a clear error instead of being
# dropped by the tokenizer and leaving its numbers to be misread as line
# coordinates -- which produces a silently wrong picture.
_TOKEN = re.compile(
    r"[MmLlHhVvCcQqSsTtAaZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
)


def svg_path_segments(
    d: str,
    rect: Rect | None = None,
    *,
    smooth: bool = False,
) -> list[Segment]:
    """Parse an SVG path's ``d`` attribute into Bézier segments.

    Supports ``M m L l H h V v C c Q q S s T t Z z`` -- enough for most icon
    and logo paths. **Arcs (``A``/``a``) are not supported** and raise, rather
    than being silently dropped and giving you a subtly wrong picture; convert
    them to curves first (most vector editors can, and ``svgo`` will).

    Cubics and quadratics map straight onto the wire format: a quadratic is
    converted to its exact cubic equivalent, so neither loses fidelity.
    """
    tokens = _TOKEN.findall(d)
    if not tokens:
        return []
    if tokens[0] not in ("M", "m"):
        raise ValueError(
            f"an SVG path must start with a moveto (M or m), got {tokens[0]!r}"
        )

    segs: list[Segment] = []
    i = 0
    cur = Point(0.0, 0.0)
    start = Point(0.0, 0.0)
    cmd = ""
    prev_cubic_ctrl: Point | None = None
    prev_quad_ctrl: Point | None = None

    def num() -> float:
        nonlocal i
        if i >= len(tokens):
            raise ValueError(
                f"SVG path ended mid-command ({cmd!r}): an odd number of coordinates"
            )
        tok = tokens[i]
        i += 1
        try:
            return float(tok)
        except ValueError as exc:
            raise ValueError(
                f"expected a number in the SVG path after {cmd!r}, got {tok!r}"
            ) from exc

    def pt(relative: bool) -> Point:
        x = num()
        y = num()
        return Point(cur.x + x, cur.y + y) if relative else Point(x, y)

    def cubic(p1: Point, p2: Point, p3: Point) -> None:
        segs.append(Segment(cur, p1, p2, p3))

    while i < len(tokens):
        tok = tokens[i]
        if tok.isalpha():
            cmd = tok
            i += 1
            if cmd in "Aa":
                raise ValueError(
                    "SVG arcs (A/a) are not supported; convert the path's arcs to "
                    "cubic curves first"
                )
            if cmd in "Zz":
                if cur != start:
                    segs.extend(polyline_segments([cur, start]))
                cur = start
                prev_cubic_ctrl = prev_quad_ctrl = None
                continue
        elif not cmd:
            raise ValueError("SVG path must start with a command")

        rel = cmd.islower()
        up = cmd.upper()

        if up == "M":
            cur = start = pt(rel)
            prev_cubic_ctrl = prev_quad_ctrl = None
            # Subsequent coordinate pairs after an M are implicit L.
            cmd = "l" if rel else "L"
        elif up == "L":
            nxt = pt(rel)
            segs.extend(polyline_segments([cur, nxt]))
            cur = nxt
            prev_cubic_ctrl = prev_quad_ctrl = None
        elif up == "H":
            x = num()
            nxt = Point(cur.x + x if rel else x, cur.y)
            segs.extend(polyline_segments([cur, nxt]))
            cur = nxt
            prev_cubic_ctrl = prev_quad_ctrl = None
        elif up == "V":
            y = num()
            nxt = Point(cur.x, cur.y + y if rel else y)
            segs.extend(polyline_segments([cur, nxt]))
            cur = nxt
            prev_cubic_ctrl = prev_quad_ctrl = None
        elif up == "C":
            p1, p2, p3 = pt(rel), pt(rel), pt(rel)
            cubic(p1, p2, p3)
            prev_cubic_ctrl, prev_quad_ctrl = p2, None
            cur = p3
        elif up == "S":
            p1 = (
                Point(2 * cur.x - prev_cubic_ctrl.x, 2 * cur.y - prev_cubic_ctrl.y)
                if prev_cubic_ctrl
                else cur
            )
            p2, p3 = pt(rel), pt(rel)
            cubic(p1, p2, p3)
            prev_cubic_ctrl, prev_quad_ctrl = p2, None
            cur = p3
        elif up == "Q":
            q, p3 = pt(rel), pt(rel)
            # Exact quadratic -> cubic elevation.
            p1 = Point(cur.x + 2 / 3 * (q.x - cur.x), cur.y + 2 / 3 * (q.y - cur.y))
            p2 = Point(p3.x + 2 / 3 * (q.x - p3.x), p3.y + 2 / 3 * (q.y - p3.y))
            cubic(p1, p2, p3)
            prev_quad_ctrl, prev_cubic_ctrl = q, None
            cur = p3
        elif up == "T":
            q = (
                Point(2 * cur.x - prev_quad_ctrl.x, 2 * cur.y - prev_quad_ctrl.y)
                if prev_quad_ctrl
                else cur
            )
            p3 = pt(rel)
            p1 = Point(cur.x + 2 / 3 * (q.x - cur.x), cur.y + 2 / 3 * (q.y - cur.y))
            p2 = Point(p3.x + 2 / 3 * (q.x - p3.x), p3.y + 2 / 3 * (q.y - p3.y))
            cubic(p1, p2, p3)
            prev_quad_ctrl, prev_cubic_ctrl = q, None
            cur = p3
        else:
            raise ValueError(f"unsupported SVG path command {cmd!r}")

    if smooth and segs:
        segs = smooth_path([s.a for s in segs] + [segs[-1].d])

    return fit_segments_into_rect(segs, rect or Rect(0.0, 0.0, 1.0, 1.0))

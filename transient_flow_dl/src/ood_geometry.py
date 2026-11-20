from __future__ import annotations

import math
from collections import deque

import numpy as np
from PIL import Image, ImageDraw


OOD_FAMILIES = ("circle_array", "snowflakes", "trees", "staggered_baffles")


def _draw_thick_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    width: int,
) -> None:
    draw.line(
        (int(round(start[1])), int(round(start[0])), int(round(end[1])), int(round(end[0]))),
        fill=1,
        width=width,
    )


def _draw_snowflake(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    radius: float,
    width: int,
    rotation: float,
) -> None:
    cx, cy = center
    for arm in range(6):
        angle = rotation + arm * math.pi / 3.0
        endpoint = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        _draw_thick_line(draw, center, endpoint, width)
        for fraction in (0.55, 0.78):
            bx = cx + radius * fraction * math.cos(angle)
            by = cy + radius * fraction * math.sin(angle)
            branch_length = radius * 0.24
            for branch_angle in (angle - math.pi / 3.0, angle + math.pi / 3.0):
                branch_end = (
                    bx + branch_length * math.cos(branch_angle),
                    by + branch_length * math.sin(branch_angle),
                )
                _draw_thick_line(draw, (bx, by), branch_end, max(1, width - 1))


def _draw_tree(
    draw: ImageDraw.ImageDraw,
    root: tuple[float, float],
    height: float,
    width: int,
    lean: float,
) -> None:
    rx, ry = root
    trunk_end = (rx + lean, ry + height * 0.52)
    _draw_thick_line(draw, root, trunk_end, width + 1)
    crown = trunk_end
    for angle, scale in ((-1.00, 0.42), (-0.58, 0.50), (0.0, 0.56), (0.58, 0.50), (1.00, 0.42)):
        length = height * scale
        endpoint = (crown[0] + length * math.sin(angle), crown[1] + length * math.cos(angle))
        _draw_thick_line(draw, crown, endpoint, width)
        midpoint = ((crown[0] + endpoint[0]) * 0.5, (crown[1] + endpoint[1]) * 0.5)
        for sign in (-1.0, 1.0):
            branch_angle = angle + sign * 0.48
            branch_end = (
                midpoint[0] + height * 0.22 * math.sin(branch_angle),
                midpoint[1] + height * 0.22 * math.cos(branch_angle),
            )
            _draw_thick_line(draw, midpoint, branch_end, max(1, width - 1))


def _circle_array(draw: ImageDraw.ImageDraw, size: int, rng: np.random.Generator, severity: int) -> None:
    rows = 4 + severity
    columns = 4 + severity
    radius = 4.2 + 0.55 * severity
    x_locations = np.linspace(15, size - 15, rows)
    y_locations = np.linspace(12, size - 12, columns)
    for row, x in enumerate(x_locations):
        for y in y_locations:
            offset = (size / (columns + 1)) * 0.32 if row % 2 else 0.0
            cx = float(x + rng.uniform(-2.0, 2.0))
            cy = float(((y + offset - 8) % (size - 16)) + 8 + rng.uniform(-1.6, 1.6))
            local_radius = radius * float(rng.uniform(0.82, 1.12))
            draw.ellipse(
                (
                    int(cy - local_radius),
                    int(cx - local_radius),
                    int(cy + local_radius),
                    int(cx + local_radius),
                ),
                fill=1,
            )


def _snowflakes(draw: ImageDraw.ImageDraw, size: int, rng: np.random.Generator, severity: int) -> None:
    del size
    centers = [(x, y) for x in (17, 39, 61, 83) for y in (16, 39, 62, 84)]
    for cx, cy in centers:
        _draw_snowflake(
            draw,
            (cx + rng.uniform(-2, 2), cy + rng.uniform(-2, 2)),
            radius=6.3 + 0.7 * severity + rng.uniform(-0.5, 0.5),
            width=2 + (severity >= 2),
            rotation=float(rng.uniform(0, math.pi / 6)),
        )


def _trees(draw: ImageDraw.ImageDraw, size: int, rng: np.random.Generator, severity: int) -> None:
    del size
    roots = [(x, y) for x in (15, 38, 62, 85) for y in (6, 38, 70)]
    for index, root in enumerate(roots):
        height = 17.0 + 1.0 * severity + rng.uniform(-1.0, 1.0)
        _draw_tree(
            draw,
            (root[0] + rng.uniform(-1, 1), root[1] + rng.uniform(-1.5, 1.5)),
            height=height,
            width=2,
            lean=(-2.0 if index % 2 else 2.0) + rng.uniform(-1.0, 1.0),
        )


def _staggered_baffles(
    draw: ImageDraw.ImageDraw,
    size: int,
    rng: np.random.Generator,
    severity: int,
) -> None:
    count = 4 + severity
    thickness = 4 + severity
    gap = 28 - 3 * severity
    x_locations = np.linspace(15, size - 15, count)
    for index, x in enumerate(x_locations):
        jitter = int(round(rng.uniform(-2, 2)))
        x0 = int(round(x - thickness / 2))
        x1 = x0 + thickness
        if index % 2 == 0:
            draw.rectangle((0, x0, size - gap + jitter, x1), fill=1)
        else:
            draw.rectangle((gap + jitter, x0, size - 1, x1), fill=1)
        tip_y = size - gap + jitter if index % 2 == 0 else gap + jitter
        draw.ellipse((tip_y - 4, x0 - 2, tip_y + 4, x1 + 2), fill=1)


def through_connected_pore(solid: np.ndarray) -> np.ndarray:
    """Return all four-neighbour fluid nodes connected to both flow faces."""
    pore = solid == 0
    height, width = pore.shape

    def flood(row: int) -> np.ndarray:
        reached = np.zeros_like(pore, dtype=bool)
        queue: deque[tuple[int, int]] = deque()
        for column in np.flatnonzero(pore[row]):
            reached[row, column] = True
            queue.append((row, int(column)))
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < height and 0 <= ny < width and pore[nx, ny] and not reached[nx, ny]:
                    reached[nx, ny] = True
                    queue.append((nx, ny))
        return reached

    return flood(0) & flood(height - 1)


def _dilate_solid(solid: np.ndarray) -> np.ndarray:
    padded = np.pad(solid, 1, mode="constant", constant_values=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    dilated = windows.max(axis=(-2, -1)).astype(np.uint8)
    dilated[:2, :] = 0
    dilated[-2:, :] = 0
    return dilated


def _match_porosity(solid: np.ndarray, target_porosity: float) -> np.ndarray:
    """Uniformly thicken motifs while preserving a through-connected fluid path."""
    if not 0.30 <= target_porosity <= 0.90:
        raise ValueError("target_porosity must lie in [0.30, 0.90]")
    best = solid.copy()
    best_error = abs(float((best == 0).mean()) - target_porosity)
    for _ in range(32):
        candidate = _dilate_solid(best)
        connected = through_connected_pore(candidate)
        if not connected[0].any() or not connected[-1].any():
            break
        error = abs(float((candidate == 0).mean()) - target_porosity)
        if error > best_error and float((candidate == 0).mean()) < target_porosity:
            break
        best = candidate
        best_error = error
        if best_error <= 0.012:
            break
    return best


def make_rule_mask(
    family: str,
    size: int = 100,
    seed: int = 2026,
    severity: int = 1,
    target_porosity: float | None = 0.65,
) -> np.ndarray:
    """Create a deterministic solid obstacle field outside the QSGS morphology distribution."""
    if family not in OOD_FAMILIES:
        raise ValueError(f"Unknown OOD family: {family}")
    if severity not in (0, 1, 2):
        raise ValueError("severity must be 0, 1, or 2")
    rng = np.random.default_rng(seed)
    image = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(image)
    generators = {
        "circle_array": _circle_array,
        "snowflakes": _snowflakes,
        "trees": _trees,
        "staggered_baffles": _staggered_baffles,
    }
    generators[family](draw, size, rng, severity)
    solid = np.asarray(image, dtype=np.uint8).copy()

    # Pressure boundaries act on the first axis. Keep both faces fully fluid.
    solid[:2, :] = 0
    solid[-2:, :] = 0
    if target_porosity is not None:
        solid = _match_porosity(solid, target_porosity)
    connected = through_connected_pore(solid)
    if not connected[0].any() or not connected[-1].any():
        raise RuntimeError(f"Generated {family} mask is not through-connected")
    return solid


def geometry_statistics(solid: np.ndarray) -> dict[str, float]:
    pore = solid == 0
    connected = through_connected_pore(solid)
    porosity = float(pore.mean())
    connected_porosity = float(connected.mean())
    return {
        "porosity": porosity,
        "connected_porosity": connected_porosity,
        "dead_pore_ratio": 1.0 - connected_porosity / porosity if porosity > 0 else 1.0,
    }

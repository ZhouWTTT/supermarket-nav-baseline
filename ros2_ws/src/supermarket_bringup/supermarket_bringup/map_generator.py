"""Generate the public fixed-scene occupancy map without server truth access."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


ORIGIN_X = -2.5
ORIGIN_Y = -3.75
RESOLUTION = 0.05
WIDTH = 100
HEIGHT = 150


def build_grid() -> list[list[int]]:
    grid = [[254 for _ in range(WIDTH)] for _ in range(HEIGHT)]

    def fill_rect(xmin: float, ymin: float, xmax: float, ymax: float) -> None:
        gx0 = max(0, math.floor((xmin - ORIGIN_X) / RESOLUTION))
        gy0 = max(0, math.floor((ymin - ORIGIN_Y) / RESOLUTION))
        gx1 = min(WIDTH - 1, math.floor((xmax - ORIGIN_X) / RESOLUTION))
        gy1 = min(HEIGHT - 1, math.floor((ymax - ORIGIN_Y) / RESOLUTION))
        for gy in range(gy0, gy1 + 1):
            for gx in range(gx0, gx1 + 1):
                grid[gy][gx] = 0

    fill_rect(-2.50, -3.78, 2.50, -3.72)
    fill_rect(-2.50, 3.72, 2.50, 3.78)
    fill_rect(-2.53, -3.75, -2.47, 3.75)
    fill_rect(2.47, -3.75, 2.53, 3.75)
    fill_rect(0.50, -3.72, 0.56, 1.70)
    for center_x in (-1.735, -0.850, 0.035, 0.920, 1.805):
        fill_rect(center_x - 0.45, 3.173, center_x + 0.45, 3.493)
    fill_rect(-2.45, -3.66, -1.43, -3.16)
    return grid


def write_pgm(path: Path) -> None:
    grid = build_grid()
    lines = ["P2", f"{WIDTH} {HEIGHT}", "255"]
    # PGM starts at the top-left while OccupancyGrid origins are bottom-left.
    lines.extend(" ".join(map(str, row)) for row in reversed(grid))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_pgm(args.output)


if __name__ == "__main__":
    main()

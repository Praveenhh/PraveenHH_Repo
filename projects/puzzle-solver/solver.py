from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

SIZE = 9
FULL_MASK = (1 << SIZE) - 1


def parse_puzzle(text: str) -> list[int]:
    """Parse an 81-character row-major puzzle using digits, dots, or zeroes."""
    cells = [char for char in text if not char.isspace()]
    if len(cells) != SIZE * SIZE:
        raise ValueError(f"Puzzle must contain exactly 81 cells; got {len(cells)}")
    if any(char not in "0123456789." for char in cells):
        raise ValueError("Puzzle cells must be digits 1-9, zero, or '.'")
    return [int(char) if char in "123456789" else 0 for char in cells]


def units() -> list[list[int]]:
    rows = [[r * 9 + c for c in range(9)] for r in range(9)]
    columns = [[r * 9 + c for r in range(9)] for c in range(9)]
    boxes = [
        [(box_r + dr) * 9 + box_c + dc for dr in range(3) for dc in range(3)]
        for box_r in (0, 3, 6)
        for box_c in (0, 3, 6)
    ]
    return rows + columns + boxes


UNITS = units()
CELL_UNITS: list[list[list[int]]] = [
    [unit for unit in UNITS if cell in unit] for cell in range(81)
]


def validate_clues(grid: list[int]) -> None:
    if len(grid) != 81 or any(value < 0 or value > 9 for value in grid):
        raise ValueError("Grid must contain 81 values in the range 0-9")
    for unit in UNITS:
        values = [grid[index] for index in unit if grid[index] != 0]
        if len(values) != len(set(values)):
            raise ValueError("Puzzle contains conflicting clues")


def is_complete_solution(grid: list[int]) -> bool:
    return len(grid) == 81 and all(
        set(grid[index] for index in unit) == set(range(1, 10)) for unit in UNITS
    )


def candidates(grid: list[int], cell: int) -> int:
    used = 0
    for unit in CELL_UNITS[cell]:
        for index in unit:
            value = grid[index]
            if value:
                used |= 1 << (value - 1)
    return FULL_MASK & ~used


def mask_values(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length()
        mask ^= bit


@dataclass
class SearchStats:
    search_nodes: int = 0
    backtracks: int = 0
    assignments: int = 0
    propagation_rounds: int = 0
    max_search_depth: int = 0


@dataclass
class SolveResult:
    status: str
    solutions: list[list[int]]
    stats: SearchStats
    elapsed_seconds: float


def propagate(grid: list[int], stats: SearchStats) -> bool:
    """Apply naked and hidden singles until stable; return false on contradiction."""
    while True:
        stats.propagation_rounds += 1
        candidate_map: dict[int, int] = {}
        for cell, value in enumerate(grid):
            if value == 0:
                options = candidates(grid, cell)
                if options == 0:
                    return False
                candidate_map[cell] = options

        forced: tuple[int, int] | None = None
        for cell, options in candidate_map.items():
            if options & (options - 1) == 0:
                forced = (cell, options.bit_length())
                break

        if forced is None:
            for unit in UNITS:
                placed = {grid[cell] for cell in unit if grid[cell]}
                for value in range(1, 10):
                    if value in placed:
                        continue
                    possible = [
                        cell for cell in unit
                        if cell in candidate_map and candidate_map[cell] & (1 << (value - 1))
                    ]
                    if not possible:
                        return False
                    if len(possible) == 1:
                        forced = (possible[0], value)
                        break
                if forced is not None:
                    break

        if forced is None:
            return True
        cell, value = forced
        grid[cell] = value
        stats.assignments += 1


def search(grid: list[int], solutions: list[list[int]], stats: SearchStats, depth: int, limit: int) -> None:
    if len(solutions) >= limit:
        return
    stats.search_nodes += 1
    stats.max_search_depth = max(stats.max_search_depth, depth)
    state = grid.copy()
    if not propagate(state, stats):
        stats.backtracks += 1
        return
    if all(state):
        if is_complete_solution(state):
            solutions.append(state)
        else:
            stats.backtracks += 1
        return

    open_cells = [cell for cell, value in enumerate(state) if value == 0]
    cell = min(open_cells, key=lambda index: candidates(state, index).bit_count())
    options = candidates(state, cell)
    for value in mask_values(options):
        if len(solutions) >= limit:
            break
        branch = state.copy()
        branch[cell] = value
        stats.assignments += 1
        search(branch, solutions, stats, depth + 1, limit)
        if len(solutions) == 0:
            stats.backtracks += 1


def solve(grid: list[int], max_solutions: int = 2) -> SolveResult:
    """Solve a Sudoku, optionally searching for enough solutions to test uniqueness."""
    if max_solutions < 1:
        raise ValueError("max_solutions must be at least 1")
    validate_clues(grid)
    started = perf_counter()
    solutions: list[list[int]] = []
    stats = SearchStats()
    search(grid.copy(), solutions, stats, depth=0, limit=max_solutions)
    elapsed = perf_counter() - started
    if not solutions:
        status = "unsatisfiable"
    elif len(solutions) > 1:
        status = "multiple"
    elif max_solutions == 1:
        status = "solution_found"
    else:
        status = "unique"
    return SolveResult(status, solutions, stats, elapsed)


def format_grid(grid: list[int]) -> str:
    lines = []
    for row in range(9):
        values = [str(value) for value in grid[row * 9 : (row + 1) * 9]]
        lines.append(" ".join(values[:3]) + " | " + " ".join(values[3:6]) + " | " + " ".join(values[6:]))
        if row in (2, 5):
            lines.append("------+-------+------")
    return "\n".join(lines)


def stats_dict(stats: SearchStats) -> dict:
    return asdict(stats)

# Constraint-Based Sudoku Solver

A command-line Sudoku solver built around constraint propagation and backtracking search. The solver applies naked and hidden singles, selects the next cell using the minimum-remaining-values heuristic, and can search for one solution or check whether a puzzle has a unique solution.

The implementation is dependency-free and exposes search diagnostics for inspecting how constraint propagation and branching affect solver effort. It is an algorithmic demonstration, not a general-purpose solver for every puzzle type.

## Puzzle format

Provide exactly 81 characters in row-major order. Digits `1`–`9` are clues; `0` and `.` denote empty cells. The input must not contain conflicting clues.

A sample puzzle is available at `data/sample.txt`.

## Run

Python 3.10+; no third-party packages are required.

```bash
python solve.py --puzzle "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
```

Load a puzzle from a text file and check for uniqueness:

```bash
python solve.py --file data/sample.txt --max-solutions 2
```

Write machine-readable output to a JSON file:

```bash
python solve.py --file data/sample.txt --output artifacts/result.json
```

`--max-solutions 2` is the default and distinguishes unique puzzles from puzzles with multiple solutions. Searching for uniqueness may require more work than finding the first solution. Use `--max-solutions 1` when only one solution is needed; the reported status will then be `solution_found`, not `unique`.

## Solver design

1. Candidate sets are derived from row, column, and 3×3 box constraints.
2. Constraint propagation repeatedly assigns naked singles and hidden singles.
3. If propagation cannot finish the puzzle, the solver branches on the unfilled cell with the fewest candidates (MRV).
4. Each branch is explored recursively until a contradiction is found or the requested solution limit is reached.
5. Every completed grid is checked against the Sudoku constraints before it is returned.

## Diagnostics

The JSON result includes:

- **status** — `unsatisfiable`, `unique`, `multiple`, or `solution_found` (when search was intentionally limited to one result).
- **solutions_found** — number of solutions discovered, capped at `--max-solutions`.
- **search_nodes** — recursive search states visited.
- **backtracks** — candidate branches that led to a contradiction or no solution.
- **assignments** — forced and guessed cell assignments made during search.
- **propagation_rounds** — constraint-propagation passes across visited states.
- **max_search_depth** — deepest branch depth reached.
- **elapsed_seconds** — wall-clock duration for the solve operation.

These are algorithm diagnostics, not predictive-quality metrics. Runtime depends on puzzle difficulty, hardware, and the solution limit. No performance guarantee is implied.

## Tests

Run the unit tests from this directory:

```bash
python -m unittest discover -s tests -v
```

Tests cover parsing, conflict detection, solved-grid validity, uniqueness detection, and the unsatisfiable case.

## Scope and limitations

- Supports standard 9×9 Sudoku only; it does not support variants such as diagonal Sudoku or Killer Sudoku.
- A puzzle with conflicting clues is rejected during validation; a valid-looking but unsatisfiable puzzle returns no solution.
- Uniqueness is established only when searching to two solutions and fewer than two are found after exhaustive search.
- The solver is intended for learning and experimentation, not as a benchmark suite.

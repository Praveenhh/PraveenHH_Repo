from __future__ import annotations

import unittest

from solver import is_complete_solution, parse_puzzle, solve


PUZZLE = (
    "53..7...."
    "6..195..."
    ".98....6."
    "8...6...3"
    "4..8.3..1"
    "7...2...6"
    ".6....28."
    "...419..5"
    "....8..79"
)


class SudokuSolverTests(unittest.TestCase):
    def test_parser_accepts_dots_zeroes_and_whitespace(self) -> None:
        grid = parse_puzzle("53..7....\n" + "0" * 72)
        self.assertEqual(len(grid), 81)
        self.assertEqual(grid[0], 5)
        self.assertEqual(grid[2], 0)

    def test_parser_rejects_wrong_length_and_characters(self) -> None:
        with self.assertRaises(ValueError):
            parse_puzzle("123")
        with self.assertRaises(ValueError):
            parse_puzzle("x" * 81)

    def test_solves_sample_and_verifies_unique_solution(self) -> None:
        result = solve(parse_puzzle(PUZZLE))
        self.assertEqual(result.status, "unique")
        self.assertEqual(len(result.solutions), 1)
        self.assertTrue(is_complete_solution(result.solutions[0]))
        self.assertGreater(result.stats.search_nodes, 0)

    def test_conflicting_clues_are_rejected(self) -> None:
        puzzle = "11" + "." * 79
        with self.assertRaisesRegex(ValueError, "conflicting clues"):
            solve(parse_puzzle(puzzle))

    def test_valid_but_unsatisfiable_puzzle_returns_no_solution(self) -> None:
        # Removing the 1 from the sample and placing it in the first cell creates
        # a locally valid clue arrangement that contradicts the puzzle's solution.
        puzzle = "13..7...." + "6..195..." + ".98....6." + "8...6...3" + "4..8.3..1" + "7...2...6" + ".6....28." + "...419..5" + "....8..79"
        result = solve(parse_puzzle(puzzle))
        self.assertEqual(result.status, "unsatisfiable")
        self.assertEqual(result.solutions, [])

    def test_one_solution_limit_does_not_claim_uniqueness(self) -> None:
        result = solve(parse_puzzle(PUZZLE), max_solutions=1)
        self.assertEqual(result.status, "solution_found")


if __name__ == "__main__":
    unittest.main()

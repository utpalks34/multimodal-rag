"""Unit tests for the programmatic checks in eval/run_eval.py (prompts.md section 4).
No test framework is pinned in requirements.txt yet, so this runs as a plain script:
    python -m eval.test_checks
"""
from eval.run_eval import expected_numbers_in_context, expected_numbers_recall, ungrounded_numbers

# (answer, context_text, expected_flagged_numbers)
CASES = [
    ("Revenue was 4,332 dollars [C1].", "The filing reports 4,332 total units.", []),
    ("The score was 1322 [C1].", "Recorded score: 1,322 points.", []),
    ("The value was 4 [C1].", "Published in 2014.", ["4"]),
    ("The total was 58 [C1].", "Precisely 58.54 percent.", ["58"]),
]

# (expected_answer, context_text, expected_numbers_found_in_context)
IN_CONTEXT_CASES = [
    ("4,332.", "Home Tests: 4,332 runs, average 58.54.", ["4332"]),
    ("Rajkumar Sharma.", "Trained under Rajkumar Sharma in Delhi.", []),  # no numbers -> nothing to find
    ("94.15.", "Batting first: 92.45 strike rate.", []),  # rounded-nearby number must not match
    ("Chasing: 64.21, versus 50.18 batting first.", "50.18 batting first vs. 64.21 chasing.", ["64.21", "50.18"]),
]

# (expected_answer, context_text, expected_recall)
RECALL_CASES = [
    ("4,332.", "Home Tests: 4,332 runs, average 58.54.", 1.0),
    ("95 matches; 65 wins.", "40 wins in 68 matches.", 0.0),
    ("Chasing: 64.21, versus 50.18 batting first.", "50.18 batting first vs. 64.21 chasing.", 1.0),
    ("Rajkumar Sharma.", "Trained under Rajkumar Sharma in Delhi.", None),  # no numbers in expected_answer
]


def test_ungrounded_numbers():
    for answer, context, expected in CASES:
        result = sorted(ungrounded_numbers(answer, context))
        assert result == sorted(expected), (
            f"ungrounded_numbers({answer!r}, {context!r}) = {result}, expected {expected}"
        )


def test_expected_numbers_in_context():
    for expected_answer, context, expected in IN_CONTEXT_CASES:
        result = sorted(expected_numbers_in_context(expected_answer, context))
        assert result == sorted(expected), (
            f"expected_numbers_in_context({expected_answer!r}, {context!r}) = {result}, expected {expected}"
        )


def test_expected_numbers_recall():
    for expected_answer, context, expected in RECALL_CASES:
        result = expected_numbers_recall(expected_answer, context)
        assert result == expected, (
            f"expected_numbers_recall({expected_answer!r}, {context!r}) = {result}, expected {expected}"
        )


if __name__ == "__main__":
    test_ungrounded_numbers()
    print("OK: test_ungrounded_numbers")
    test_expected_numbers_in_context()
    print("OK: test_expected_numbers_in_context")
    test_expected_numbers_recall()
    print("OK: test_expected_numbers_recall")

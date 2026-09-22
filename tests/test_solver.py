"""Unit tests: mocked transport, real local verification logic."""
import json
import math
import unittest

from mathsolver_help import solve, eval_expression, SolverError

GOOD = json.dumps({"answer": 4, "steps": ["Subtract 3: 2x = 8", "Divide by 2: x = 4"],
                   "verification": {"expression": "(11-3)/2"}})
WRONG = json.dumps({"answer": 4, "steps": ["..."], "verification": {"expression": "(11-3)/3"}})


class TestEvalExpression(unittest.TestCase):
    def test_arithmetic_precedence(self):
        self.assertEqual(eval_expression("2*3+4"), 10)
        self.assertEqual(eval_expression("2+3*4"), 14)
        self.assertEqual(eval_expression("(2+3)*4"), 20)
        self.assertEqual(eval_expression("2^3^2"), 512)
        self.assertEqual(eval_expression("-3^2"), -9)
        self.assertAlmostEqual(eval_expression("10%3"), 1)

    def test_functions_constants(self):
        self.assertEqual(eval_expression("sqrt(16)"), 4)
        self.assertEqual(eval_expression("min(3,5)"), 3)
        self.assertAlmostEqual(eval_expression("pi"), math.pi)
        self.assertAlmostEqual(eval_expression("log(1000)"), 3, places=12)
        self.assertAlmostEqual(eval_expression("ln(e)"), 1, places=12)

    def test_rejects_dangerous(self):
        for bad in ["__import__('os').system('x')", "1+2)", "foo(1)", ""]:
            with self.assertRaises(SolverError):
                eval_expression(bad)


class TestSolve(unittest.TestCase):
    def test_verified_first_try(self):
        calls = []
        def transport(url, body, key):
            calls.append((url, body, key))
            return GOOD
        r = solve("2x + 3 = 11, solve for x", api_key="sk-test", transport=transport)
        self.assertTrue(r.verified)
        self.assertEqual(r.answer, 4)
        self.assertEqual(r.evaluated, 4)
        self.assertEqual(r.retries, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], "sk-test")
        self.assertTrue(calls[0][0].endswith("/chat/completions"))

    def test_retry_recovers(self):
        n = [0]
        def transport(url, body, key):
            n[0] += 1
            return WRONG if n[0] == 1 else GOOD
        r = solve("2x+3=11", api_key="sk", transport=transport)
        self.assertTrue(r.verified)
        self.assertEqual(r.retries, 1)
        self.assertEqual(r.evaluated, 4)

    def test_invalid_json_then_ok(self):
        n = [0]
        def transport(url, body, key):
            n[0] += 1
            return "blah no json" if n[0] == 1 else GOOD
        r = solve("1+1", api_key="sk", transport=transport)
        self.assertTrue(r.verified)

    def test_invalid_json_twice_raises(self):
        def transport(url, body, key):
            return "still no json"
        with self.assertRaises(SolverError) as ctx:
            solve("1+1", api_key="sk", transport=transport)
        self.assertEqual(ctx.exception.code, "INVALID_JSON")

    def test_no_api_key(self):
        with self.assertRaises(SolverError) as ctx:
            solve("1+1")
        self.assertEqual(ctx.exception.code, "NO_API_KEY")

    def test_http_error_no_retry(self):
        calls = [0]
        def transport(url, body, key):
            calls[0] += 1
            raise SolverError("HTTP_ERROR", "401")
        with self.assertRaises(SolverError) as ctx:
            solve("1+1", api_key="sk", transport=transport)
        self.assertEqual(ctx.exception.code, "HTTP_ERROR")
        self.assertEqual(calls[0], 1)

    def test_retry_still_wrong_returns_unverified(self):
        r = solve("2x+3=11", api_key="sk", transport=lambda *a: WRONG)
        self.assertFalse(r.verified)
        self.assertEqual(r.retries, 1)


if __name__ == "__main__":
    unittest.main()

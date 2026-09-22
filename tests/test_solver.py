"""Unit tests: mocked transport, real local verification logic."""
import json
import math
import unittest

from mathsolver_help import MathSolver, eval_expression, SolverError

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


class TestClient(unittest.TestCase):
    def test_no_api_key_fails_at_init(self):
        with self.assertRaises(SolverError) as ctx:
            MathSolver(api_key="")
        self.assertEqual(ctx.exception.code, "NO_API_KEY")

    def test_bad_base_url_fails_at_init(self):
        with self.assertRaises(SolverError) as ctx:
            MathSolver(api_key="sk", base_url="not-a-url")
        self.assertEqual(ctx.exception.code, "BAD_BASE_URL")

    def test_verified_first_try(self):
        calls = []
        def transport(url, body, key):
            calls.append((url, body, key))
            return GOOD
        solver = MathSolver(api_key="sk-test", base_url="https://api.deepseek.com/v1",
                            model="deepseek-chat", transport=transport)
        r = solver.solve("2x + 3 = 11, solve for x")
        self.assertTrue(r.verified)
        self.assertEqual(r.answer, 4)
        self.assertEqual(r.evaluated, 4)
        self.assertEqual(r.retries, 0)
        self.assertEqual(len(calls), 1)
        # OpenAI 格式: base_url + /chat/completions, Bearer key, chat body
        self.assertEqual(calls[0][0], "https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(calls[0][2], "sk-test")
        self.assertEqual(calls[0][1]["model"], "deepseek-chat")
        self.assertEqual(calls[0][1]["messages"][0]["role"], "system")
        self.assertEqual(calls[0][1]["temperature"], 0)

    def test_retry_recovers(self):
        n = [0]
        def transport(url, body, key):
            n[0] += 1
            return WRONG if n[0] == 1 else GOOD
        r = MathSolver(api_key="sk", transport=transport).solve("2x+3=11")
        self.assertTrue(r.verified)
        self.assertEqual(r.retries, 1)
        self.assertEqual(r.evaluated, 4)

    def test_invalid_json_then_ok(self):
        n = [0]
        def transport(url, body, key):
            n[0] += 1
            return "blah no json" if n[0] == 1 else GOOD
        r = MathSolver(api_key="sk", transport=transport).solve("1+1")
        self.assertTrue(r.verified)

    def test_invalid_json_twice_raises(self):
        solver = MathSolver(api_key="sk", transport=lambda *a: "still no json")
        with self.assertRaises(SolverError) as ctx:
            solver.solve("1+1")
        self.assertEqual(ctx.exception.code, "INVALID_JSON")

    def test_empty_problem_raises_before_http(self):
        calls = [0]
        solver = MathSolver(api_key="sk", transport=lambda *a: calls.__setitem__(0, calls[0] + 1) or GOOD)
        with self.assertRaises(SolverError) as ctx:
            solver.solve("   ")
        self.assertEqual(ctx.exception.code, "NO_PROBLEM")
        self.assertEqual(calls[0], 0)

    def test_http_error_no_retry(self):
        calls = [0]
        def transport(url, body, key):
            calls[0] += 1
            raise SolverError("HTTP_ERROR", "401")
        solver = MathSolver(api_key="sk", transport=transport)
        with self.assertRaises(SolverError) as ctx:
            solver.solve("1+1")
        self.assertEqual(ctx.exception.code, "HTTP_ERROR")
        self.assertEqual(calls[0], 1)

    def test_retry_still_wrong_returns_unverified(self):
        solver = MathSolver(api_key="sk", transport=lambda *a: WRONG)
        r = solver.solve("2x+3=11")
        self.assertFalse(r.verified)
        self.assertEqual(r.retries, 1)


if __name__ == "__main__":
    unittest.main()

"""v0.2 mock tests: simulate model replies, verify the post-receipt chain
(program execution -> answer -> check substitution -> retry logic)."""
import json
import math
import os
import unittest

from mathsolver_help import MathSolver, eval_expression, run_program, run_check, SolverError

GOOD = json.dumps({
    "program": "let d = 11 - 3;\nlet x = d / 2;\nresult = x",
    "steps": ["Subtract 3: 2x = 8", "Divide by 2: x = 4"],
    "check": "2*{x} + 3 - 11",
})
NO_CHECK = json.dumps({"program": "result = 0.15 * 80", "steps": ["Compute 15% of 80"]})
WRONG_CHECK = json.dumps({
    "program": "let d = 11 - 3;\nresult = d / 2",
    "steps": ["..."],
    "check": "2*{x} + 3 - 12",  # evaluates to -1
})
BROKEN_PROGRAM = json.dumps({"program": "result = undefinedvar + 1", "steps": []})


class TestEvalExpression(unittest.TestCase):
    def test_arithmetic_precedence(self):
        self.assertEqual(eval_expression("2*3+4"), 10)
        self.assertEqual(eval_expression("2+3*4"), 14)
        self.assertEqual(eval_expression("(2+3)*4"), 20)
        self.assertEqual(eval_expression("2^3^2"), 512)
        self.assertEqual(eval_expression("-3^2"), -9)

    def test_env_variables(self):
        self.assertEqual(eval_expression("d / 2", {"d": 8}), 4)
        self.assertEqual(eval_expression("x + y", {"x": 1.5, "y": 2.5}), 4)
        with self.assertRaises(SolverError):
            eval_expression("d")  # undefined var


class TestRunProgram(unittest.TestCase):
    def test_let_and_result(self):
        self.assertEqual(run_program("let d = 11 - 3;\nlet x = d / 2;\nresult = x"), 4)

    def test_semicolons_and_bare_final(self):
        self.assertEqual(run_program("let a = 3; let b = 4; a * b"), 12)
        self.assertEqual(run_program("0.15 * 80"), 12)

    def test_rejects_broken(self):
        with self.assertRaises(SolverError):
            run_program("result = undefinedvar + 1")
        with self.assertRaises(SolverError):
            run_program("")
        with self.assertRaises(SolverError):
            run_program("let a = 1; let b = 2")  # no result


class TestRunCheck(unittest.TestCase):
    def test_pass_fail(self):
        value, passed = run_check("2*{x} + 3 - 11", 4)
        self.assertEqual(value, 0)
        self.assertTrue(passed)
        value, passed = run_check("2*{x} + 3 - 12", 4)
        self.assertEqual(value, -1)
        self.assertFalse(passed)

    def test_arithmetic_inverse_path(self):
        _, passed = run_check("80*15/100 - {x}", 12)
        self.assertTrue(passed)


class TestClient(unittest.TestCase):
    def test_no_api_key_fails_at_init(self):
        with self.assertRaises(SolverError) as ctx:
            MathSolver(api_key="")
        self.assertEqual(ctx.exception.code, "NO_API_KEY")

    def test_bad_base_url_fails_at_init(self):
        with self.assertRaises(SolverError) as ctx:
            MathSolver(api_key="sk", base_url="not-a-url")
        self.assertEqual(ctx.exception.code, "BAD_BASE_URL")

    def test_answer_from_execution_check_passes_first_try(self):
        calls = []

        def transport(url, body, key):
            calls.append((url, body, key))
            return GOOD

        solver = MathSolver(api_key="sk-test", base_url="https://api.deepseek.com/v1",
                            model="deepseek-chat", transport=transport)
        r = solver.solve("2x + 3 = 11, solve for x")
        self.assertTrue(r.verified)
        self.assertEqual(r.answer, 4)
        self.assertEqual(r.check_value, 0)
        self.assertEqual(r.retries, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(calls[0][1]["model"], "deepseek-chat")
        # v0.2 协议: 模型 JSON 里没有 answer 字段
        self.assertNotIn("answer", json.loads(GOOD))

    def test_no_check_answer_from_execution_verified_false(self):
        r = MathSolver(api_key="sk", transport=lambda *a: NO_CHECK).solve("15% of 80")
        self.assertEqual(r.answer, 12)
        self.assertFalse(r.verified)
        self.assertIsNone(r.check)

    def test_check_fail_retry_recovers(self):
        n = [0]

        def transport(url, body, key):
            n[0] += 1
            return WRONG_CHECK if n[0] == 1 else GOOD

        r = MathSolver(api_key="sk", transport=transport).solve("2x+3=11")
        self.assertTrue(r.verified)
        self.assertEqual(r.retries, 1)

    def test_program_error_retry_recovers(self):
        n = [0]

        def transport(url, body, key):
            n[0] += 1
            return BROKEN_PROGRAM if n[0] == 1 else GOOD

        r = MathSolver(api_key="sk", transport=transport).solve("2x+3=11")
        self.assertTrue(r.verified)
        self.assertEqual(r.answer, 4)

    def test_program_error_persists_raises(self):
        solver = MathSolver(api_key="sk", transport=lambda *a: BROKEN_PROGRAM)
        with self.assertRaises(SolverError) as ctx:
            solver.solve("2x+3=11")
        self.assertTrue(ctx.exception.code.startswith(("PROGRAM_", "EXPR_")))

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

    def test_check_still_failing_unverified_answer_from_execution(self):
        r = MathSolver(api_key="sk", transport=lambda *a: WRONG_CHECK).solve("2x+3=11")
        self.assertEqual(r.answer, 4)
        self.assertFalse(r.verified)
        self.assertEqual(r.retries, 1)


@unittest.skipIf(not os.environ.get("SMOKE_API_KEY"), "smoke: set SMOKE_API_KEY to run")
class TestSmokeRealAPI(unittest.TestCase):
    def test_round_trip(self):
        base_url = os.environ.get("SMOKE_BASE_URL") or "https://api.openai.com/v1"
        model = os.environ.get("SMOKE_MODEL") or "gpt-4o-mini"
        solver = MathSolver(api_key=os.environ["SMOKE_API_KEY"], base_url=base_url, model=model)
        r = solver.solve("2x + 3 = 11, solve for x")
        print("smoke:", {"answer": r.answer, "verified": r.verified, "retries": r.retries})
        self.assertTrue(r.verified)
        self.assertEqual(r.answer, 4)


if __name__ == "__main__":
    unittest.main()

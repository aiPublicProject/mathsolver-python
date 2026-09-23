"""HTTP-interface mock (no server, no sockets): patch urlopen so the client's
DEFAULT transport runs its real code path (URL building, headers, body
serialization, response parsing) against synthetic OpenAI-shaped responses.
Runs identically in local dev and GitHub CI."""
import json
import unittest
from unittest import mock

import mathsolver_help.solver as solver_mod
from mathsolver_help import MathSolver, SolverError

GOOD = json.dumps({
    "program": "let d = 11 - 3;\nlet x = d / 2;\nresult = x",
    "steps": ["Subtract 3: 2x = 8", "Divide by 2: x = 4"],
    "check": "2*{x} + 3 - 11",
})
WRONG_CHECK = json.dumps({
    "program": "let d = 11 - 3;\nresult = d / 2",
    "steps": ["..."],
    "check": "2*{x} + 3 - 12",  # evaluates to -1 -> must trigger a retry
})


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestHttpInterfaceMock(unittest.TestCase):
    def _run(self, script, fn):
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append({
                "url": req.full_url,
                "auth": req.headers.get("Authorization"),
                "body": json.loads(req.data),
            })
            nxt = script.pop(0) if script else GOOD
            if isinstance(nxt, dict) and "status" in nxt:
                raise solver_mod.SolverError("HTTP_ERROR", f"API responded {nxt['status']}")
            return _FakeResponse({"choices": [{"message": {"content": nxt}}]})

        with mock.patch.object(solver_mod, "urlopen", fake_urlopen):
            return fn(calls)

    def test_full_round_trip_via_default_transport(self):
        def scenario(calls):
            s = MathSolver(api_key="sk-mock", base_url="https://mock.test/v1", model="mock-model")
            r = s.solve("2x + 3 = 11, solve for x")
            self.assertTrue(r.verified)
            self.assertEqual(r.answer, 4)
            self.assertEqual(r.retries, 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["url"], "https://mock.test/v1/chat/completions")
            self.assertEqual(calls[0]["auth"], "Bearer sk-mock")
            self.assertEqual(calls[0]["body"]["model"], "mock-model")
            self.assertEqual(calls[0]["body"]["messages"][0]["role"], "system")
            self.assertIn("STRICT JSON", calls[0]["body"]["messages"][0]["content"])
            self.assertEqual(calls[0]["body"]["temperature"], 0)

        self._run([GOOD], scenario)

    def test_check_fail_retry(self):
        def scenario(calls):
            r = MathSolver(api_key="sk", base_url="https://mock.test/v1").solve("2x+3=11")
            self.assertTrue(r.verified)
            self.assertEqual(r.retries, 1)
            self.assertEqual(len(calls), 2)
            second = calls[1]["body"]["messages"]
            self.assertTrue(any("failed verification" in m["content"] for m in second))

        self._run([WRONG_CHECK, GOOD], scenario)

    def test_invalid_json_then_ok(self):
        def scenario(calls):
            r = MathSolver(api_key="sk", base_url="https://mock.test/v1").solve("1+1")
            self.assertTrue(r.verified)
            self.assertEqual(len(calls), 2)

        self._run(["certainly not json", GOOD], scenario)

    def test_http_500_no_retry(self):
        def scenario(calls):
            with self.assertRaises(SolverError) as ctx:
                MathSolver(api_key="sk", base_url="https://mock.test/v1").solve("1+1")
            self.assertEqual(ctx.exception.code, "HTTP_ERROR")
            self.assertEqual(len(calls), 1)

        self._run([{"status": 500}], scenario)

    def test_http_401(self):
        def scenario(_calls):
            with self.assertRaises(SolverError) as ctx:
                MathSolver(api_key="sk-bad", base_url="https://mock.test/v1").solve("1+1")
            self.assertEqual(ctx.exception.code, "HTTP_ERROR")

        self._run([{"status": 401}], scenario)


if __name__ == "__main__":
    unittest.main()

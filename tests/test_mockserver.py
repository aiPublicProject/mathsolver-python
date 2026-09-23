"""Full-stack mock: a local HTTP server imitating the OpenAI-compatible
/chat/completions endpoint. The client uses its DEFAULT transport (real
urllib) against it — URL building, auth header, request serialization,
response parsing and wire-level retries are all exercised. No real key."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mathsolver_help import MathSolver, SolverError

GOOD = json.dumps({
    "program": "let d = 11 - 3;\nlet x = d / 2;\nresult = x",
    "steps": ["Subtract 3: 2x = 8", "Divide by 2: x = 4"],
    "check": "2*{x} + 3 - 11",
})
WRONG_CHECK = json.dumps({
    "program": "let d = 11 - 3;\nresult = d / 2",
    "steps": ["..."],
    "check": "2*{x} + 3 - 12",  # evaluates to -1 -> must trigger a wire retry
})


class _MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        self.server.requests.append({
            "path": self.path,
            "auth": self.headers.get("Authorization"),
            "body": json.loads(raw),
        })
        nxt = self.server.script.pop(0) if self.server.script else GOOD
        if isinstance(nxt, dict) and "status" in nxt:
            self.send_error(nxt["status"])
            return
        payload = json.dumps({"choices": [{"message": {"content": nxt}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence
        pass


def make_server(script):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
    srv.script = list(script)
    srv.requests = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class TestMockServerRoundTrip(unittest.TestCase):
    def test_full_http_round_trip(self):
        srv = make_server([GOOD])
        try:
            base = f"http://127.0.0.1:{srv.server_port}/v1"
            solver = MathSolver(api_key="sk-mock", base_url=base, model="mock-model")
            r = solver.solve("2x + 3 = 11, solve for x")

            self.assertTrue(r.verified)
            self.assertEqual(r.answer, 4)
            self.assertEqual(r.retries, 0)

            # 请求侧: 经过真实 HTTP 传输后服务端收到了什么
            req = srv.requests[0]
            self.assertEqual(req["path"], "/v1/chat/completions")
            self.assertEqual(req["auth"], "Bearer sk-mock")
            self.assertEqual(req["body"]["model"], "mock-model")
            self.assertEqual(req["body"]["messages"][0]["role"], "system")
            self.assertIn("STRICT JSON", req["body"]["messages"][0]["content"])
            self.assertEqual(req["body"]["temperature"], 0)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_check_fail_retry_over_the_wire(self):
        srv = make_server([WRONG_CHECK, GOOD])
        try:
            base = f"http://127.0.0.1:{srv.server_port}/v1"
            r = MathSolver(api_key="sk", base_url=base).solve("2x+3=11")

            self.assertTrue(r.verified)
            self.assertEqual(r.retries, 1)
            self.assertEqual(len(srv.requests), 2)
            # 第二次请求携带了纠错上下文
            second = srv.requests[1]["body"]["messages"]
            self.assertTrue(any("failed verification" in m["content"] for m in second))
        finally:
            srv.shutdown()
            srv.server_close()

    def test_invalid_json_over_the_wire_then_ok(self):
        srv = make_server(["certainly not json", GOOD])
        try:
            base = f"http://127.0.0.1:{srv.server_port}/v1"
            r = MathSolver(api_key="sk", base_url=base).solve("1+1")
            self.assertTrue(r.verified)
            self.assertEqual(len(srv.requests), 2)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_http_500_http_error_no_retry(self):
        srv = make_server([{"status": 500}])
        try:
            base = f"http://127.0.0.1:{srv.server_port}/v1"
            with self.assertRaises(SolverError) as ctx:
                MathSolver(api_key="sk", base_url=base).solve("1+1")
            self.assertEqual(ctx.exception.code, "HTTP_ERROR")
            self.assertEqual(len(srv.requests), 1)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_http_401_http_error(self):
        srv = make_server([{"status": 401}])
        try:
            base = f"http://127.0.0.1:{srv.server_port}/v1"
            with self.assertRaises(SolverError) as ctx:
                MathSolver(api_key="sk-bad", base_url=base).solve("1+1")
            self.assertEqual(ctx.exception.code, "HTTP_ERROR")
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()

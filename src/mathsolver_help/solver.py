"""Core solver: BYOK call to an OpenAI-compatible endpoint + local verification."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from urllib.request import Request, urlopen

SYSTEM_PROMPT = "\n".join([
    "You are a precise math solver.",
    "Reply with STRICT JSON only, no markdown fences, in this exact shape:",
    '{"answer": <number>, "steps": [<string>, ...], "verification": {"expression": "<string>"}}',
    "Rules:",
    '- "answer" must be a single number (the final result).',
    '- "steps" must be an array of short plain-language explanation strings.',
    '- "verification.expression" must be a pure arithmetic expression that',
    "  evaluates to the answer. Allowed: numbers, + - * / % ^ ( ), and the",
    "  functions abs sqrt sin cos tan ln log exp floor ceil round min max",
    "  (log is base 10, ln is natural), and the constants pi and e.",
    "- The expression must recompute the answer independently.",
])


class SolverError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------
# Expression evaluator (recursive descent, no eval, stdlib only)
# ---------------------------------------------------------------------

_FUNCS = {
    "abs": abs, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "ln": math.log, "log": math.log10, "exp": math.exp,
    "floor": math.floor, "ceil": math.ceil, "round": round,
    "min": min, "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e}

_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+)"
    r"|(?P<id>[a-zA-Z_][a-zA-Z_0-9]*)"
    r"|(?P<op>[-+*/%^(),]))"
)


def _tokenize(src: str):
    tokens, pos = [], 0
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m or m.end() == pos:
            if src[pos:].strip() == "":
                break
            raise SolverError("EXPR_BAD_CHAR", f"unexpected character at {pos}")
        pos = m.end()
        if m.group("num") is not None:
            tokens.append(("num", float(m.group("num"))))
        elif m.group("id") is not None:
            tokens.append(("id", m.group("id")))
        else:
            tokens.append((m.group("op"), None))
    return tokens


def eval_expression(src: str) -> float:
    """Evaluate a pure arithmetic expression string. Raises SolverError on anything else."""
    if not isinstance(src, str) or not src.strip():
        raise SolverError("EXPR_EMPTY", "empty expression")
    tokens = _tokenize(src)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def eat(kind: Optional[str] = None):
        nonlocal pos
        tok = peek()
        if tok is None or (kind is not None and tok[0] != kind):
            raise SolverError("EXPR_SYNTAX", f"expected {kind or 'more tokens'}")
        pos += 1
        return tok

    def parse_expr() -> float:
        v = parse_term()
        while peek() and peek()[0] in "+-":
            op = eat()[0]
            r = parse_term()
            v = v + r if op == "+" else v - r
        return v

    def parse_term() -> float:
        v = parse_unary()
        while peek() and peek()[0] in "*/%":
            op = eat()[0]
            r = parse_unary()
            v = v * r if op == "*" else (v / r if op == "/" else math.fmod(v, r))
        return v

    def parse_unary() -> float:
        if peek() and peek()[0] == "-":
            eat()
            return -parse_unary()
        if peek() and peek()[0] == "+":
            eat()
            return parse_unary()
        return parse_power()

    def parse_power() -> float:
        base = parse_atom()
        if peek() and peek()[0] == "^":
            eat("^")
            return math.pow(base, parse_unary())  # right associative
        return base

    def parse_atom() -> float:
        tok = peek()
        if tok is None:
            raise SolverError("EXPR_SYNTAX", "unexpected end of expression")
        kind = tok[0]
        if kind == "num":
            return eat()[1]
        if kind == "id":
            name = eat()[1].lower()
            if peek() and peek()[0] == "(":
                eat("(")
                args = [parse_expr()]
                while peek() and peek()[0] == ",":
                    eat(",")
                    args.append(parse_expr())
                eat(")")
                fn = _FUNCS.get(name)
                if fn is None:
                    raise SolverError("EXPR_UNKNOWN_FUNC", f"unknown function {name}")
                return float(fn(*args))
            if name in _CONSTS:
                return _CONSTS[name]
            raise SolverError("EXPR_UNKNOWN_ID", f"unknown identifier {name}")
        if kind == "(":
            eat("(")
            v = parse_expr()
            eat(")")
            return v
        raise SolverError("EXPR_SYNTAX", f"unexpected token {kind}")

    value = parse_expr()
    if pos != len(tokens):
        raise SolverError("EXPR_TRAILING", "trailing tokens in expression")
    if not math.isfinite(value):
        raise SolverError("EXPR_NON_FINITE", "expression evaluated to non-finite value")
    return value


# ---------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------

@dataclass
class SolverResult:
    answer: float
    steps: List[str] = field(default_factory=list)
    expression: str = ""
    evaluated: Optional[float] = None
    verified: bool = False
    retries: int = 0


def _parse_solver_json(text: str) -> dict:
    m = re.search(r"\{[\s\S]*\}", str(text))
    if not m:
        raise SolverError("INVALID_JSON", "model reply contained no JSON object")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        raise SolverError("INVALID_JSON", "model reply was not valid JSON")
    answer = data.get("answer")
    if isinstance(answer, str):
        m2 = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", answer)
        answer = float(m2.group(0)) if m2 else None
    if not isinstance(answer, (int, float)):
        raise SolverError("INVALID_JSON", "model JSON is missing a numeric answer")
    expr = (data.get("verification") or {}).get("expression")
    if not isinstance(expr, str):
        raise SolverError("INVALID_JSON", "model JSON is missing verification.expression")
    steps = data.get("steps")
    return {"answer": float(answer), "steps": [str(s) for s in steps] if isinstance(steps, list) else [],
            "expression": expr}


def _numerically_equal(a: float, b: float, rel_tol: float = 1e-6) -> bool:
    return abs(a - b) <= rel_tol * max(1.0, abs(a), abs(b))


# ---------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------

def _default_transport(url: str, body: dict, api_key: str) -> str:
    req = Request(url, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
    }, method="POST")
    try:
        with urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001 — surface any transport failure uniformly
        raise SolverError("HTTP_ERROR", f"API call failed: {exc}") from exc
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SolverError("HTTP_ERROR", "API response missing message content") from exc


# ---------------------------------------------------------------------
# MathSolver client (instantiate once, solve many)
# ---------------------------------------------------------------------

class MathSolver:
    """BYOK client for an OpenAI-compatible endpoint.

    Usage::

        from mathsolver_help import MathSolver

        solver = MathSolver(api_key="sk-...", base_url="https://api.openai.com/v1")
        result = solver.solve("2x + 3 = 11, solve for x")
        # OpenAI-compatible alternatives: DeepSeek, Groq, Moonshot, local Ollama/vLLM, ...

    The answer is only ``verified=True`` when the model's verification
    expression independently re-evaluates (locally) to the same number.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4o-mini", timeout: int = 60,
                 transport: Optional[Callable[[str, dict, str], str]] = None):
        if not api_key:
            raise SolverError("NO_API_KEY", "api_key is required (BYOK: bring your own key)")
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise SolverError("BAD_BASE_URL", "base_url must be an http(s) URL, e.g. https://api.deepseek.com/v1")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._transport = transport

    def solve(self, problem: str) -> SolverResult:
        if not isinstance(problem, str) or not problem.strip():
            raise SolverError("NO_PROBLEM", "problem must be a non-empty string")
        transport = self._transport or _default_transport
        url = f"{self.base_url}/chat/completions"
        api_key, model = self.api_key, self.model

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        call = lambda: transport(url, {"model": model, "messages": messages, "temperature": 0}, api_key)  # noqa: E731

        try:
            parsed = _parse_solver_json(call())
        except SolverError as err:
            if err.code != "INVALID_JSON":
                raise
            messages.append({"role": "assistant", "content": "invalid JSON"})
            messages.append({"role": "user", "content": "Your reply was not valid JSON. Reply again with the exact strict JSON shape."})
            parsed = _parse_solver_json(call())  # second failure throws

        def attempt(p: dict) -> tuple:
            try:
                ev = eval_expression(p["expression"])
            except SolverError:
                return None, False
            return ev, _numerically_equal(ev, p["answer"])

        evaluated, verified = attempt(parsed)
        retries = 0
        if not verified:
            retries = 1
            messages.append({"role": "assistant", "content": json.dumps(parsed)})
            messages.append({"role": "user", "content":
                             f"Your verification expression evaluated to {evaluated if evaluated is not None else 'an error'}, "
                             f"which does not match your answer {parsed['answer']}. "
                             "Re-derive the problem carefully and reply again with the same strict JSON shape."})
            try:
                second = _parse_solver_json(call())
                ev2, ok2 = attempt(second)
                if ev2 is not None:
                    evaluated = ev2
                if ok2:
                    parsed, verified = second, True
            except SolverError:
                pass  # keep first attempt; verified stays False

        return SolverResult(answer=parsed["answer"], steps=parsed["steps"], expression=parsed["expression"],
                            evaluated=evaluated, verified=verified, retries=retries)

"""Core solver: BYOK call to an OpenAI-compatible endpoint + execution-based verification.

v0.2 (PAL-style): the model never states the answer. It returns a small
JavaScript-like PROGRAM; this package executes the program deterministically
and the execution output IS the answer. For equations, a CHECK expression
({x} placeholder) must evaluate to 0 when the computed answer is substituted
back into the original equation.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
from urllib.request import Request, urlopen

SYSTEM_PROMPT = "\n".join([
    "You are a precise math solver.",
    "Reply with STRICT JSON only, no markdown fences, in this exact shape:",
    '{"program": "<string>", "steps": [<string>, ...], "check": "<string>"}',
    "Rules:",
    '- "program" is a small JavaScript-like program that computes the final answer.',
    "  One statement per line (or ; separated). Allowed statements:",
    "      let NAME = EXPRESSION",
    "      result = EXPRESSION",
    "  EXPRESSIONs may use numbers, + - * / % ^ ( ), the functions",
    "  abs sqrt sin cos tan ln log exp floor ceil round min max",
    "  (log is base 10, ln is natural), the constants pi and e, and any",
    "  variable defined by an earlier let. The value assigned to result",
    "  is the answer. Never state the answer as a number in text.",
    '- "steps" is an array of short plain-language explanation strings.',
    '- "check" is a verification expression containing the placeholder {x}.',
    "  After solving, {x} is replaced by the computed answer and the whole",
    "  expression must evaluate to 0.",
    "  For equations, substitute the answer back into the original equation",
    '  (e.g. 2x+3=11 -> "2*{x}+3-11").',
    "  For arithmetic, recompute via a different path and subtract the answer",
    '  (e.g. 15% of 80 -> "80*15/100-{x}"). Provide "check" whenever possible.',
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


def eval_expression(src: str, env: Optional[dict] = None) -> float:
    """Evaluate a pure arithmetic expression string. Raises SolverError on anything else."""
    if not isinstance(src, str) or not src.strip():
        raise SolverError("EXPR_EMPTY", "empty expression")
    env = env or {}
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
            raw = eat()[1]
            if raw in env:
                return float(env[raw])
            name = raw.lower()
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
# Program interpreter + check
# ---------------------------------------------------------------------

@dataclass
class SolverResult:
    answer: float
    steps: List[str] = field(default_factory=list)
    program: str = ""
    check: Optional[str] = None
    check_value: Optional[float] = None
    verified: bool = False
    retries: int = 0


def run_program(src: str) -> float:
    """Execute a model-generated JS-dialect program (let / assignment / result)."""
    if not isinstance(src, str) or not src.strip():
        raise SolverError("PROGRAM_EMPTY", "empty program")
    env: dict = {}
    result_defined = False
    last_value = None
    lines = [ln.strip() for ln in re.split(r"[\n;]+", src) if ln.strip()]
    if not lines:
        raise SolverError("PROGRAM_EMPTY", "empty program")
    for line in lines:
        m = re.match(r"^let\s+([a-zA-Z_]\w*)\s*=\s*([\s\S]+)$", line)
        if m:
            env[m.group(1)] = eval_expression(m.group(2), env)
            if m.group(1) == "result":
                result_defined = True
            continue
        m = re.match(r"^([a-zA-Z_]\w*)\s*=\s*([\s\S]+)$", line)
        if m:
            env[m.group(1)] = eval_expression(m.group(2), env)
            if m.group(1) == "result":
                result_defined = True
            continue
        last_value = eval_expression(line, env)
    if result_defined:
        return float(env["result"])
    if last_value is not None:
        return float(last_value)
    raise SolverError("PROGRAM_NO_RESULT", "program produced no result")


def run_check(check_src: str, answer: float) -> Tuple[float, bool]:
    """Substitute {x} with the computed answer; passes when value ~ 0."""
    substituted = re.sub(r"\{\s*x\s*\}", "(" + repr(answer) + ")", str(check_src), flags=re.I)
    value = eval_expression(substituted)
    passed = abs(value) <= 1e-6 * max(1.0, abs(answer))
    return value, passed


# ---------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------

def _parse_solver_json(text: str) -> dict:
    m = re.search(r"\{[\s\S]*\}", str(text))
    if not m:
        raise SolverError("INVALID_JSON", "model reply contained no JSON object")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        raise SolverError("INVALID_JSON", "model reply was not valid JSON")
    program = data.get("program")
    if not isinstance(program, str):
        raise SolverError("INVALID_JSON", "model JSON is missing program")
    check = data.get("check")
    steps = data.get("steps")
    return {
        "program": program,
        "steps": [str(s) for s in steps] if isinstance(steps, list) else [],
        "check": check if isinstance(check, str) and check.strip() else None,
    }


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
# Client (instantiate once, solve many)
# ---------------------------------------------------------------------

class MathSolver:
    """BYOK client for an OpenAI-compatible endpoint.

    Usage::

        from mathsolver_help import MathSolver

        solver = MathSolver(api_key="sk-...", base_url="https://api.openai.com/v1")
        result = solver.solve("2x + 3 = 11, solve for x")
        # result.answer comes from executing the model-generated program
        # locally — never from a number the model stated.
        # result.verified is True only when the check expression passes
        # (equations: answer substituted back satisfies the original equation).
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

        def call() -> str:
            return transport(url, {"model": model, "messages": messages, "temperature": 0}, api_key)

        try:
            parsed = _parse_solver_json(call())
        except SolverError as err:
            if err.code != "INVALID_JSON":
                raise
            messages.append({"role": "assistant", "content": "invalid JSON"})
            messages.append({"role": "user", "content": "Your reply was not valid JSON. Reply again with the exact strict JSON shape."})
            parsed = _parse_solver_json(call())  # second failure throws

        def attempt(p: dict) -> dict:
            try:
                answer = run_program(p["program"])
                check_value = None
                verified = False
                if p["check"]:
                    check_value, verified = run_check(p["check"], answer)
                return {"ok": True, "answer": answer, "check_value": check_value, "verified": verified}
            except SolverError as err:
                return {"ok": False, "error": err}

        outcome = attempt(parsed)
        retries = 0
        if not outcome["ok"] or not outcome["verified"]:
            retries = 1
            reason = (
                f"program failed to execute ({outcome['error'].code}: {outcome['error']})"
                if not outcome["ok"]
                else f"check evaluated to {outcome['check_value']} instead of 0"
            )
            messages.append({"role": "assistant", "content": json.dumps(parsed)})
            messages.append({"role": "user", "content":
                             f"Your submission failed verification: {reason}. "
                             "Re-derive the problem carefully and reply again with the same strict JSON shape."})
            parsed = _parse_solver_json(call())
            outcome = attempt(parsed)
            if not outcome["ok"]:
                raise outcome["error"]  # PROGRAM_* persisted after retry

        return SolverResult(answer=outcome["answer"], steps=parsed["steps"], program=parsed["program"],
                            check=parsed["check"], check_value=outcome["check_value"],
                            verified=outcome["verified"], retries=retries)

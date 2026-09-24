# mathsolver-python

BYOK AI math solver with execution-based verification (PAL-style) — bring your own OpenAI-compatible API key; the answer is computed locally by executing a model-generated program, never taken from a number the model stated.

## How it works

```
// instantiate once with your own credentials, solve many
solver = new Client(apiKey, baseUrl?, model?)
result = solver.solve(problem)
  -> { answer, steps[], program, check, checkValue (check_value), verified, retries }
```

1. Calls your OpenAI-compatible endpoint (any base_url) with a strict-JSON prompt.
2. The model returns a small **program** (a JavaScript-like arithmetic dialect), explanation steps, and a **check** expression containing `{x}` — the model never states the answer as a number.
3. The package **executes the program locally** with its own interpreter (recursive-descent parser, no eval, no model code execution). The execution output IS the answer.
4. The check expression is evaluated with the computed answer substituted for `{x}` and must equal ~0 — for equations, substitute the answer back into the original equation (e.g. `2x+3=11` -> `2*{x}+3-11`); for arithmetic, recompute via a different path and subtract the answer.
5. Program error or failed check -> automatic corrective retry (max 1). `verified: true` means the check passed; `verified: false` means it never did — treat those answers with care.

## Verification promise (honest scope)

- ✅ the answer is produced by local program execution — the model has no way to "declare" a result.
- ✅ `verified: true` means an independent check expression evaluated to ~0.
- ❌ proof steps / textual reasoning are model-generated and NOT step-verified.
- ❌ if the model misreads the problem, a self-consistent program+check can still be wrong.

## Test

See `.github/workflows/test.yml` (CI runs on every push). Tests cover the expression evaluator (precedence, env variables), the program interpreter (let / result / bare expressions), check substitution, answer-comes-from-execution (model JSON contains no answer field), check-fail retry-recover, program-error retry, PROGRAM_* persistent error, invalid-JSON retry, double-invalid error, HTTP error no-retry, NO_API_KEY guard — plus an HTTP-interface mock (no server, no sockets) that runs the default transport's real code path and asserts URL, headers, body, temperature and the corrective retry payload.

Maintained by [mathsolver.help](https://mathsolver.help) — free step-by-step math solver, try it online.

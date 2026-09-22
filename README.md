# mathsolver-python

BYOK AI math solver with independent verification — bring your own OpenAI-compatible API key, get answers that are verified by local expression evaluation before they reach you.

## How it works

```
// instantiate once with your own credentials, solve many
solver = new Client(apiKey, baseUrl?, model?)
result = solver.solve(problem)
  → { answer, steps[], expression, evaluated, verified, retries }
```

1. Calls your OpenAI-compatible endpoint (any base_url) with a strict-JSON prompt.
2. The model returns an answer, explanation steps, and a **verification expression** (pure arithmetic).
3. The package evaluates that expression **locally** with its own recursive-descent parser — no eval, no model code execution.
4. Mismatch → automatic corrective retry (max 1). `verified: false` means the two never agreed; treat those answers with care.

## Verification promise (honest scope)

- ✅ Numeric answers are verified by independent local evaluation.
- ❌ Proof steps / textual reasoning are model-generated and NOT step-verified.

## Test

See `.github/workflows/test.yml` (CI runs on every push). Mock-transport unit tests cover: first-try verified, mismatch-retry-recover, invalid-JSON retry, double-invalid error, HTTP error no-retry, NO_API_KEY guard, still-wrong unverified.

Maintained by [mathsolver.help](https://mathsolver.help) — free step-by-step math solver, try it online.

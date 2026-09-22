"""mathsolver-help — BYOK AI math solver with independent verification.

An answer is only marked ``verified=True`` when the model's verification
expression (pure arithmetic) is evaluated locally and matches the answer.
No model output is ever executed as code.
"""
from .solver import MathSolver, eval_expression, SolverError

__all__ = ["MathSolver", "eval_expression", "SolverError"]
__version__ = "0.1.0"

"""CCS_FNE Bundle Lagrangian solver. Importing this package performs no I/O."""


def solve_lr(*args, **kwargs):
    from .solver import solve_lr as solve
    return solve(*args, **kwargs)


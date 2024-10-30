def compute_f1(pr: float, rc: float) -> float:
    return 2 * (pr * rc) / (pr + rc + 1e-10)

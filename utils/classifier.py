# Score breakpoints shared with utils/display.get_color — keep them in sync
# by deriving color from the label below, not from a second copy of the score.
_BANDS = [
    (80, "ATENTO"),
    (60, "MULTIPLAS TAREFAS"),
    (40, "DESATENCAO"),
    (20, "RISCO"),
]
DEFAULT_STATE = "CRITICO"


def classify_from_score(score: float) -> str:
    for threshold, label in _BANDS:
        if score > threshold:
            return label
    return DEFAULT_STATE

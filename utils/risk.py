from utils.constants import WEIGHTS

def compute_score(events):
    score = 100

    for key, active in events.items():
        if active:
            score -= WEIGHTS.get(key, 0)

    return max(score, 0)

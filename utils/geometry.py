import numpy as np

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def normalized_distance(p1, p2, ref1, ref2):
    d = distance(p1, p2)
    d_ref = distance(ref1, ref2)

    if d_ref == 0:
        return 0

    return d / d_ref

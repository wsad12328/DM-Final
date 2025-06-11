# evaluate.py
import numpy as np

def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        score = 0.0
        num_hits = 0.0
        for i, pred_label in enumerate(p[:k]):
            if pred_label == a and pred_label not in p[:i]:
                num_hits += 1.0
                score += num_hits / (i+1.0)
                break
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

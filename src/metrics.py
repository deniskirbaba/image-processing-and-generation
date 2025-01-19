from typing import Tuple

import numpy as np

def tpr_tnr(true: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    assert true.shape == pred.shape
    tp = (true == 1) & (pred == 1)
    fp = (true == 0) & (pred == 1)
    tn = (true == 0) & (pred == 0)
    fn = (true == 1) & (pred == 0)

    tpr = float(sum(tp) / (sum(tp) + sum(fn)))
    tnr = float(sum(tn) / (sum(tn) + sum(fp)))

    return tpr, tnr

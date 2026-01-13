
import numpy as np
import torch
import random
from sklearn.metrics import precision_score, recall_score, f1_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sample_accuracy(y_true_np: np.ndarray,
                     y_pred_np: np.ndarray) -> float:

    assert y_true_np.shape == y_pred_np.shape
    n_samples = y_true_np.shape[0]

    count = 0.0
    for i in range(n_samples):
        yt = y_true_np[i].astype(bool)
        yp = y_pred_np[i].astype(bool)

        inter = np.logical_and(yt, yp).sum()   # |Yi ∩ Ŷi|
        union = np.logical_or(yt, yp).sum()    # |Yi ∪ Ŷi|

        if union == 0:
            continue
        count += inter / union

    acc = count / n_samples
    return float(acc)


def multilabel_metrics(y_true: torch.Tensor,
                       y_pred_prob: torch.Tensor,
                       threshold: float = 0.5):

    ts = y_true.detach().cpu().numpy().astype(int)
    zs = y_pred_prob.detach().cpu().numpy()
    preds = (zs >= threshold).astype(int)

    # sample-based accuracy
    acc = _sample_accuracy(ts, preds)

    # sample-based precision / recall / F1
    precision = precision_score(ts, preds, average="samples", zero_division=0)
    recall = recall_score(ts, preds, average="samples", zero_division=0)
    f1score = f1_score(ts, preds, average="samples", zero_division=0)

    return {
        "acc": float(acc),
        "prec": float(precision),
        "rec": float(recall),
        "f1": float(f1score),
    }

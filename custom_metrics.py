
from sklearn.metrics import f1_score
from pytorch_tabnet.metrics import Metric

class F1Macro(Metric):
    def __init__(self):
        self._name = "f1_macro"
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_pred = (y_score[:, 1] > 0.5).astype(int)  # binary classification
        f1_macro = f1_score(y_true, y_pred, average='macro')
        return f1_macro

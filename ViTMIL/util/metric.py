from sklearn.metrics import roc_auc_score, f1_score
import numpy as np


def calculate_auroc(is_binary, valid_labels, valid_preds):
    if is_binary:
        auc = roc_auc_score(valid_labels, valid_preds)
    else:
        auc = roc_auc_score(valid_labels, valid_preds, multi_class="ovr")

    return auc


def calculate_f1_score(is_binary, valid_labels, valid_preds):
    if is_binary:
        y_pred_labels = (valid_preds > 0.5).astype(int)
        f1 = f1_score(valid_labels, y_pred_labels)
    else:
        y_pred_labels = np.argmax(valid_preds, axis=1)
        f1 = f1_score(valid_labels, y_pred_labels, average='weighted')

    return f1

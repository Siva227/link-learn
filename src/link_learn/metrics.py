import numpy as np
from sklearn.metrics import auc
from sklearn.utils import assert_all_finite, check_consistent_length, column_or_1d
from sklearn.utils.multiclass import type_of_target


def magnified_roc_curve(y_true, y_score, pos_label=None):
    """
    Calculate Magnified ROC curve as specified in
    `Muscoloni, A.; Cannistraci, C.V.
    Early Retrieval Problem and Link Prediction Evaluation via the Area Under the Magnified ROC.
    Preprints 2022, 2022090277. <https://doi.org/10.20944/preprints202209.0277.v1>`_

    Args:
        y_true:
            True binary labels. If labels are not either {-1, 1} or {0, 1}, then
            pos_label should be explicitly given.

        y_score:
            Target scores, can either be probability estimates of the positive
            class, confidence values, or non-thresholded measure of decisions
            (as returned by "decision_function" on some classifiers).

        pos_label:
            The label of the positive class.

    Returns:
        Returns a tuple (fpr_m, tpr_m_norm, thresholds) where -

        - fpr_m: Increasing false positive rates such that element `i` \
        is the false positive rate of predictions with score >= `thresholds[i]`.

        - tpr_m_norm: Increasing true positive rates such that element `i` \
        is the true positive rate of predictions with score >= `thresholds[i]`.

        - thresholds: Decreasing thresholds on the decision function used to compute fpr and tpr.\
        `thresholds[0]` represents no instances being predicted and is arbitrarily set to `np.inf`.

    """
    y_type = type_of_target(y_true, input_name="y_true")
    if y_type != "binary":
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    pos_label = 1 if pos_label is None else pos_label

    # make y_true a boolean vector
    y_true = y_true == pos_label
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)
    fps = np.cumsum((1 - y_true))
    num_trues = tps[-1]
    num_falses = fps[-1]
    if num_trues <= 0:
        raise ValueError("No positive samples in y_true")
    if num_falses <= 0:
        raise ValueError("No negative samples in y_true")
    tps_rand = fps * (num_trues / num_falses)

    # adjusted roc
    tpr_m = np.log1p(tps) / np.log1p(num_trues)
    fpr_m = np.log1p(fps) / np.log1p(num_falses)
    tpr_m_rand = np.log1p(tps_rand) / np.log1p(num_trues)
    # Calculate magnified roc
    tpr_m_norm = (tpr_m - tpr_m_rand) / (1 - tpr_m_rand) * (1 - fpr_m) + fpr_m
    tpr_m_norm = np.nan_to_num(tpr_m_norm, nan=1.0)
    fpr_m = np.r_[0, fpr_m[threshold_idxs]]
    tpr_m_norm = np.r_[0, tpr_m_norm[threshold_idxs]]
    thresholds = np.r_[np.inf, y_score[threshold_idxs]]
    return fpr_m, tpr_m_norm, thresholds


def magnified_roc_auc_score(y_true, y_score, pos_label=None):
    """Compute Area Under the Magnified Receiver Operating Characteristic Curve (ROC AUC) \
    from prediction scores.

    `Muscoloni, A.; Cannistraci, C.V.
    Early Retrieval Problem and Link Prediction Evaluation via the Area Under the Magnified ROC.
    Preprints 2022, 2022090277. <https://doi.org/10.20944/preprints202209.0277.v1>`_

    Args:
        y_true:
            True binary labels. If labels are not either {-1, 1} or {0, 1}, then
            pos_label should be explicitly given.

        y_score:
            Target scores, can either be probability estimates of the positive
            class, confidence values, or non-thresholded measure of decisions
            (as returned by "decision_function" on some classifiers).

        pos_label:
            The label of the positive class.

    Returns:
        Area under the magnified ROC curve
    """
    fpr_m, tpr_m_norm, _ = magnified_roc_curve(y_true, y_score, pos_label)
    return auc(fpr_m, tpr_m_norm)

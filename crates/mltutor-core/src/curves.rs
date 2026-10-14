//! ROC and precision–recall curves (fase2-regresion.md C4), with the semantics of
//! scikit-learn 1.7:
//!
//! * [`roc_curve`]: scores sorted in decreasing order, one point per distinct
//!   score, `drop_intermediate=True` (collinear points removed) and a first
//!   threshold of `+∞` at `(0, 0)`.
//! * [`precision_recall_curve`]: points in increasing threshold order, plus the
//!   final point `precision = 1, recall = 0` without threshold.
//! * [`auc`] (trapezoids, `np.trapz`) and [`average_precision`]
//!   (`−Σ (Rₙ − Rₙ₋₁)·Pₙ`).
//! * [`binary_curves`] and [`multiclass_curves`]: what `viz/roc.py` draws
//!   (one-vs-rest per class, micro = everything flattened, macro = mean of the
//!   per-class TPRs interpolated on the union of FPRs) plus the weighted AUC/AP that
//!   Trees and KNN may use (D20).
//!
//! Sums follow NumPy's pairwise order, so AUCs and APs agree with Python to the
//! last bit when the scores are the same.

use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};

use crate::stats::pairwise_sum;

/// A curve: `x`/`y` points and, for ROC and PR curves, the threshold of each point
/// (`+∞` for the first ROC point; the last PR point has none, so
/// `thresholds.len() == x.len() - 1` for PR curves).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Curve {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub thresholds: Vec<f64>,
}

/// `_binary_clf_curve`: false and true positives at each distinct score
/// (decreasing), and those scores.
fn binary_clf_curve(y: &[bool], score: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = y.len().min(score.len());
    // np.argsort(score, kind="mergesort")[::-1]: decreasing scores; ties do not
    // matter because only the last index of each run of equal scores is used.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| score[a].total_cmp(&score[b]));
    order.reverse();
    let mut fps = Vec::new();
    let mut tps = Vec::new();
    let mut thresholds = Vec::new();
    let mut cum_tp = 0.0;
    for (pos, &i) in order.iter().enumerate() {
        if y[i] {
            cum_tp += 1.0;
        }
        let last = pos + 1 == n;
        if last || score[order[pos + 1]] != score[i] {
            tps.push(cum_tp);
            fps.push(1.0 + pos as f64 - cum_tp);
            thresholds.push(score[i]);
        }
    }
    (fps, tps, thresholds)
}

/// `roc_curve(y, score)` (`drop_intermediate=True`): `x` = FPR, `y` = TPR.
/// With no negatives (or positives) the FPR (TPR) is NaN, as in scikit-learn.
pub fn roc_curve(y: &[bool], score: &[f64]) -> Curve {
    let (mut fps, mut tps, mut thr) = binary_clf_curve(y, score);
    if fps.len() > 2 {
        let n = fps.len();
        let keep: Vec<bool> = (0..n)
            .map(|i| {
                if i == 0 || i == n - 1 {
                    return true;
                }
                // np.diff(a, 2)[i-1] = (a[i+1] − a[i]) − (a[i] − a[i−1]).
                let d2 = |a: &[f64]| (a[i + 1] - a[i]) - (a[i] - a[i - 1]);
                d2(&fps) != 0.0 || d2(&tps) != 0.0
            })
            .collect();
        let filter = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .zip(&keep)
                .filter(|(_, &k)| k)
                .map(|(&x, _)| x)
                .collect()
        };
        fps = filter(&fps);
        tps = filter(&tps);
        thr = filter(&thr);
    }
    fps.insert(0, 0.0);
    tps.insert(0, 0.0);
    thr.insert(0, f64::INFINITY);
    let (fp_last, tp_last) = (
        fps.last().copied().unwrap_or(0.0),
        tps.last().copied().unwrap_or(0.0),
    );
    let fpr = fps
        .iter()
        .map(|&f| if fp_last > 0.0 { f / fp_last } else { f64::NAN })
        .collect();
    let tpr = tps
        .iter()
        .map(|&t| if tp_last > 0.0 { t / tp_last } else { f64::NAN })
        .collect();
    Curve {
        x: fpr,
        y: tpr,
        thresholds: thr,
    }
}

/// `precision_recall_curve(y, score)`: `x` = recall, `y` = precision, thresholds
/// increasing. Without positives the recall is 1 everywhere, as in scikit-learn.
pub fn precision_recall_curve(y: &[bool], score: &[f64]) -> Curve {
    let (fps, tps, thr) = binary_clf_curve(y, score);
    let tp_last = tps.last().copied().unwrap_or(0.0);
    let mut precision: Vec<f64> = tps
        .iter()
        .zip(&fps)
        .map(|(&t, &f)| if t + f != 0.0 { t / (t + f) } else { 0.0 })
        .collect();
    let mut recall: Vec<f64> = tps
        .iter()
        .map(|&t| if tp_last == 0.0 { 1.0 } else { t / tp_last })
        .collect();
    precision.reverse();
    recall.reverse();
    precision.push(1.0);
    recall.push(0.0);
    let mut thresholds = thr;
    thresholds.reverse();
    Curve {
        x: recall,
        y: precision,
        thresholds,
    }
}

/// `sklearn.metrics.auc(x, y)`: trapezoidal rule (`np.trapz`) on a monotonic `x`;
/// NaN if `x` is not monotonic or has fewer than 2 points.
pub fn auc(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return f64::NAN;
    }
    let dx: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    let direction = if dx.iter().all(|&d| d >= 0.0) {
        1.0
    } else if dx.iter().all(|&d| d <= 0.0) {
        -1.0
    } else {
        return f64::NAN;
    };
    let terms: Vec<f64> = (0..n - 1)
        .map(|i| dx[i] * (y[i + 1] + y[i]) / 2.0)
        .collect();
    direction * pairwise_sum(&terms)
}

/// `average_precision_score(y, score)` (binary): `−Σ diff(recall)·precision[:-1]`.
pub fn average_precision(y: &[bool], score: &[f64]) -> f64 {
    let pr = precision_recall_curve(y, score);
    let terms: Vec<f64> = (0..pr.x.len().saturating_sub(1))
        .map(|i| (pr.x[i + 1] - pr.x[i]) * pr.y[i])
        .collect();
    -pairwise_sum(&terms)
}

/// `np.interp(x, xp, fp)` for increasing `xp` (ties allowed, as in ROC curves:
/// the last of equal `xp` wins, like NumPy's binary search).
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len().min(fp.len());
    if n == 0 {
        return f64::NAN;
    }
    if x < xp[0] {
        return fp[0];
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    let j = xp[..n].partition_point(|&v| v <= x).saturating_sub(1);
    if j >= n - 1 || xp[j] == x {
        return fp[j.min(n - 1)];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let v = slope * (x - xp[j]) + fp[j];
    if v.is_nan() {
        let v2 = slope * (x - xp[j + 1]) + fp[j + 1];
        if v2.is_nan() && fp[j] == fp[j + 1] {
            return fp[j];
        }
        return v2;
    }
    v
}

/// ROC and PR curves of one class (or of an average), with AUC and AP.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassCurves {
    /// Class code (`None` for micro).
    pub class: Option<usize>,
    pub roc: Curve,
    pub auc: f64,
    pub pr: Curve,
    pub ap: f64,
}

fn class_curves(class: Option<usize>, y: &[bool], score: &[f64]) -> ClassCurves {
    let roc = roc_curve(y, score);
    let auc_v = auc(&roc.x, &roc.y);
    let pr = precision_recall_curve(y, score);
    let ap = average_precision(y, score);
    ClassCurves {
        class,
        roc,
        auc: auc_v,
        pr,
        ap,
    }
}

/// The binary view: curves of the positive class and the PR baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryCurves {
    /// Code of the positive class (`classes_[1]`).
    pub positive_class: usize,
    pub roc: Curve,
    pub auc: f64,
    pub pr: Curve,
    pub ap: f64,
    /// Proportion of positives in the test set (`np.sum(y_test) / len(y_test)`).
    pub baseline: f64,
}

/// `plot_roc_curve` for a binary model: `score` = probability of
/// `positive_class` (`predict_proba[:, 1]`).
pub fn binary_curves(y_true: &[usize], score: &[f64], positive_class: usize) -> BinaryCurves {
    let y: Vec<bool> = y_true.iter().map(|&c| c == positive_class).collect();
    let c = class_curves(Some(positive_class), &y, score);
    let n_pos = y.iter().filter(|&&b| b).count();
    BinaryCurves {
        positive_class,
        roc: c.roc,
        auc: c.auc,
        pr: c.pr,
        ap: c.ap,
        baseline: if y.is_empty() {
            f64::NAN
        } else {
            n_pos as f64 / y.len() as f64
        },
    }
}

/// The multiclass view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MulticlassCurves {
    /// One-vs-rest curves of each model class present in the test set.
    pub per_class: Vec<ClassCurves>,
    /// All the one-vs-rest problems flattened (`y_bin.ravel()`, `proba.ravel()`),
    /// over the classes **present in the test set** only. Decision (D20, as the
    /// generated script does with `Y_test[:, presentes]`, `proba[:, presentes]`): an
    /// absent class adds only negatives, which would lower the micro FPR with
    /// scores of a class that cannot be evaluated. With every class present (the
    /// built-in datasets) it is exactly scikit-learn's micro-average.
    pub micro: ClassCurves,
    /// Union of the per-class FPRs and the mean of the interpolated TPRs.
    pub macro_roc: Curve,
    pub macro_auc: f64,
    /// Mean of the per-class APs (drawn as a horizontal line).
    pub macro_ap: f64,
    /// `roc_auc_score(average="weighted", multi_class="ovr")` and the weighted AP
    /// (not shown in F2, D20).
    pub weighted_auc: f64,
    pub weighted_ap: f64,
    /// Model classes without samples in the test set: their curves are undefined
    /// and they are left out of every average (D20; Python misaligns the columns).
    pub absent_classes: Vec<usize>,
}

/// `plot_roc_curve` for 3+ classes. `proba` has one column per entry of `classes`
/// (the model's `classes_`, as class codes).
///
/// Classes of the model without samples in `y_true` are listed in
/// `absent_classes` and excluded from the per-class curves and from **every**
/// average, the micro-average included (D20; see [`MulticlassCurves::micro`]).
pub fn multiclass_curves(
    y_true: &[usize],
    proba: ArrayView2<'_, f64>,
    classes: &[usize],
) -> MulticlassCurves {
    let mut per_class = Vec::new();
    let mut absent = Vec::new();
    let mut present_cols = Vec::new();
    let mut supports = Vec::new();
    for (col, &class) in classes.iter().enumerate() {
        let y: Vec<bool> = y_true.iter().map(|&c| c == class).collect();
        let support = y.iter().filter(|&&b| b).count();
        if support == 0 || col >= proba.ncols() {
            absent.push(class);
            continue;
        }
        let score = proba.column(col).to_vec();
        per_class.push(class_curves(Some(class), &y, &score));
        present_cols.push(col);
        supports.push(support as f64);
    }

    // Micro: row-major ravel of the present columns.
    let mut y_flat = Vec::with_capacity(y_true.len() * present_cols.len());
    let mut s_flat = Vec::with_capacity(y_true.len() * present_cols.len());
    for (i, &t) in y_true.iter().enumerate() {
        for &col in &present_cols {
            y_flat.push(t == classes[col]);
            s_flat.push(proba[[i, col]]);
        }
    }
    let micro = class_curves(None, &y_flat, &s_flat);

    // Macro: np.unique(np.concatenate(fprs)), mean of np.interp.
    let mut all_fpr: Vec<f64> = per_class
        .iter()
        .flat_map(|c| c.roc.x.iter().copied())
        .collect();
    all_fpr.sort_by(f64::total_cmp);
    all_fpr.dedup();
    let k = per_class.len() as f64;
    let mut mean_tpr = vec![0.0; all_fpr.len()];
    for c in &per_class {
        for (m, &x) in mean_tpr.iter_mut().zip(&all_fpr) {
            *m += interp(x, &c.roc.x, &c.roc.y);
        }
    }
    for m in &mut mean_tpr {
        *m /= k;
    }
    let macro_auc = auc(&all_fpr, &mean_tpr);
    let aps: Vec<f64> = per_class.iter().map(|c| c.ap).collect();
    let aucs: Vec<f64> = per_class.iter().map(|c| c.auc).collect();
    let mean = |v: &[f64]| {
        if v.is_empty() {
            f64::NAN
        } else {
            pairwise_sum(v) / v.len() as f64
        }
    };
    let weighted = |v: &[f64]| {
        let total = pairwise_sum(&supports);
        let terms: Vec<f64> = v.iter().zip(&supports).map(|(a, w)| a * w).collect();
        pairwise_sum(&terms) / total
    };
    MulticlassCurves {
        macro_roc: Curve {
            x: all_fpr,
            y: mean_tpr,
            thresholds: Vec::new(),
        },
        macro_auc,
        macro_ap: mean(&aps),
        weighted_auc: weighted(&aucs),
        weighted_ap: weighted(&aps),
        per_class,
        micro,
        absent_classes: absent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roc_matches_sklearn_example() {
        // sklearn docs: y = [1, 1, 2, 2], scores = [0.1, 0.4, 0.35, 0.8], pos_label=2.
        let y = [false, false, true, true];
        let s = [0.1, 0.4, 0.35, 0.8];
        let r = roc_curve(&y, &s);
        assert_eq!(r.x, vec![0.0, 0.0, 0.5, 0.5, 1.0]);
        assert_eq!(r.y, vec![0.0, 0.5, 0.5, 1.0, 1.0]);
        assert_eq!(r.thresholds[1..], [0.8, 0.4, 0.35, 0.1]);
        assert!(r.thresholds[0].is_infinite());
        assert!((auc(&r.x, &r.y) - 0.75).abs() < 1e-15);
        // average_precision_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) = 0.8333…
        assert!((average_precision(&y, &s) - 0.833_333_333_333_333_3).abs() < 1e-15);
        let pr = precision_recall_curve(&y, &s);
        assert_eq!(pr.y, vec![0.5, 2.0 / 3.0, 0.5, 1.0, 1.0]);
        assert_eq!(pr.x, vec![1.0, 1.0, 0.5, 0.5, 0.0]);
        assert_eq!(pr.thresholds, vec![0.1, 0.35, 0.4, 0.8]);
    }

    #[test]
    fn interp_like_numpy() {
        let xp = [0.0, 0.0, 0.5, 1.0];
        let fp = [0.0, 0.2, 0.6, 1.0];
        assert_eq!(interp(0.0, &xp, &fp), 0.2);
        assert_eq!(interp(0.25, &xp, &fp), 0.4);
        assert_eq!(interp(1.0, &xp, &fp), 1.0);
        assert_eq!(interp(-1.0, &xp, &fp), 0.0);
    }
}

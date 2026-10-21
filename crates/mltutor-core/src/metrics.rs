//! Evaluation metrics shown in "📈 Evaluación" (fase2-regresion.md C1 and C2).
//!
//! * [`classification`]: `classification_report(y_test, y_pred, labels=range(k))`,
//!   `confusion_matrix` and the two sample tables, with scikit-learn's
//!   `zero_division="warn"` behaviour (0 and a warning).
//! * [`regression`]: MSE, RMSE, MAE and R² (`sklearn.metrics`), the errors
//!   histogram (`bins="auto"`) and the KDE drawn by `sns.histplot(kde=True)`.
//!
//! Inputs are class codes or values, never a model, so Trees, KNN and the neural
//! network reuse them.

use serde::{Deserialize, Serialize};

use crate::stats::{
    gaussian_kde_scott_cut, histogram_bin_edges_auto, histogram_counts, pairwise_sum, Histogram,
};

/// Precision, recall and F1 of one class (or an average). `support` is the number
/// of true samples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: u32,
}

/// One row of the "Muestras correctamente/incorrectamente clasificadas" tables.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampleRow {
    /// Position in the test set (the table's index).
    pub position: u32,
    /// Real class code.
    pub real: usize,
    /// Predicted class code.
    pub predicted: usize,
    pub correct: bool,
}

/// scikit-learn's `UndefinedMetricWarning`s with `zero_division="warn"`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MetricWarning {
    /// "Precision is ill-defined and being set to 0.0 in labels with no predicted
    /// samples." `classes` = class codes.
    NoPredictedSamples { classes: Vec<usize> },
    /// "Recall is ill-defined and being set to 0.0 in labels with no true samples."
    NoTrueSamples { classes: Vec<usize> },
}

impl MetricWarning {
    /// Spanish text for the student.
    pub fn message_es(&self, class_names: &[String]) -> String {
        let names = |classes: &[usize]| {
            classes
                .iter()
                .map(|&c| class_names.get(c).cloned().unwrap_or_else(|| c.to_string()))
                .collect::<Vec<_>>()
                .join(", ")
        };
        match self {
            MetricWarning::NoPredictedSamples { classes } => format!(
                "La precisión no está definida para las clases sin predicciones ({}); se toma 0.",
                names(classes)
            ),
            MetricWarning::NoTrueSamples { classes } => format!(
                "La exhaustividad no está definida para las clases sin muestras reales ({}); se toma 0.",
                names(classes)
            ),
        }
    }
}

/// Everything C1 shows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassificationEvaluation {
    pub accuracy: f64,
    /// One entry per class code `0..n_classes` (also classes absent from the test
    /// set, with support 0, like the app).
    pub per_class: Vec<ClassMetrics>,
    pub macro_avg: ClassMetrics,
    pub weighted_avg: ClassMetrics,
    /// `confusion[real][predicted]`.
    pub confusion: Vec<Vec<u32>>,
    /// First 10 correct / incorrect test samples.
    pub correct_head: Vec<SampleRow>,
    pub incorrect_head: Vec<SampleRow>,
    pub warnings: Vec<MetricWarning>,
}

/// Rows of each sample table (`head(10)`).
pub const SAMPLE_HEAD: usize = 10;

fn ratio(num: u32, den: u32) -> Option<f64> {
    (den > 0).then(|| f64::from(num) / f64::from(den))
}

/// `classification_report` + `confusion_matrix` + sample tables for labels
/// `0..n_classes`. Codes `>= n_classes` count as wrong predictions of no class.
pub fn classification(
    y_true: &[usize],
    y_pred: &[usize],
    n_classes: usize,
) -> ClassificationEvaluation {
    let n = y_true.len().min(y_pred.len());
    let mut confusion = vec![vec![0_u32; n_classes]; n_classes];
    let mut tp = vec![0_u32; n_classes];
    let mut pred_count = vec![0_u32; n_classes];
    let mut support = vec![0_u32; n_classes];
    let mut correct = 0_u32;
    let mut correct_head = Vec::new();
    let mut incorrect_head = Vec::new();
    for i in 0..n {
        let (t, p) = (y_true[i], y_pred[i]);
        if t < n_classes {
            support[t] += 1;
        }
        if p < n_classes {
            pred_count[p] += 1;
        }
        if t < n_classes && p < n_classes {
            confusion[t][p] += 1;
        }
        let ok = t == p;
        if ok {
            correct += 1;
            if t < n_classes {
                tp[t] += 1;
            }
        }
        let row = SampleRow {
            position: u32::try_from(i).unwrap_or(u32::MAX),
            real: t,
            predicted: p,
            correct: ok,
        };
        let head = if ok {
            &mut correct_head
        } else {
            &mut incorrect_head
        };
        if head.len() < SAMPLE_HEAD {
            head.push(row);
        }
    }

    let mut no_pred = Vec::new();
    let mut no_true = Vec::new();
    let per_class: Vec<ClassMetrics> = (0..n_classes)
        .map(|c| {
            let precision = ratio(tp[c], pred_count[c]).unwrap_or_else(|| {
                no_pred.push(c);
                0.0
            });
            let recall = ratio(tp[c], support[c]).unwrap_or_else(|| {
                no_true.push(c);
                0.0
            });
            // sklearn ≥ 1.3: F = 2·tp / (2·tp + fp + fn), zero_division if 0/0.
            let fp = pred_count[c] - tp[c];
            let fn_ = support[c] - tp[c];
            let f1 = ratio(2 * tp[c], 2 * tp[c] + fp + fn_).unwrap_or(0.0);
            ClassMetrics {
                precision,
                recall,
                f1,
                support: support[c],
            }
        })
        .collect();

    let total: u32 = support.iter().sum();
    let mean_of = |f: fn(&ClassMetrics) -> f64| {
        let v: Vec<f64> = per_class.iter().map(f).collect();
        if v.is_empty() {
            0.0
        } else {
            pairwise_sum(&v) / v.len() as f64
        }
    };
    let weighted_of = |f: fn(&ClassMetrics) -> f64| {
        if total == 0 {
            return 0.0;
        }
        let v: Vec<f64> = per_class
            .iter()
            .map(|m| f(m) * f64::from(m.support))
            .collect();
        pairwise_sum(&v) / f64::from(total)
    };
    let macro_avg = ClassMetrics {
        precision: mean_of(|m| m.precision),
        recall: mean_of(|m| m.recall),
        f1: mean_of(|m| m.f1),
        support: total,
    };
    let weighted_avg = ClassMetrics {
        precision: weighted_of(|m| m.precision),
        recall: weighted_of(|m| m.recall),
        f1: weighted_of(|m| m.f1),
        support: total,
    };
    let mut warnings = Vec::new();
    if !no_pred.is_empty() {
        warnings.push(MetricWarning::NoPredictedSamples { classes: no_pred });
    }
    if !no_true.is_empty() {
        warnings.push(MetricWarning::NoTrueSamples { classes: no_true });
    }
    ClassificationEvaluation {
        accuracy: if n == 0 {
            0.0
        } else {
            f64::from(correct) / n as f64
        },
        per_class,
        macro_avg,
        weighted_avg,
        confusion,
        correct_head,
        incorrect_head,
        warnings,
    }
}

/// The KDE line of `sns.histplot(errors, kde=True)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KdeCurve {
    pub x: Vec<f64>,
    /// Density scaled to counts (`density · n · bin_width`).
    pub y: Vec<f64>,
}

/// Everything C2 shows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegressionEvaluation {
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub r2: f64,
    /// Errors `y − ŷ`, the histogram (`bins="auto"`) and its KDE (`None` with fewer
    /// than 2 distinct errors, like seaborn).
    pub errors_hist: Histogram,
    pub errors_kde: Option<KdeCurve>,
}

/// `mean_squared_error`, `np.sqrt(mse)`, `mean_absolute_error` and `r2_score`.
pub fn regression_scores(y_true: &[f64], y_pred: &[f64]) -> (f64, f64, f64, f64) {
    let n = y_true.len().min(y_pred.len());
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let sq: Vec<f64> = (0..n).map(|i| (y_true[i] - y_pred[i]).powi(2)).collect();
    let ab: Vec<f64> = (0..n).map(|i| (y_true[i] - y_pred[i]).abs()).collect();
    let mse = pairwise_sum(&sq) / n as f64;
    let mae = pairwise_sum(&ab) / n as f64;
    (mse, mse.sqrt(), mae, r2_score(&y_true[..n], &y_pred[..n]))
}

/// `sklearn.metrics.r2_score` (1.7, `force_finite=True`): `1 − SS_res/SS_tot`; with
/// a constant `y_true`, 1.0 if the predictions are perfect and 0.0 otherwise.
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let n = y_true.len().min(y_pred.len());
    if n == 0 {
        return f64::NAN;
    }
    let mean = pairwise_sum(y_true) / n as f64;
    let res: Vec<f64> = (0..n).map(|i| (y_true[i] - y_pred[i]).powi(2)).collect();
    let tot: Vec<f64> = y_true.iter().map(|v| (v - mean).powi(2)).collect();
    let (num, den) = (pairwise_sum(&res), pairwise_sum(&tot));
    if den != 0.0 {
        1.0 - num / den
    } else if num == 0.0 {
        1.0
    } else {
        0.0
    }
}

/// C2: scores, errors histogram and KDE.
pub fn regression(y_true: &[f64], y_pred: &[f64]) -> RegressionEvaluation {
    let (mse, rmse, mae, r2) = regression_scores(y_true, y_pred);
    let errors: Vec<f64> = y_true.iter().zip(y_pred).map(|(t, p)| t - p).collect();
    let edges = histogram_bin_edges_auto(&errors);
    let counts = histogram_counts(&errors, &edges);
    let bin_width = if edges.len() >= 2 {
        edges[1] - edges[0]
    } else {
        0.0
    };
    let n = errors.iter().filter(|v| v.is_finite()).count() as f64;
    let errors_kde = gaussian_kde_scott_cut(&errors, 1.0, 0.0).map(|kde| KdeCurve {
        x: kde.grid(),
        y: kde.density.iter().map(|d| d * n * bin_width).collect(),
    });
    RegressionEvaluation {
        mse,
        rmse,
        mae,
        r2,
        errors_hist: Histogram { edges, counts },
        errors_kde,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_division_gives_zero_and_warnings() {
        // Class 2 never predicted, class 1 never true.
        let e = classification(&[0, 0, 2, 2], &[0, 1, 0, 0], 3);
        assert_eq!(e.per_class[2].precision, 0.0);
        assert_eq!(e.per_class[1].recall, 0.0);
        assert_eq!(
            e.warnings,
            vec![
                MetricWarning::NoPredictedSamples { classes: vec![2] },
                MetricWarning::NoTrueSamples { classes: vec![1] }
            ]
        );
        assert_eq!(
            e.confusion,
            vec![vec![1, 1, 0], vec![0, 0, 0], vec![2, 0, 0]]
        );
        assert_eq!(e.accuracy, 0.25);
        assert_eq!(e.incorrect_head.len(), 3);
    }

    #[test]
    fn r2_constant_target() {
        assert_eq!(r2_score(&[1.0, 1.0], &[1.0, 1.0]), 1.0);
        assert_eq!(r2_score(&[1.0, 1.0], &[1.0, 2.0]), 0.0);
    }
}

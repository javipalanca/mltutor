//! "📊 Distribución de Probabilidades" (C5) and the threshold analysis (C6) of
//! `viz/roc.py::plot_threshold_analysis`.

use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};

use crate::grid::np_arange;
use crate::stats::{histogram, Histogram};

/// Bins of every probability histogram (`ax.hist(..., bins=20)`).
pub const PROBA_BINS: usize = 20;

/// One histogram of C5 with the class it belongs to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassHistogram {
    /// Class code: the **real** class (binary) or the class whose probability is
    /// shown (multiclass).
    pub class: usize,
    /// 20 bins between the minimum and the maximum of the values (NumPy widens a
    /// constant range by ±0.5).
    pub hist: Histogram,
}

/// One row of the threshold table (C6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThresholdRow {
    /// Exact `np.arange(0.1, 1.0, 0.1)` value (0.30000000000000004, …).
    pub threshold: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// C6: metrics at each threshold where both classes are predicted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThresholdAnalysis {
    pub rows: Vec<ThresholdRow>,
    /// Threshold of the first maximum F1 (`idxmax`); `None` if no row qualifies.
    pub best_f1_threshold: Option<f64>,
    /// D21: the rows the table shows, 0.3, 0.5 and 0.7 (Python only managed 0.5
    /// because `np.arange` gives 0.30000000000000004).
    pub shown_thresholds: Vec<f64>,
}

/// The values C6 shows (0.3, 0.5, 0.7; D21).
pub const SHOWN_THRESHOLDS: [f64; 3] = [0.3, 0.5, 0.7];

/// C5 + C6 for a binary model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryProbabilities {
    /// Positive class (`classes_[1]`); the histograms show its probability.
    pub positive_class: usize,
    /// One histogram per real class (`classes_[0]`, `classes_[1]`), omitted if the
    /// class has no test samples.
    pub per_real_class: Vec<ClassHistogram>,
    /// Samples with `0.3 < p < 0.7`.
    pub uncertain: u32,
    /// Samples with `p < 0.2` or `p > 0.8`.
    pub confident: u32,
    pub n: u32,
    pub thresholds: ThresholdAnalysis,
}

impl BinaryProbabilities {
    /// `uncertain / n` ("Porcentaje de incertidumbre").
    pub fn uncertain_fraction(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            f64::from(self.uncertain) / f64::from(self.n)
        }
    }
}

/// C5/C6 data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProbabilityView {
    Binary(BinaryProbabilities),
    /// One histogram of `proba[:, i]` per model class.
    Multiclass {
        per_class: Vec<ClassHistogram>,
    },
}

/// Builds C5 (+ C6 if binary). `proba` columns follow `classes` (`classes_`).
pub fn distribution(
    y_true: &[usize],
    proba: ArrayView2<'_, f64>,
    classes: &[usize],
) -> ProbabilityView {
    if classes.len() == 2 && proba.ncols() >= 2 {
        let p1: Vec<f64> = proba.column(1).to_vec();
        let positive = classes[1];
        let per_real_class = classes
            .iter()
            .filter_map(|&c| {
                let values: Vec<f64> = y_true
                    .iter()
                    .zip(&p1)
                    .filter(|(&t, _)| t == c)
                    .map(|(_, &p)| p)
                    .collect();
                (!values.is_empty()).then(|| ClassHistogram {
                    class: c,
                    hist: histogram(&values, PROBA_BINS),
                })
            })
            .collect();
        let in_classes: Vec<f64> = y_true
            .iter()
            .zip(&p1)
            .filter(|(t, _)| classes.contains(t))
            .map(|(_, &p)| p)
            .collect();
        let uncertain = in_classes.iter().filter(|&&p| p > 0.3 && p < 0.7).count();
        let confident = p1.iter().filter(|&&p| p < 0.2 || p > 0.8).count();
        ProbabilityView::Binary(BinaryProbabilities {
            positive_class: positive,
            per_real_class,
            uncertain: u32::try_from(uncertain).unwrap_or(u32::MAX),
            confident: u32::try_from(confident).unwrap_or(u32::MAX),
            n: u32::try_from(y_true.len()).unwrap_or(u32::MAX),
            thresholds: threshold_analysis(y_true, &p1, positive),
        })
    } else {
        let per_class = classes
            .iter()
            .enumerate()
            .filter(|(col, _)| *col < proba.ncols())
            .map(|(col, &c)| ClassHistogram {
                class: c,
                hist: histogram(&proba.column(col).to_vec(), PROBA_BINS),
            })
            .collect();
        ProbabilityView::Multiclass { per_class }
    }
}

/// C6: for each `t` in `np.arange(0.1, 1.0, 0.1)`, predict the positive class when
/// `p ≥ t` and compute accuracy/precision/recall/F1 (`zero_division=0`); rows
/// where only one class is predicted are skipped, as in Python.
pub fn threshold_analysis(
    y_true: &[usize],
    p_positive: &[f64],
    positive: usize,
) -> ThresholdAnalysis {
    let n = y_true.len().min(p_positive.len());
    let mut rows = Vec::new();
    for t in np_arange(0.1, 1.0, 0.1) {
        let pred: Vec<bool> = p_positive[..n].iter().map(|&p| p >= t).collect();
        let n_pos = pred.iter().filter(|&&b| b).count();
        if n_pos == 0 || n_pos == n {
            continue;
        }
        let (mut tp, mut fp, mut fn_, mut correct) = (0_u32, 0_u32, 0_u32, 0_u32);
        for i in 0..n {
            let truth = y_true[i] == positive;
            match (truth, pred[i]) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => fn_ += 1,
                (false, false) => {}
            }
            if truth == pred[i] {
                correct += 1;
            }
        }
        let div = |a: u32, b: u32| {
            if b == 0 {
                0.0
            } else {
                f64::from(a) / f64::from(b)
            }
        };
        rows.push(ThresholdRow {
            threshold: t,
            accuracy: f64::from(correct) / n as f64,
            precision: div(tp, tp + fp),
            recall: div(tp, tp + fn_),
            f1: div(2 * tp, 2 * tp + fp + fn_),
        });
    }
    let mut best: Option<&ThresholdRow> = None;
    for r in &rows {
        if best.is_none_or(|b| r.f1 > b.f1) {
            best = Some(r);
        }
    }
    let shown_thresholds = rows
        .iter()
        .map(|r| r.threshold)
        .filter(|t| SHOWN_THRESHOLDS.iter().any(|s| (s - t).abs() < 1e-9))
        .collect();
    ThresholdAnalysis {
        best_f1_threshold: best.map(|r| r.threshold),
        rows,
        shown_thresholds,
    }
}

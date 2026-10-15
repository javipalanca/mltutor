//! C19: 📏 Análisis de Distancias entre Clases (`knn_app.py`, fase4-knn.md §2.4.2).
//!
//! For every pair of classes `i ≠ j` of the training rows, the mean distance between
//! their samples (`mean(pairwise_distances(X_i, X_j, metric))`); the diagonal stays at 0
//! as in Python. Then the "Análisis de Separabilidad": `min` of the cells > 0, `max`,
//! `ratio = max/min` (0 without a minimum) and the verdict ratio > 5 🟢, > 2 🟡, else 🔴.
//!
//! Differences with Python:
//!
//! * scikit-learn's `'euclidean'` uses the dot-product formula with BLAS and
//!   `'minkowski'` SciPy's `cdist`; here every distance is exact ([`super::search::rdist`]):
//!   the fixtures differ by ≤ 1e-13 relative.
//! * D56: the data is the model's training split (Python always re-splits with
//!   `test_size = 0.3`; with the default slider both are the same rows).
//! * D67: also the mean distance **within** each class ([`ClassDistances::within`],
//!   over the `n(n−1)/2` distinct pairs) for the diagonal, and with two classes the
//!   verdict is [`SeparationVerdict::Binary`]: the matrix then has two equal cells, the
//!   ratio is always 1 and Python always said "🔴 muy próximas". The Python verdict is
//!   still available in [`ClassDistances::ratio_verdict`].
//!
//! Each unordered pair of classes is computed once (`M[i, j] = M[j, i]`): Python's two
//! cells differ in the last bits only. Sums are pairwise per row and then over the
//! rows (error O(log n · ε)).

use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};

use super::search::rdist;
use super::KnnMetric;
use crate::model::{check_n_classes, ModelError, TrainControl};
use crate::stats::pairwise_sum;

/// The message of the "Análisis de Separabilidad".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeparationVerdict {
    /// ratio > 5: "🟢 Las clases están bien separadas (ratio > 5)".
    Separated,
    /// 2 < ratio ≤ 5: "🟡 Separación moderada entre clases (2 < ratio ≤ 5)".
    Moderate,
    /// ratio ≤ 2: "🔴 Las clases están muy próximas (ratio ≤ 2)".
    Close,
    /// D67: two classes (`[distancias_nuevo] binaria`).
    Binary,
    /// Fewer than 2 classes in the data: "Se necesitan al menos 2 clases…".
    TooFewClasses,
}

/// Python's rule on the ratio.
pub fn verdict_of_ratio(ratio: f64) -> SeparationVerdict {
    if ratio > 5.0 {
        SeparationVerdict::Separated
    } else if ratio > 2.0 {
        SeparationVerdict::Moderate
    } else {
        SeparationVerdict::Close
    }
}

/// Result of [`class_distances`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassDistances {
    /// Class codes present in the data, sorted (`np.unique(y_train)`).
    pub classes: Vec<usize>,
    /// Samples of each class.
    pub counts: Vec<usize>,
    /// `k × k` mean distances between classes; the diagonal is 0 (Python).
    pub matrix: Vec<Vec<f64>>,
    /// D67: mean distance between distinct samples of each class (`None` with fewer
    /// than 2 samples). Shown on the diagonal by the new view.
    pub within: Vec<Option<f64>>,
    /// Smallest cell > 0 (`None` if there is none).
    pub min: Option<f64>,
    /// Largest cell.
    pub max: Option<f64>,
    /// `max/min`, or 0 without a minimum.
    pub ratio: f64,
    /// The verdict Python shows (on the ratio).
    pub ratio_verdict: SeparationVerdict,
    /// The verdict the app shows (D67: [`SeparationVerdict::Binary`] with 2 classes).
    pub verdict: SeparationVerdict,
    pub metric: KnnMetric,
    /// Rows used.
    pub n_samples: usize,
}

/// Sum of the distances between every row of `a` and every row of `b` (row-major,
/// `p` columns); with `same`, only the pairs `i < j`.
fn sum_distances(
    a: &[&[f64]],
    b: &[&[f64]],
    same: bool,
    metric: KnnMetric,
    control: &TrainControl<'_>,
) -> Result<f64, ModelError> {
    let m = metric.effective();
    let mut row_sums = Vec::with_capacity(a.len());
    let mut row = Vec::with_capacity(b.len());
    for (i, ra) in a.iter().enumerate() {
        if i % 64 == 0 {
            control.check()?;
        }
        row.clear();
        let others = if same { &b[i + 1..] } else { b };
        row.extend(others.iter().map(|rb| m.to_distance(rdist(ra, rb, m))));
        row_sums.push(pairwise_sum(&row));
    }
    Ok(pairwise_sum(&row_sums))
}

/// C19 on the training rows `x` with their class `codes`.
pub fn class_distances(
    x: ArrayView2<'_, f64>,
    codes: &[usize],
    metric: KnnMetric,
    control: &TrainControl<'_>,
) -> Result<ClassDistances, ModelError> {
    let (n, p) = x.dim();
    if codes.len() != n {
        return Err(ModelError::LengthMismatch {
            rows: n,
            targets: codes.len(),
        });
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(ModelError::NonFinite);
    }
    let mut classes = codes.to_vec();
    classes.sort_unstable();
    classes.dedup();
    check_n_classes(classes.len())?;
    let data = x.as_standard_layout();
    let flat = data.as_slice().unwrap_or(&[]);
    let rows_of = |c: usize| -> Vec<&[f64]> {
        codes
            .iter()
            .enumerate()
            .filter(|&(_, &code)| code == c)
            .map(|(i, _)| &flat[i * p..(i + 1) * p])
            .collect()
    };
    let groups: Vec<Vec<&[f64]>> = classes.iter().map(|&c| rows_of(c)).collect();
    let k = classes.len();
    let mut matrix = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in i + 1..k {
            let total = sum_distances(&groups[i], &groups[j], false, metric, control)?;
            let mean = total / (groups[i].len() * groups[j].len()) as f64;
            matrix[i][j] = mean;
            matrix[j][i] = mean;
        }
    }
    let mut within = Vec::with_capacity(k);
    for g in &groups {
        let m = g.len();
        within.push(if m > 1 {
            let total = sum_distances(g, g, true, metric, control)?;
            Some(total / ((m * (m - 1)) as f64 / 2.0))
        } else {
            None
        });
    }
    let cells = matrix.iter().flatten().copied();
    let min = cells
        .clone()
        .filter(|&v| v > 0.0)
        .fold(None, |acc: Option<f64>, v| {
            Some(acc.map_or(v, |a| a.min(v)))
        });
    let max = cells.fold(None, |acc: Option<f64>, v| {
        Some(acc.map_or(v, |a| a.max(v)))
    });
    let ratio = match (min, max) {
        (Some(lo), Some(hi)) if lo > 0.0 => hi / lo,
        _ => 0.0,
    };
    let ratio_verdict = verdict_of_ratio(ratio);
    let verdict = match k {
        0 | 1 => SeparationVerdict::TooFewClasses,
        2 => SeparationVerdict::Binary,
        _ => ratio_verdict,
    };
    Ok(ClassDistances {
        counts: groups.iter().map(Vec::len).collect(),
        classes,
        matrix,
        within,
        min,
        max: if k > 1 { max } else { None },
        ratio,
        ratio_verdict,
        verdict,
        metric,
        n_samples: n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn two_classes_are_binary() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [12.0, 0.0]];
        let d = class_distances(
            x.view(),
            &[0, 0, 1, 1],
            KnnMetric::Manhattan,
            &TrainControl::none(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.matrix[0][1], (10.0 + 12.0 + 9.0 + 11.0) / 4.0);
        assert_eq!(d.ratio, 1.0);
        assert_eq!(d.ratio_verdict, SeparationVerdict::Close);
        assert_eq!(d.verdict, SeparationVerdict::Binary);
        assert_eq!(d.within, vec![Some(1.0), Some(2.0)]);
    }

    #[test]
    fn one_class() {
        let x = array![[0.0], [1.0]];
        let d = class_distances(
            x.view(),
            &[1, 1],
            KnnMetric::Euclidean,
            &TrainControl::none(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.verdict, SeparationVerdict::TooFewClasses);
        assert_eq!((d.min, d.max, d.ratio), (None, None, 0.0));
    }
}

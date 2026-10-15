//! C19: 📏 Análisis de Distancias entre Clases (`knn_app.py`, fase4-knn.md §2.4.2).
//!
//! For every pair of classes `i ≠ j` of the training rows, the mean distance between
//! their samples (`mean(pairwise_distances(X_i, X_j, metric))`); the diagonal stays at 0
//! as in Python. Then Python's "Análisis de Separabilidad": `min` of the cells > 0,
//! `max`, `ratio = max/min` (0 without a minimum) and the verdict ratio > 5 🟢, > 2 🟡,
//! else 🔴 ([`ClassDistances::ratio_verdict`], kept for parity and only informative).
//!
//! **The verdict the app shows** ([`ClassDistances::verdict`], D67 generalised in the
//! F4 review) does not use that ratio: `max/min` compares two *between*-class
//! distances and says nothing about overlap (Iris gives 2.53 although versicolor and
//! virginica overlap; with two classes it is always 1). It compares each between-class
//! distance with the spread of the classes involved:
//!
//! ```text
//! W_i        = mean distance between distinct samples of class i   (within)
//! separation = min_{i≠j} M[i, j] / max(W_i, W_j)
//! verdict    = separation ≥ 2 → Separated, ≥ 1 → Moderate, < 1 → Close
//! ```
//!
//! A separation below 1 means that two classes are, on average, closer to each other
//! than the samples of one of them are among themselves. Classes without an internal
//! distance follow the student's script (`np.fmax` / `np.nanmin`, see [`separation`]):
//! a class with one sample uses the other class's `W`; with no usable pair the
//! separation is `None` and the verdict 🔴 ([`TooFewClasses`] with < 2 classes).
//! Reference values (training rows of `test_size = 0.3`, Euclidean, unscaled):
//! Iris 1.58 🟡 (setosa is far, versicolor and virginica touch), Digits 1.11 🟡
//! (Python's ratio said 🔴), Wine 1.04 🟡 (proline dominates), Cancer 1.26 🟡 (two
//! classes: Python's ratio is always 1 🔴), Titanic 0.79 🔴, Penguins 0.90 🔴 (body
//! mass dominates and Adelie/Chinstrap overlap in it).
//!
//! Differences with Python:
//!
//! * scikit-learn's `'euclidean'` uses the dot-product formula with BLAS and
//!   `'minkowski'` SciPy's `cdist`; here every distance is exact ([`super::search::rdist`]):
//!   the fixtures differ by ≤ 1e-13 relative.
//! * D56: the data is the model's training split (Python always re-splits with
//!   `test_size = 0.3`; with the default slider both are the same rows).
//! * Cost: all the pairs are O(n²·p). Classes with more than [`MAX_ROWS_PER_CLASS`]
//!   rows are replaced by a random sample of that size (`RandomState(42)`, the rows
//!   kept in their order) and [`ClassDistances::sampled`] tells the view to say so;
//!   the rows of each sum run on up to 8 threads and the cancellation flag is checked
//!   every 64 rows.
//!
//! Each unordered pair of classes is computed once (`M[i, j] = M[j, i]`): Python's two
//! cells differ in the last bits only. Each row of a block is summed pairwise on its
//! own, and the row sums pairwise in row order (error O(log n · ε)): the result does
//! not depend on the number of threads.
//!
//! [`TooFewClasses`]: SeparationVerdict::TooFewClasses

use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};

use super::search::rdist;
use super::KnnMetric;
use crate::model::{check_n_classes, ModelError, TrainControl};
use crate::sampling::sample_rows;
use crate::stats::pairwise_sum;

/// Classes with more rows than this are sampled (O(n²) pairs: 5 000 × 5 000 = 25 M
/// distances per block).
pub const MAX_ROWS_PER_CLASS: usize = 5000;
/// Seed of that sample (`random_state=42`).
const SAMPLE_SEED: u32 = 42;
/// Below this many pairs a block runs on the calling thread.
const PARALLEL_MIN_PAIRS: usize = 1 << 18;
/// Upper bound of worker threads.
const MAX_THREADS: usize = 8;
/// The cancellation flag is checked every this many rows of a block.
const CANCEL_EVERY: usize = 64;

/// The message of the "Análisis de Separabilidad".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeparationVerdict {
    /// Python: ratio > 5. App: separation ≥ 2.
    Separated,
    /// Python: 2 < ratio ≤ 5. App: 1 ≤ separation < 2.
    Moderate,
    /// Python: ratio ≤ 2. App: separation < 1.
    Close,
    /// Former D67 verdict for two classes. No longer produced (two classes now get a
    /// verdict from [`ClassDistances::separation`]); kept so that the IPC enum and
    /// serialized values stay valid.
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

/// The app's rule on the separation (see the module docs).
pub fn verdict_of_separation(separation: f64) -> SeparationVerdict {
    if separation >= 2.0 {
        SeparationVerdict::Separated
    } else if separation >= 1.0 {
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
    /// Samples of each class in the data (before any sampling).
    pub counts: Vec<usize>,
    /// `k × k` mean distances between classes; the diagonal is 0 (Python).
    pub matrix: Vec<Vec<f64>>,
    /// D67: mean distance between distinct samples of each class (`None` with fewer
    /// than 2 samples). Shown on the diagonal by the new view.
    pub within: Vec<Option<f64>>,
    /// `min_{i≠j} M[i, j] / max(W_i, W_j)` (a missing `W` is ignored, `np.fmax`);
    /// `None` with fewer than 2 classes or no usable pair; `+∞` if two classes of
    /// identical samples are apart (serialized as `null`).
    pub separation: Option<f64>,
    /// Smallest cell > 0 (`None` if there is none).
    pub min: Option<f64>,
    /// Largest cell.
    pub max: Option<f64>,
    /// `max/min`, or 0 without a minimum (Python; informative).
    pub ratio: f64,
    /// The verdict Python shows (on the ratio; informative).
    pub ratio_verdict: SeparationVerdict,
    /// The verdict the app shows: on `separation` (`Close` when it is `None` with 2+
    /// classes); [`SeparationVerdict::TooFewClasses`] with < 2.
    pub verdict: SeparationVerdict,
    pub metric: KnnMetric,
    /// Rows received.
    pub n_samples: usize,
    /// Some class had more than [`MAX_ROWS_PER_CLASS`] rows and was sampled.
    pub sampled: bool,
}

/// Sum of the distances of each row `a[i]`, `i` in `rows`, against `b` (with
/// `same`, row `i` only against `b[i + 1..]`): one pairwise sum per row.
fn row_sums(
    a: &[&[f64]],
    b: &[&[f64]],
    rows: impl Iterator<Item = usize>,
    same: bool,
    metric: KnnMetric,
    control: &TrainControl<'_>,
) -> Result<Vec<f64>, ModelError> {
    let m = metric.effective();
    let mut out = Vec::with_capacity(rows.size_hint().0);
    let mut row = Vec::with_capacity(b.len());
    for (r, i) in rows.enumerate() {
        if r % CANCEL_EVERY == 0 {
            control.check()?;
        }
        let ra = a[i];
        row.clear();
        let others = if same { &b[i + 1..] } else { b };
        row.extend(others.iter().map(|rb| m.to_distance(rdist(ra, rb, m))));
        out.push(pairwise_sum(&row));
    }
    Ok(out)
}

/// Sum of the distances between every row of `a` and every row of `b`; with `same`
/// (`a` = `b`), only the pairs `i < j`. The rows are split among threads, but every
/// row is summed alone and the row sums are added in row order, so the result is the
/// same with any number of threads.
fn sum_distances(
    a: &[&[f64]],
    b: &[&[f64]],
    same: bool,
    metric: KnnMetric,
    control: &TrainControl<'_>,
) -> Result<f64, ModelError> {
    let n = a.len();
    let pairs = if same {
        n * n.saturating_sub(1) / 2
    } else {
        n * b.len()
    };
    let threads = if pairs >= PARALLEL_MIN_PAIRS {
        std::thread::available_parallelism()
            .map_or(1, |t| t.get())
            .min(MAX_THREADS)
            .min(n)
            .max(1)
    } else {
        1
    };
    let sums = if threads <= 1 {
        row_sums(a, b, 0..n, same, metric, control)?
    } else {
        // The rows are interleaved among the T threads (with `same` the first rows
        // are the longest, so contiguous blocks would be unbalanced): thread t gets
        // the rows t, t + T, t + 2T… and the sums are put back in row order.
        let parts: Vec<Result<Vec<f64>, ModelError>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    s.spawn(move || row_sums(a, b, (t..n).step_by(threads), same, metric, control))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| {
                        Err(ModelError::Numerical {
                            message: "fallo en el cálculo de distancias".to_owned(),
                        })
                    })
                })
                .collect()
        });
        let parts = parts.into_iter().collect::<Result<Vec<_>, _>>()?;
        let mut sums = vec![0.0; n];
        for (t, part) in parts.iter().enumerate() {
            for (r, &v) in part.iter().enumerate() {
                sums[t + r * threads] = v;
            }
        }
        sums
    };
    let total = pairwise_sum(&sums);
    if total.is_finite() {
        Ok(total)
    } else {
        Err(ModelError::NonFinite)
    }
}

/// C19 on the training rows `x` with their class `codes`. `control` is checked
/// every 64 rows of every block of distances.
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
    control.check()?;
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
    let full: Vec<Vec<&[f64]>> = classes.iter().map(|&c| rows_of(c)).collect();
    let counts: Vec<usize> = full.iter().map(Vec::len).collect();
    let sampled = counts.iter().any(|&c| c > MAX_ROWS_PER_CLASS);
    let groups: Vec<Vec<&[f64]>> = full
        .into_iter()
        .map(|g| {
            if g.len() <= MAX_ROWS_PER_CLASS {
                return g;
            }
            let mut keep = sample_rows(g.len(), MAX_ROWS_PER_CLASS, SAMPLE_SEED);
            keep.sort_unstable();
            keep.into_iter().map(|i| g[i]).collect()
        })
        .collect();
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
    let separation = separation(&matrix, &within);
    let ratio_verdict = verdict_of_ratio(ratio);
    let verdict = match (k, separation) {
        (0 | 1, _) => SeparationVerdict::TooFewClasses,
        (_, Some(s)) => verdict_of_separation(s),
        // The script's `np.nanmin` of nothing is NaN, and NaN ≥ 1 is false: 🔴.
        (_, None) => SeparationVerdict::Close,
    };
    Ok(ClassDistances {
        counts,
        classes,
        matrix,
        within,
        separation,
        min,
        max: if k > 1 { max } else { None },
        ratio,
        ratio_verdict,
        verdict,
        metric,
        n_samples: n,
        sampled,
    })
}

/// `min_{i≠j} M[i, j] / max(W_i, W_j)` (J2) with the rule of the student's script
/// (`knn_entrenamiento.py.j2`: `np.nanmin` of `M / np.fmax(W_i, W_j)`):
///
/// * a missing `W` (a class with one sample) is ignored by the `max` (`np.fmax`);
///   a pair where both are missing is skipped;
/// * `W = 0` (all the samples of a class identical) is a real 0: with the other also
///   0 the pair gives `M / 0 = +∞` (perfectly separated) or, if `M = 0` too, `0/0`,
///   which is skipped (`np.nanmin`);
/// * `None` with fewer than 2 classes or no usable pair.
///
/// The value can be `+∞` (serde_json writes it as `null`; the verdict is
/// `Separated`).
fn separation(matrix: &[Vec<f64>], within: &[Option<f64>]) -> Option<f64> {
    let k = within.len();
    if k < 2 {
        return None;
    }
    let mut best: Option<f64> = None;
    for i in 0..k {
        for j in i + 1..k {
            let denom = match (within[i], within[j]) {
                (Some(a), Some(b)) => a.max(b),
                (Some(w), None) | (None, Some(w)) => w,
                (None, None) => continue,
            };
            let s = matrix[i][j] / denom;
            if s.is_nan() {
                continue;
            }
            best = Some(best.map_or(s, |b: f64| b.min(s)));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{load_builtin, DatasetId};
    use crate::sampling::RandomState;
    use crate::split::train_test_split;
    use ndarray::{array, Array2, Axis};

    fn distances_of(id: DatasetId, metric: KnnMetric) -> ClassDistances {
        let ds = load_builtin(id).unwrap_or_else(|e| panic!("{e}"));
        let s = train_test_split(ds.n_rows(), 0.3, 42).unwrap_or_else(|e| panic!("{e}"));
        let x = ds.features.select(Axis(0), &s.train);
        let codes = ds
            .target
            .select(&s.train)
            .class_codes()
            .unwrap_or_else(|| panic!("classes"));
        class_distances(x.view(), &codes, metric, &TrainControl::none())
            .unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn two_classes_get_a_real_verdict() {
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
        assert_eq!(d.within, vec![Some(1.0), Some(2.0)]);
        // 10.5 / max(1, 2)
        assert_eq!(d.separation, Some(5.25));
        assert_eq!(d.verdict, SeparationVerdict::Separated);
        assert!(!d.sampled);
    }

    #[test]
    fn overlapping_classes_are_close() {
        // Two interleaved classes on a line: M = 2, W = 2 → 1 (Moderate); a third
        // class on top of class 0 is Close.
        let x = array![[0.0], [1.0], [2.0], [3.0], [0.5], [2.5]];
        let d = class_distances(
            x.view(),
            &[0, 1, 0, 1, 2, 2],
            KnnMetric::Manhattan,
            &TrainControl::none(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let s = d.separation.unwrap_or_else(|| panic!("separation"));
        assert!(s < 1.0, "{s}");
        assert_eq!(d.verdict, SeparationVerdict::Close);
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
        assert_eq!(d.separation, None);
    }

    #[test]
    fn classes_without_internal_distance_follow_the_script() {
        // Class 1 has one sample: its pairs use the other class's W (np.fmax).
        let x = array![[0.0], [1.0], [10.0], [20.0], [21.0]];
        let d = class_distances(
            x.view(),
            &[0, 0, 1, 2, 2],
            KnnMetric::Manhattan,
            &TrainControl::none(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.within, vec![Some(1.0), None, Some(1.0)]);
        // M[0,1] = 9.5, M[1,2] = 10.5, M[0,2] = 20 → min 9.5 / 1.
        assert_eq!(d.separation, Some(9.5));
        assert_eq!(d.verdict, SeparationVerdict::Separated);
        // Identical samples (W = 0) in both classes, apart: +∞.
        let x = array![[0.0], [0.0], [5.0], [5.0]];
        let d = class_distances(
            x.view(),
            &[0, 0, 1, 1],
            KnnMetric::Manhattan,
            &TrainControl::none(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.separation, Some(f64::INFINITY));
        assert_eq!(d.verdict, SeparationVerdict::Separated);
        // One sample per class: no usable pair → None and 🔴, like NaN in the script.
        let x = array![[0.0], [5.0]];
        let d = class_distances(
            x.view(),
            &[0, 1],
            KnnMetric::Manhattan,
            &TrainControl::none(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.separation, None);
        assert_eq!(d.verdict, SeparationVerdict::Close);
    }

    #[test]
    fn gaussian_triangle_is_separated() {
        // Three clouds (σ = 1, 200 points each) at the vertices of a triangle of
        // side 100: M ≈ 100, W ≈ 1.77 (σ·√π) → separation ≈ 56.
        let mut rs = RandomState::new(42);
        let h = 100.0 * 3f64.sqrt() / 2.0;
        let centres = [[0.0, 0.0], [100.0, 0.0], [50.0, h]];
        let n = 200;
        let x = Array2::from_shape_fn((3 * n, 2), |(i, j)| {
            centres[i / n][j] + rs.standard_normal()
        });
        let codes: Vec<usize> = (0..3 * n).map(|i| i / n).collect();
        let d = class_distances(
            x.view(),
            &codes,
            KnnMetric::Euclidean,
            &TrainControl::none(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let s = d.separation.unwrap_or_else(|| panic!("separation"));
        assert!(s > 40.0 && s < 70.0, "{s}");
        assert_eq!(d.verdict, SeparationVerdict::Separated);
        // Python's ratio: the three cells are ≈ 100 → ≈ 1 → "muy próximas".
        assert_eq!(d.ratio_verdict, SeparationVerdict::Close);
    }

    #[test]
    fn digits_is_no_longer_close() {
        let d = distances_of(DatasetId::Digits, KnnMetric::Minkowski);
        let s = d.separation.unwrap_or_else(|| panic!("separation"));
        // 1.106 (Python's ratio said 🔴 "muy próximas").
        assert!((s - 1.106).abs() < 1e-3, "{s}");
        assert_eq!(d.verdict, SeparationVerdict::Moderate);
    }

    #[test]
    fn iris_verdict() {
        let d = distances_of(DatasetId::Iris, KnnMetric::Minkowski);
        let s = d.separation.unwrap_or_else(|| panic!("separation"));
        // 1.582: setosa is far, versicolor/virginica touch → 🟡 Moderate, the same
        // message as Python's ratio 2.53, now for the right reason.
        assert!((s - 1.582).abs() < 1e-3, "{s}");
        assert_eq!(d.verdict, SeparationVerdict::Moderate);
        assert_eq!(d.ratio_verdict, SeparationVerdict::Moderate);
    }

    #[test]
    fn cancels_and_threads_do_not_change_the_sums() {
        // 2 × 600 rows: the within blocks (≈ 180 k pairs) are serial, the
        // between block (360 k pairs) is threaded.
        let n = 1200;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| ((i * 7 + j * 13) % 101) as f64 * 0.1);
        let codes: Vec<usize> = (0..n).map(|i| i % 2).collect();
        let d = class_distances(
            x.view(),
            &codes,
            KnnMetric::Euclidean,
            &TrainControl::none(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let g0: Vec<&[f64]> = (0..n)
            .step_by(2)
            .map(|i| x.slice(ndarray::s![i, ..]).to_slice().unwrap_or(&[]))
            .collect();
        let g1: Vec<&[f64]> = (1..n)
            .step_by(2)
            .map(|i| x.slice(ndarray::s![i, ..]).to_slice().unwrap_or(&[]))
            .collect();
        let serial = row_sums(
            &g0,
            &g1,
            0..g0.len(),
            false,
            KnnMetric::Euclidean,
            &TrainControl::none(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let mean = pairwise_sum(&serial) / (g0.len() * g1.len()) as f64;
        assert_eq!(d.matrix[0][1].to_bits(), mean.to_bits());
        let flag = std::sync::atomic::AtomicBool::new(true);
        assert_eq!(
            class_distances(
                x.view(),
                &codes,
                KnnMetric::Euclidean,
                &TrainControl::none().with_cancel(&flag),
            )
            .unwrap_err(),
            ModelError::Cancelled
        );
    }

    #[test]
    fn large_classes_are_sampled() {
        let n = 2 * MAX_ROWS_PER_CLASS + 20;
        let x = Array2::from_shape_fn((n, 1), |(i, _)| (i % 2) as f64 * 10.0 + (i % 7) as f64);
        let codes: Vec<usize> = (0..n).map(|i| i % 2).collect();
        let d = class_distances(
            x.view(),
            &codes,
            KnnMetric::Manhattan,
            &TrainControl::none(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert!(d.sampled);
        assert_eq!(d.counts, vec![MAX_ROWS_PER_CLASS + 10; 2]);
        assert_eq!(d.n_samples, n);
    }

    #[test]
    fn overflowing_distances_are_rejected() {
        // (2e160)² overflows.
        let x = array![[1e160], [-1e160], [0.0], [1.0]];
        assert_eq!(
            class_distances(
                x.view(),
                &[0, 1, 0, 1],
                KnnMetric::Euclidean,
                &TrainControl::none(),
            )
            .unwrap_err(),
            ModelError::NonFinite
        );
    }
}

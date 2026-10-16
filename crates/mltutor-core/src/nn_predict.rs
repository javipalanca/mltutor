//! 🔮 Predicciones of the 🧠 network: "📊 Predicción por lotes" and "🎲 Exploración
//! interactiva" (fase5-redes.md C29, D81).
//!
//! Both come from `viz/nn.py::show_neural_network_predictions`, which the Python app
//! never calls (§12.13): they are NEW subviews with Python's own semantics, checked
//! against that function rendered headlessly with the final weights of
//! `nn/exact/<case>` (`tests/parity/nn/predictions.json`).
//!
//! The network itself lives in `mltutor-nn`, which depends on this crate, so the
//! functions take any [`NetworkOutput`]: a [`Predictor`] that also exposes its raw
//! outputs. `mltutor_nn::NeuralNetwork` implements it.
//!
//! What a row's output means (Python's `config['output_size']`):
//!
//! | Outputs | Kind | Class | Probability / confidence |
//! |---|---|---|---|
//! | 1, classification | binary | `classes[p > 0.5]` (D87) | "Probabilidad" = `p`; "Confianza" = `max(p, 1 − p)` (float32) |
//! | k > 1, classification | multiclass | `classes[argmax]` (first maximum) | "Confianza" = the largest output |
//! | regression | regression | — | the first output |
//!
//! With a **tanh** output (`output_is_probability = false`) the same numbers are
//! computed, as Python does, but they are not probabilities: the UI must say so.

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::dataset::TaskType;
use crate::model::{ModelError, Predictor, TrainControl};
use crate::stats::histogram::linspace;
use crate::stats::sum::pairwise_sum_by;

/// Points of the exploration curve in Python (`np.linspace(min, max, 50)`).
pub const EXPLORE_DEFAULT_POINTS: usize = 50;
/// Fewest points of a curve.
pub const EXPLORE_MIN_POINTS: usize = 2;
/// Most points of a curve (the IPC cannot ask for more).
pub const EXPLORE_MAX_POINTS: usize = 1000;
/// Most rows of a batch prediction. A 5 × 256 network predicts them in about a
/// second; the CSV reader has its own limits before this one.
pub const BATCH_MAX_ROWS: usize = 200_000;
/// Rows predicted between two cancellation checks in a batch.
pub const BATCH_CHUNK_ROWS: usize = 4096;
/// Binary threshold of the class (Keras `BinaryAccuracy`, D87) and of the
/// exploration's "Umbral de decisión" line.
pub const BINARY_THRESHOLD: f64 = 0.5;
/// Probability range above which a feature is "muy influyente" (binary; NEW: also
/// the base-class probability in multiclass).
pub const PROBABILITY_RANGE_HIGH: f64 = 0.3;
/// Probability range above which a feature is "moderadamente influyente".
pub const PROBABILITY_RANGE_MODERATE: f64 = 0.1;
/// Relative impact (%) above which a feature is "muy influyente" (regression).
pub const RELATIVE_IMPACT_HIGH: f64 = 20.0;
/// Relative impact (%) above which a feature is "moderadamente influyente".
pub const RELATIVE_IMPACT_MODERATE: f64 = 5.0;

/// A trained network as the 🔮 views see it.
pub trait NetworkOutput: Predictor {
    /// Raw outputs for rows in original units (`n × n_outputs`; float32 values
    /// widened to f64). Errors like [`Predictor::predict`].
    fn raw_output(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>, ModelError>;

    /// `false` with a tanh output: the "probabilities" may be negative or above 1.
    fn output_is_probability(&self) -> bool;
}

/// How the outputs of a network are read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputKind {
    /// One output neuron in classification.
    Binary,
    /// One output neuron per class.
    Multiclass,
    Regression,
}

fn output_kind(net: &dyn NetworkOutput, n_outputs: usize) -> OutputKind {
    match net.task() {
        TaskType::Regression => OutputKind::Regression,
        TaskType::Classification if n_outputs == 1 => OutputKind::Binary,
        TaskType::Classification => OutputKind::Multiclass,
    }
}

/// The prediction of one row (the "🎯 Predicción Base" of the exploration).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RowPrediction {
    /// "{clase} (Probabilidad: {p:.3f})".
    Binary {
        /// Dataset class code.
        class: usize,
        /// The output `p` (probability of `classes[1]`).
        probability: f64,
        /// `max(p, 1 − p)`.
        confidence: f64,
    },
    /// "{clase} (Confianza: {c:.3f})".
    Multiclass {
        class: usize,
        /// The largest output.
        confidence: f64,
        /// Every output, in the order of the network's classes.
        probabilities: Vec<f64>,
    },
    /// "{valor:.6f}".
    Regression { value: f64 },
}

/// Class, probability and confidence of one output row (`classes` = codes).
fn read_row(kind: OutputKind, row: &[f64], classes: &[usize]) -> RowPrediction {
    let code = |i: usize| classes.get(i).copied().unwrap_or(i);
    match kind {
        OutputKind::Binary => {
            let p = row.first().copied().unwrap_or(f64::NAN);
            RowPrediction::Binary {
                class: code(usize::from(p > BINARY_THRESHOLD)),
                probability: p,
                confidence: binary_confidence(p),
            }
        }
        OutputKind::Multiclass => {
            let best = argmax(row);
            RowPrediction::Multiclass {
                class: code(best),
                confidence: row.get(best).copied().unwrap_or(f64::NAN),
                probabilities: row.to_vec(),
            }
        }
        OutputKind::Regression => RowPrediction::Regression {
            value: row.first().copied().unwrap_or(f64::NAN),
        },
    }
}

/// `np.maximum(p, 1 - p)` on Keras' float32 output: `1 − p` is rounded to float32,
/// as in Python.
fn binary_confidence(p: f64) -> f64 {
    let p32 = p as f32;
    f64::from(p32.max(1.0 - p32))
}

/// First index of the largest value (`np.argmax`; NaN never wins).
fn argmax(row: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// 🎲 Exploración interactiva
// ---------------------------------------------------------------------------

/// The response curve of the exploration, by kind of output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExploreCurve {
    /// "Efecto de {f} en la Probabilidad": `p` at every point, with the threshold
    /// line at [`BINARY_THRESHOLD`].
    Binary { probability: Vec<f64> },
    /// "Efecto de {f} en la Predicción": predicted class code (left axis) and
    /// confidence (right axis), as Python; NEW: the probability of the base
    /// sample's class at every point, which the analysis uses.
    Multiclass {
        classes: Vec<usize>,
        confidence: Vec<f64>,
        /// Code of the base sample's predicted class.
        base_class: usize,
        base_class_probability: Vec<f64>,
    },
    /// "Efecto de {f} en la Predicción": the predicted value.
    Regression { values: Vec<f64> },
}

/// "📈 Análisis de Resultados".
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExploreStats {
    /// `max − min` of the analysed curve (probability, base-class probability or
    /// value), in float32 like Python.
    pub range: f64,
    /// "Volatilidad": population standard deviation (`np.std`) of the curve.
    pub std: f64,
    /// Mean of the curve.
    pub mean: f64,
    /// Regression: "Impacto Relativo" `range / |mean| · 100` (0 when the mean is 0,
    /// as Python); `None` in classification. Kept for parity, but note that it
    /// depends on where the target's zero is: the same curve shifted by a constant
    /// (degrees Celsius instead of Kelvin, a target centred on 0) gives a very
    /// different percentage, and a mean of the predictions near 0 makes it explode.
    /// `range` alone is the scale-honest figure.
    pub relative_impact_pct: Option<f64>,
}

/// Verdict of the analysis (Python's three notices).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Influence {
    /// "🎯 **Característica muy influyente:** …" (success).
    High,
    /// "📊 **Característica moderadamente influyente:** …" (warning).
    Moderate,
    /// "📉 **Característica poco influyente:** …" (info).
    Low,
}

/// The whole exploration of one sample and one feature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExploreView {
    /// Row of the dataset used as the base sample.
    pub sample_index: usize,
    /// Explored column.
    pub feature: usize,
    /// The base sample in original units (all features).
    pub base: Vec<f64>,
    /// `base[feature]` (the "Valor Base" vertical line).
    pub base_value: f64,
    pub base_prediction: RowPrediction,
    pub kind: OutputKind,
    /// `false` with a tanh output (the probabilities are not probabilities).
    pub output_is_probability: bool,
    /// The values of the feature: `linspace(min, max, n_points)` over the dataset.
    pub xs: Vec<f64>,
    pub curve: ExploreCurve,
    pub stats: ExploreStats,
    pub influence: Influence,
}

/// Python's thresholds on the probability range (binary; base class in multiclass).
pub fn probability_influence(range: f64) -> Influence {
    if range > PROBABILITY_RANGE_HIGH {
        Influence::High
    } else if range > PROBABILITY_RANGE_MODERATE {
        Influence::Moderate
    } else {
        Influence::Low
    }
}

/// Python's thresholds on the relative impact (%) of a regression.
pub fn regression_influence(relative_impact_pct: f64) -> Influence {
    if relative_impact_pct > RELATIVE_IMPACT_HIGH {
        Influence::High
    } else if relative_impact_pct > RELATIVE_IMPACT_MODERATE {
        Influence::Moderate
    } else {
        Influence::Low
    }
}

/// Range (float32, like `max(list) - min(list)` of `np.float32`), population std
/// and mean of a curve.
fn curve_stats(values: &[f64]) -> (f64, f64, f64) {
    let n = values.len();
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max) as f32;
    let min = values.iter().copied().fold(f64::INFINITY, f64::min) as f32;
    let mean = pairwise_sum_by(n, |i| values[i]) / n as f64;
    let var = pairwise_sum_by(n, |i| (values[i] - mean) * (values[i] - mean)) / n as f64;
    (f64::from(max - min), var.sqrt(), mean)
}

fn invalid(reason: String) -> ModelError {
    ModelError::InvalidSelection { reason }
}

/// 🎲 Exploración interactiva: the network's output when `feature` of the row
/// `sample_index` of `x` (the **whole dataset**, original units: Python's
/// `nn_df`) takes `n_points` evenly spaced values from the column's minimum to its
/// maximum over `x`, with every other feature fixed at the sample's values.
///
/// The rows are predicted in one call (one forward pass; every row is computed
/// independently, so this equals Python's one `predict` per value up to Keras'
/// float32 rounding). Errors: `InvalidSelection` for a row, feature or number of
/// points out of range ([`EXPLORE_MIN_POINTS`]..=[`EXPLORE_MAX_POINTS`]),
/// `NonFinite` if the sample or the column has NaN/∞, `Cancelled`, and the
/// network's own errors (`Numerical` for a diverged network).
pub fn nn_explore(
    network: &dyn NetworkOutput,
    x: ArrayView2<'_, f64>,
    sample_index: usize,
    feature: usize,
    n_points: usize,
    control: &TrainControl<'_>,
) -> Result<ExploreView, ModelError> {
    let p = network.n_features();
    if x.ncols() != p {
        return Err(ModelError::FeatureMismatch {
            expected: p,
            found: x.ncols(),
        });
    }
    if sample_index >= x.nrows() {
        return Err(invalid(format!(
            "la muestra {sample_index} no existe (hay {} filas)",
            x.nrows()
        )));
    }
    if feature >= p {
        return Err(invalid(format!(
            "la característica {feature} no existe (hay {p})"
        )));
    }
    if !(EXPLORE_MIN_POINTS..=EXPLORE_MAX_POINTS).contains(&n_points) {
        return Err(invalid(format!(
            "entre {EXPLORE_MIN_POINTS} y {EXPLORE_MAX_POINTS} puntos de exploración"
        )));
    }
    let base: Vec<f64> = x.row(sample_index).to_vec();
    let column = x.column(feature);
    if base.iter().any(|v| !v.is_finite()) || column.iter().any(|v| !v.is_finite()) {
        return Err(ModelError::NonFinite);
    }
    control.check()?;
    let min = column.iter().copied().fold(f64::INFINITY, f64::min);
    let max = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let xs = linspace(min, max, n_points);

    // The base row first, then the explored rows: one forward pass.
    let mut rows = Array2::zeros((n_points + 1, p));
    for (i, mut row) in rows.rows_mut().into_iter().enumerate() {
        row.assign(&ndarray::aview1(&base));
        if i > 0 {
            row[feature] = xs[i - 1];
        }
    }
    let out = network.raw_output(rows.view())?;
    control.check()?;
    let k = out.ncols();
    let kind = output_kind(network, k);
    let classes = network.classes().unwrap_or(&[]).to_vec();
    let row_of = |i: usize| -> Vec<f64> { out.row(i).to_vec() };
    let base_prediction = read_row(kind, &row_of(0), &classes);

    let (curve, analysed) = match kind {
        OutputKind::Binary => {
            let probability: Vec<f64> = (1..=n_points).map(|i| out[[i, 0]]).collect();
            (
                ExploreCurve::Binary {
                    probability: probability.clone(),
                },
                probability,
            )
        }
        OutputKind::Multiclass => {
            let base_idx = argmax(&row_of(0));
            let mut cls = Vec::with_capacity(n_points);
            let mut confidence = Vec::with_capacity(n_points);
            let mut base_p = Vec::with_capacity(n_points);
            for i in 1..=n_points {
                let row = row_of(i);
                let best = argmax(&row);
                cls.push(classes.get(best).copied().unwrap_or(best));
                confidence.push(row[best]);
                base_p.push(row[base_idx]);
            }
            (
                ExploreCurve::Multiclass {
                    classes: cls,
                    confidence,
                    base_class: classes.get(base_idx).copied().unwrap_or(base_idx),
                    base_class_probability: base_p.clone(),
                },
                base_p,
            )
        }
        OutputKind::Regression => {
            let values: Vec<f64> = (1..=n_points).map(|i| out[[i, 0]]).collect();
            (
                ExploreCurve::Regression {
                    values: values.clone(),
                },
                values,
            )
        }
    };
    let (range, std, mean) = curve_stats(&analysed);
    let (relative_impact_pct, influence) = match kind {
        OutputKind::Regression => {
            let impact = if mean != 0.0 {
                range / mean.abs() * 100.0
            } else {
                0.0
            };
            (Some(impact), regression_influence(impact))
        }
        _ => (None, probability_influence(range)),
    };
    Ok(ExploreView {
        sample_index,
        feature,
        base_value: base[feature],
        base,
        base_prediction,
        kind,
        output_is_probability: network.output_is_probability(),
        xs,
        curve,
        stats: ExploreStats {
            range,
            std,
            mean,
            relative_impact_pct,
        },
        influence,
    })
}

// ---------------------------------------------------------------------------
// Training ranges (warnings for values the network never saw)
// ---------------------------------------------------------------------------

/// Minimum and maximum of one feature over the training rows, in original units.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FeatureRange {
    pub min: f64,
    pub max: f64,
}

impl FeatureRange {
    /// Whether `v` lies inside `[min, max]`.
    pub fn contains(&self, v: f64) -> bool {
        v >= self.min && v <= self.max
    }
}

/// Range of every feature over the **unscaled** training rows `train` of `x` (the
/// dataset in original units; the network's split, fit + validation). NaN/∞ values
/// are skipped; a feature without finite values gets `[NaN, NaN]` (nothing is
/// inside it). Errors: `InvalidSelection` if a row is out of range.
///
/// The exploration axis is NOT this range: as in Python, it is the min–max of the
/// whole dataset (`nn_explore`). This one is for the form's and the batch's
/// "fuera del rango de entrenamiento" warnings.
pub fn training_ranges(
    x: ArrayView2<'_, f64>,
    train: &[usize],
) -> Result<Vec<FeatureRange>, ModelError> {
    if let Some(&r) = train.iter().find(|&&r| r >= x.nrows()) {
        return Err(invalid(format!(
            "la fila {r} no existe (hay {} filas)",
            x.nrows()
        )));
    }
    Ok((0..x.ncols())
        .map(|j| {
            let mut range = FeatureRange {
                min: f64::INFINITY,
                max: f64::NEG_INFINITY,
            };
            for &r in train {
                let v = x[[r, j]];
                if v.is_finite() {
                    range.min = range.min.min(v);
                    range.max = range.max.max(v);
                }
            }
            if range.min > range.max {
                FeatureRange {
                    min: f64::NAN,
                    max: f64::NAN,
                }
            } else {
                range
            }
        })
        .collect())
}

/// For each row of `rows` (model column order), the columns whose value is outside
/// its training range; rows without any are left out. `(row, columns)` pairs in
/// row order.
pub fn out_of_range(
    rows: ArrayView2<'_, f64>,
    ranges: &[FeatureRange],
) -> Vec<(usize, Vec<usize>)> {
    rows.rows()
        .into_iter()
        .enumerate()
        .filter_map(|(i, row)| {
            let cols: Vec<usize> = row
                .iter()
                .zip(ranges)
                .enumerate()
                .filter(|(_, (v, r))| !r.contains(**v))
                .map(|(j, _)| j)
                .collect();
            (!cols.is_empty()).then_some((i, cols))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 📊 Predicción por lotes
// ---------------------------------------------------------------------------

/// Which columns of an uploaded file are used (Python: `set(feature_cols) -
/// set(new_df.columns)` and the reverse).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchColumns {
    /// For every feature of the model, the index of its column in the file
    /// (`None` when it is missing).
    pub positions: Vec<Option<usize>>,
    /// "❌ Faltan características: …" (model order). Non-empty = no prediction.
    pub missing: Vec<String>,
    /// "⚠️ Características adicionales (serán ignoradas): …" (file order).
    pub extra: Vec<String>,
}

impl BatchColumns {
    /// Whether every feature was found.
    pub fn complete(&self) -> bool {
        self.missing.is_empty()
    }
}

/// Matches the model's feature columns (`X.columns`, the names the expected-format
/// table shows) with the columns of a file, by exact name. Python reported the
/// sets in arbitrary order; here they keep the model's and the file's order.
pub fn batch_columns(features: &[String], file_columns: &[String]) -> BatchColumns {
    let positions: Vec<Option<usize>> = features
        .iter()
        .map(|f| file_columns.iter().position(|c| c == f))
        .collect();
    let missing = features
        .iter()
        .zip(&positions)
        .filter(|(_, p)| p.is_none())
        .map(|(f, _)| f.clone())
        .collect();
    let extra = file_columns
        .iter()
        .filter(|c| !features.contains(c))
        .cloned()
        .collect();
    BatchColumns {
        positions,
        missing,
        extra,
    }
}

/// The result columns of the batch, by kind of output. A row whose outputs are not
/// all finite ([`BatchView::finite`] = `false`) has `None` as its class (it is not
/// given class 0, nor any other) and keeps its NaN/∞ outputs as they are.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BatchPredictions {
    /// Columns "Predicción", "Probabilidad", "Confianza".
    Binary {
        predicted: Vec<Option<usize>>,
        probability: Vec<f64>,
        confidence: Vec<f64>,
    },
    /// Columns "Predicción", "Confianza"; NEW: every output of each row
    /// (`probabilities[row][class]`, the network's class order) for a table or a
    /// download with one column per class.
    Multiclass {
        predicted: Vec<Option<usize>>,
        confidence: Vec<f64>,
        probabilities: Vec<Vec<f64>>,
    },
    /// Column "Predicción".
    Regression { values: Vec<f64> },
}

/// "✅ Predicciones generadas para {rows} muestras".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchView {
    pub rows: usize,
    pub kind: OutputKind,
    /// Class codes of the network (the order of `probabilities`); empty in
    /// regression.
    pub classes: Vec<usize>,
    /// `false` with a tanh output.
    pub output_is_probability: bool,
    /// One per row: whether every output of the row is finite. A `false` row (the
    /// float32 forward pass overflowed: huge inputs, or weights near divergence)
    /// has no class (`predicted[i] = None`); the UI must say it has no prediction.
    pub finite: Vec<bool>,
    pub predictions: BatchPredictions,
}

/// Rejects the inputs the float32 network cannot read: NaN, ±∞ and finite values
/// with `|v| > f32::MAX`, which become ±∞ when converted to float32 (Keras would
/// cast them too). The error names the first such cell (0-based row and column, in
/// the model's column order).
pub fn check_f32_inputs(rows: ArrayView2<'_, f64>) -> Result<(), ModelError> {
    for (row, values) in rows.rows().into_iter().enumerate() {
        if let Some(column) = values
            .iter()
            .position(|v| !v.is_finite() || v.abs() > f64::from(f32::MAX))
        {
            return Err(ModelError::NonFiniteCell { row, column });
        }
    }
    Ok(())
}

/// 📊 Predicción por lotes of `rows` (original units, the model's column order:
/// see [`batch_columns`]). Predicted in chunks of [`BATCH_CHUNK_ROWS`] with a
/// cancellation check before each one; at most [`BATCH_MAX_ROWS`] rows.
///
/// Errors: `Empty`, `InvalidInput` (too many rows), `FeatureMismatch`,
/// `NonFiniteCell { row, column }` (the first missing, infinite or beyond-float32
/// value, see [`check_f32_inputs`]: Python's `scaler.transform` would fail or
/// predict NaN), `Cancelled` and the network's own errors.
///
/// Rows are predicted independently: a row whose output is NaN/∞ is flagged in
/// [`BatchView::finite`] and gets no class, and the other rows are unaffected.
pub fn nn_predict_batch(
    network: &dyn NetworkOutput,
    rows: ArrayView2<'_, f64>,
    control: &TrainControl<'_>,
) -> Result<BatchView, ModelError> {
    let n = rows.nrows();
    if n == 0 {
        return Err(ModelError::Empty);
    }
    if n > BATCH_MAX_ROWS {
        return Err(ModelError::InvalidInput {
            reason: format!("el archivo tiene {n} filas y el máximo es {BATCH_MAX_ROWS}"),
        });
    }
    if rows.ncols() != network.n_features() {
        return Err(ModelError::FeatureMismatch {
            expected: network.n_features(),
            found: rows.ncols(),
        });
    }
    check_f32_inputs(rows)?;
    let classes = network.classes().unwrap_or(&[]).to_vec();
    let mut kind = None;
    let mut finite = Vec::with_capacity(n);
    let mut predicted = Vec::with_capacity(n);
    let mut first = Vec::with_capacity(n);
    let mut confidence = Vec::with_capacity(n);
    let mut probabilities = Vec::new();
    let mut start = 0;
    while start < n {
        control.check()?;
        let end = (start + BATCH_CHUNK_ROWS).min(n);
        let out = network.raw_output(rows.slice(ndarray::s![start..end, ..]))?;
        let k = *kind.get_or_insert_with(|| output_kind(network, out.ncols()));
        for row in out.rows() {
            let row = row.to_vec();
            let ok = row.iter().all(|v| v.is_finite());
            finite.push(ok);
            match read_row(k, &row, &classes) {
                RowPrediction::Binary {
                    class,
                    probability,
                    confidence: c,
                } => {
                    predicted.push(ok.then_some(class));
                    first.push(probability);
                    confidence.push(c);
                }
                RowPrediction::Multiclass {
                    class,
                    confidence: c,
                    probabilities: p,
                } => {
                    predicted.push(ok.then_some(class));
                    confidence.push(c);
                    probabilities.push(p);
                }
                RowPrediction::Regression { value } => first.push(value),
            }
        }
        start = end;
    }
    let kind = kind.unwrap_or(OutputKind::Regression);
    let predictions = match kind {
        OutputKind::Binary => BatchPredictions::Binary {
            predicted,
            probability: first,
            confidence,
        },
        OutputKind::Multiclass => BatchPredictions::Multiclass {
            predicted,
            confidence,
            probabilities,
        },
        OutputKind::Regression => BatchPredictions::Regression { values: first },
    };
    Ok(BatchView {
        rows: n,
        kind,
        classes,
        output_is_probability: network.output_is_probability(),
        finite,
        predictions,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::dataset::Target;
    use crate::model::Prediction;
    use std::sync::atomic::AtomicBool;

    /// out = [w · x + b] (regression), sigmoid(...) (binary) or softmax of
    /// `[x0, x1, 0]` (multiclass).
    #[derive(Debug)]
    struct Toy(OutputKind);

    impl NetworkOutput for Toy {
        fn raw_output(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>, ModelError> {
            let k = if self.0 == OutputKind::Multiclass {
                3
            } else {
                1
            };
            let mut out = Array2::zeros((x.nrows(), k));
            for (i, r) in x.rows().into_iter().enumerate() {
                match self.0 {
                    OutputKind::Regression => out[[i, 0]] = 2.0 * r[0] + r[1],
                    OutputKind::Binary => out[[i, 0]] = 1.0 / (1.0 + (-r[0]).exp()),
                    OutputKind::Multiclass => {
                        let z = [r[0].exp(), r[1].exp(), 1.0];
                        let s: f64 = z.iter().sum();
                        for j in 0..3 {
                            out[[i, j]] = z[j] / s;
                        }
                    }
                }
            }
            Ok(out)
        }
        fn output_is_probability(&self) -> bool {
            true
        }
    }

    impl Predictor for Toy {
        fn task(&self) -> TaskType {
            if self.0 == OutputKind::Regression {
                TaskType::Regression
            } else {
                TaskType::Classification
            }
        }
        fn n_features(&self) -> usize {
            2
        }
        fn classes(&self) -> Option<&[usize]> {
            match self.0 {
                OutputKind::Regression => None,
                OutputKind::Binary => Some(&[0, 1]),
                OutputKind::Multiclass => Some(&[0, 1, 2]),
            }
        }
        fn predict(&self, _x: ArrayView2<'_, f64>) -> Result<Prediction, ModelError> {
            Err(ModelError::Empty)
        }
        fn predict_proba(
            &self,
            _x: ArrayView2<'_, f64>,
        ) -> Option<Result<Array2<f64>, ModelError>> {
            None
        }
        fn refit(
            &self,
            _x: ArrayView2<'_, f64>,
            _y: &Target,
            _c: &TrainControl<'_>,
        ) -> Result<Box<dyn Predictor>, ModelError> {
            Err(ModelError::Empty)
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn data() -> Array2<f64> {
        ndarray::array![[-2.0, 1.0], [0.0, 0.5], [3.0, -1.0]]
    }

    #[test]
    fn regression_curve_and_impact() {
        let v = nn_explore(
            &Toy(OutputKind::Regression),
            data().view(),
            1,
            0,
            EXPLORE_DEFAULT_POINTS,
            &TrainControl::none(),
        )
        .unwrap();
        assert_eq!(v.xs.len(), 50);
        assert_eq!((v.xs[0], v.xs[49]), (-2.0, 3.0));
        let ExploreCurve::Regression { values } = &v.curve else {
            panic!()
        };
        assert!((values[0] - (-3.5)).abs() < 1e-12 && (values[49] - 6.5).abs() < 1e-12);
        assert!((v.stats.range - 10.0).abs() < 1e-6);
        assert!((v.stats.mean - 1.5).abs() < 1e-12);
        let impact = v.stats.relative_impact_pct.unwrap();
        assert!((impact - 10.0 / 1.5 * 100.0).abs() < 1e-4);
        assert_eq!(v.influence, Influence::High);
        assert_eq!(v.base_prediction, RowPrediction::Regression { value: 0.5 });
        assert_eq!(v.base_value, 0.0);
    }

    #[test]
    fn binary_and_multiclass_curves() {
        let ctl = TrainControl::none();
        let v = nn_explore(&Toy(OutputKind::Binary), data().view(), 0, 1, 5, &ctl).unwrap();
        // x0 is fixed at -2: the probability does not move with x1.
        assert_eq!(v.stats.range, 0.0);
        assert_eq!(v.influence, Influence::Low);
        let RowPrediction::Binary {
            class, confidence, ..
        } = v.base_prediction
        else {
            panic!()
        };
        assert_eq!(class, 0);
        assert!(confidence > 0.5);
        let v = nn_explore(&Toy(OutputKind::Multiclass), data().view(), 2, 0, 9, &ctl).unwrap();
        let ExploreCurve::Multiclass {
            classes,
            base_class,
            base_class_probability,
            ..
        } = &v.curve
        else {
            panic!()
        };
        assert_eq!(*base_class, 0);
        // x0 = -2, x1 = -1: softmax of [e⁻², e⁻¹, 1] → class 2; the base [3, -1] → class 0.
        assert_eq!(classes[0], 2);
        assert_eq!(base_class_probability.len(), 9);
        assert_eq!(v.influence, Influence::High);
    }

    #[test]
    fn explore_rejects_bad_requests() {
        let ctl = TrainControl::none();
        let toy = Toy(OutputKind::Regression);
        let x = data();
        assert!(nn_explore(&toy, x.view(), 3, 0, 50, &ctl).is_err());
        assert!(nn_explore(&toy, x.view(), 0, 2, 50, &ctl).is_err());
        assert!(nn_explore(&toy, x.view(), 0, 0, 1, &ctl).is_err());
        assert!(nn_explore(&toy, x.view(), 0, 0, EXPLORE_MAX_POINTS + 1, &ctl).is_err());
        let mut bad = x.clone();
        bad[[1, 0]] = f64::NAN;
        assert_eq!(
            nn_explore(&toy, bad.view(), 0, 0, 50, &ctl),
            Err(ModelError::NonFinite)
        );
        let flag = AtomicBool::new(true);
        assert_eq!(
            nn_explore(&toy, x.view(), 0, 0, 50, &ctl.with_cancel(&flag)),
            Err(ModelError::Cancelled)
        );
    }

    #[test]
    fn batch_by_kind_and_limits() {
        let ctl = TrainControl::none();
        let x = data();
        let b = nn_predict_batch(&Toy(OutputKind::Binary), x.view(), &ctl).unwrap();
        let BatchPredictions::Binary {
            predicted,
            probability,
            confidence,
        } = &b.predictions
        else {
            panic!()
        };
        assert_eq!(predicted, &[Some(0), Some(0), Some(1)]);
        assert_eq!(b.finite, vec![true; 3]);
        assert_eq!(probability[1], 0.5); // exactly 0.5 is class 0 (p > 0.5)
        assert_eq!(confidence[1], 0.5);
        let b = nn_predict_batch(&Toy(OutputKind::Multiclass), x.view(), &ctl).unwrap();
        let BatchPredictions::Multiclass {
            probabilities,
            predicted,
            ..
        } = &b.predictions
        else {
            panic!()
        };
        assert_eq!(probabilities.len(), 3);
        assert_eq!(predicted, &[Some(1), Some(1), Some(0)]);
        let empty = Array2::<f64>::zeros((0, 2));
        assert_eq!(
            nn_predict_batch(&Toy(OutputKind::Binary), empty.view(), &ctl),
            Err(ModelError::Empty)
        );
        let flag = AtomicBool::new(true);
        assert_eq!(
            nn_predict_batch(&Toy(OutputKind::Binary), x.view(), &ctl.with_cancel(&flag)),
            Err(ModelError::Cancelled)
        );
        let wide = Array2::<f64>::zeros((2, 3));
        assert!(nn_predict_batch(&Toy(OutputKind::Binary), wide.view(), &ctl).is_err());
    }

    /// Review M1: a row whose output is NaN gets no class (it used to be class 0),
    /// and the other rows are unaffected.
    #[test]
    fn non_finite_outputs_have_no_class() {
        let ctl = TrainControl::none();
        // x0 = ±∞ is rejected before predicting; a finite x0 whose exp overflows
        // in the toy softmax gives NaN outputs (∞/∞).
        let x = ndarray::array![[800.0, 0.0], [0.0, 3.0]];
        let b = nn_predict_batch(&Toy(OutputKind::Multiclass), x.view(), &ctl).unwrap();
        assert_eq!(b.finite, vec![false, true]);
        let BatchPredictions::Multiclass { predicted, .. } = &b.predictions else {
            panic!()
        };
        assert_eq!(predicted, &[None, Some(1)]);
        // Binary: a NaN probability is not "p ≤ 0.5 → class 0".
        #[derive(Debug)]
        struct NanBinary;
        impl NetworkOutput for NanBinary {
            fn raw_output(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>, ModelError> {
                Ok(x.slice(ndarray::s![.., 0..1])
                    .mapv(|v| if v > 0.0 { f64::NAN } else { 0.9 }))
            }
            fn output_is_probability(&self) -> bool {
                true
            }
        }
        impl Predictor for NanBinary {
            fn task(&self) -> TaskType {
                TaskType::Classification
            }
            fn n_features(&self) -> usize {
                2
            }
            fn classes(&self) -> Option<&[usize]> {
                Some(&[0, 1])
            }
            fn predict(&self, _x: ArrayView2<'_, f64>) -> Result<Prediction, ModelError> {
                Err(ModelError::Empty)
            }
            fn predict_proba(
                &self,
                _x: ArrayView2<'_, f64>,
            ) -> Option<Result<Array2<f64>, ModelError>> {
                None
            }
            fn refit(
                &self,
                _x: ArrayView2<'_, f64>,
                _y: &Target,
                _c: &TrainControl<'_>,
            ) -> Result<Box<dyn Predictor>, ModelError> {
                Err(ModelError::Empty)
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        let x = ndarray::array![[1.0, 0.0], [-1.0, 0.0]];
        let b = nn_predict_batch(&NanBinary, x.view(), &ctl).unwrap();
        assert_eq!(b.finite, vec![false, true]);
        let BatchPredictions::Binary { predicted, .. } = &b.predictions else {
            panic!()
        };
        assert_eq!(predicted, &[None, Some(1)]);
    }

    /// Review M1: NaN, ∞ and values beyond float32 are rejected with their cell.
    #[test]
    fn inputs_beyond_f32_are_rejected_with_their_cell() {
        let ctl = TrainControl::none();
        let toy = Toy(OutputKind::Regression);
        for bad in [f64::NAN, f64::INFINITY, 1e39, -4e38] {
            let mut x = data();
            x[[2, 1]] = bad;
            assert_eq!(
                nn_predict_batch(&toy, x.view(), &ctl),
                Err(ModelError::NonFiniteCell { row: 2, column: 1 }),
                "{bad}"
            );
        }
        let mut x = data();
        x[[0, 0]] = f64::from(f32::MAX);
        assert!(nn_predict_batch(&toy, x.view(), &ctl).is_ok());
        let msg = ModelError::NonFiniteCell { row: 2, column: 1 }.to_string();
        assert!(msg.contains("fila 3") && msg.contains("columna 2"), "{msg}");
    }

    #[test]
    fn columns_are_matched_by_name() {
        let s = |v: &[&str]| v.iter().map(|x| (*x).to_owned()).collect::<Vec<_>>();
        let c = batch_columns(&s(&["a", "b", "c"]), &s(&["c", "x", "a"]));
        assert_eq!(c.positions, vec![Some(2), None, Some(0)]);
        assert_eq!(c.missing, s(&["b"]));
        assert_eq!(c.extra, s(&["x"]));
        assert!(!c.complete());
        assert!(batch_columns(&s(&["a"]), &s(&["a"])).complete());
    }

    #[test]
    fn training_ranges_use_the_given_rows() {
        let x = data();
        let r = training_ranges(x.view(), &[0, 2]).unwrap();
        assert_eq!(
            r[0],
            FeatureRange {
                min: -2.0,
                max: 3.0
            }
        );
        assert_eq!(
            r[1],
            FeatureRange {
                min: -1.0,
                max: 1.0
            }
        );
        let r = training_ranges(x.view(), &[1]).unwrap();
        let out = out_of_range(x.view(), &r);
        assert_eq!(out, vec![(0, vec![0, 1]), (2, vec![0, 1])]);
        assert!(training_ranges(x.view(), &[3]).is_err());
    }

    #[test]
    fn thresholds_are_strict() {
        assert_eq!(probability_influence(0.3), Influence::Moderate);
        assert_eq!(probability_influence(0.1), Influence::Low);
        assert_eq!(regression_influence(20.0), Influence::Moderate);
        assert_eq!(regression_influence(5.0), Influence::Low);
        assert_eq!(regression_influence(20.1), Influence::High);
    }
}

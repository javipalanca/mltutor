//! Numbers behind 🧠 Análisis de Pesos (C25) and 📉 Análisis de Activaciones (C26).
//!
//! `viz/nn.py::show_weights_analysis_tab` and `show_layer_activations_tab`
//! (`tools/nn_reference.py::weight_stats` / `activation_stats`). The network is
//! float32; the statistics are computed in f64 from those values (`np.std` =
//! population std).
//!
//! The histograms (K2) are binned **here**, with Plotly's own algorithm for
//! `go.Histogram(x, nbinsx)` ([`crate::stats::histogram_plotly`], checked against
//! plotly.js), so the views send `{ edges, counts }` instead of every weight or
//! activation (a 5 × 256 network has 264 000 weights): [`weights_view`] and
//! [`activations_view`]. Per neuron only means and standard deviations travel.
//!
//! Layer names follow the architecture (D77): "Oculta 1", "Oculta 2", "Salida";
//! the Keras name (`dense_1`, `output_layer`) goes alongside.

use serde::{Deserialize, Serialize};

use crate::stats::{histogram_plotly, Histogram};

/// A dense layer seen from the views: `kernel` is row-major `inputs × units`.
#[derive(Debug, Clone, Copy)]
pub struct DenseView<'a> {
    pub inputs: usize,
    pub units: usize,
    pub kernel: &'a [f32],
    pub bias: &'a [f32],
}

fn mean(v: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = v.clone().count();
    if n == 0 {
        return f64::NAN;
    }
    v.sum::<f64>() / n as f64
}

fn std(v: impl Iterator<Item = f64> + Clone) -> f64 {
    let m = mean(v.clone());
    mean(v.map(|x| (x - m) * (x - m))).sqrt()
}

fn near_zero_pct(v: &[f32]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().filter(|w| w.abs() < 1e-3).count() as f64 / v.len() as f64 * 100.0
}

/// Python's `round(x, 1)` (correctly rounded on the binary value, like `{:.1f}`).
fn round1(x: f64) -> f64 {
    format!("{x:.1}").parse().unwrap_or(x)
}

/// One row of "📋 Resumen por Capa".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerWeightStats {
    /// `[inputs, units]`.
    pub shape: [usize; 2],
    pub mean_w: f64,
    pub std_w: f64,
    /// `% |w|<1e-3`, rounded to 1 decimal as in the table.
    pub near_zero_pct: f64,
    /// The same, unrounded (the caption shows `{:.1f}`).
    pub near_zero_pct_raw: f64,
    pub mean_b: f64,
    pub std_b: f64,
    pub n_params: usize,
    pub min_w: f64,
    pub max_w: f64,
    pub min_b: f64,
    pub max_b: f64,
}

/// Message under the global metrics (`[pesos]`), by the global std.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightsVerdict {
    /// `global_std < 0.01` (warning).
    TooSmall,
    /// `global_std > 2` (warning).
    TooLarge,
    /// success.
    Healthy,
}

/// Bullets of "### 🩺 Evaluación General", in Python's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightsInsight {
    /// `% |w|<1e-3 > 75`.
    ManyNearZero,
    /// `global_std < 0.01`.
    LowVariance,
    /// `global_std > 2`.
    HighVariance,
    /// None of the above.
    Balanced,
}

/// Everything of 🧠 Pesos except the raw values (the view sends those to Plotly).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightStats {
    pub layers: Vec<LayerWeightStats>,
    /// Over all the kernels (no biases).
    pub global_mean: f64,
    pub global_std: f64,
    pub global_near_zero_pct: f64,
    pub verdict: WeightsVerdict,
    pub insights: Vec<WeightsInsight>,
}

/// `weight_stats(params)`.
pub fn weight_stats(layers: &[DenseView<'_>]) -> WeightStats {
    let rows = layers
        .iter()
        .map(|l| {
            let w = || l.kernel.iter().map(|&v| f64::from(v));
            let b = || l.bias.iter().map(|&v| f64::from(v));
            let nz = near_zero_pct(l.kernel);
            LayerWeightStats {
                shape: [l.inputs, l.units],
                mean_w: mean(w()),
                std_w: std(w()),
                near_zero_pct: round1(nz),
                near_zero_pct_raw: nz,
                mean_b: mean(b()),
                std_b: std(b()),
                n_params: l.kernel.len() + l.bias.len(),
                min_w: w().fold(f64::INFINITY, f64::min),
                max_w: w().fold(f64::NEG_INFINITY, f64::max),
                min_b: b().fold(f64::INFINITY, f64::min),
                max_b: b().fold(f64::NEG_INFINITY, f64::max),
            }
        })
        .collect();
    let all = || {
        layers
            .iter()
            .flat_map(|l| l.kernel.iter().map(|&v| f64::from(v)))
    };
    let all_w: Vec<f32> = layers
        .iter()
        .flat_map(|l| l.kernel.iter().copied())
        .collect();
    let global_std = std(all());
    let global_near_zero_pct = near_zero_pct(&all_w);
    let verdict = if global_std < 0.01 {
        WeightsVerdict::TooSmall
    } else if global_std > 2.0 {
        WeightsVerdict::TooLarge
    } else {
        WeightsVerdict::Healthy
    };
    let mut insights = Vec::new();
    if global_near_zero_pct > 75.0 {
        insights.push(WeightsInsight::ManyNearZero);
    }
    if global_std < 0.01 {
        insights.push(WeightsInsight::LowVariance);
    }
    if global_std > 2.0 {
        insights.push(WeightsInsight::HighVariance);
    }
    if insights.is_empty() {
        insights.push(WeightsInsight::Balanced);
    }
    WeightStats {
        layers: rows,
        global_mean: mean(all()),
        global_std,
        global_near_zero_pct,
        verdict,
        insights,
    }
}

/// Default thresholds of ⚙️ Parámetros de Análisis.
pub const DEFAULT_SAMPLE_SIZE: usize = 50;
pub const DEFAULT_DEAD_THRESHOLD: f64 = 0.0;
pub const DEFAULT_CONSTANT_STD: f64 = 1e-3;

/// One hidden layer of 📉 Activaciones.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerActivationStats {
    pub units: usize,
    /// Neurons whose activations are all `<= dead_threshold`.
    pub dead: Vec<usize>,
    /// Not dead and std `< constant_std`.
    pub constant: Vec<usize>,
    /// % of activations `>= 0.99` when the layer's maximum is `<= 1.2`, else 0.
    pub saturation_pct: f64,
    /// Mean over neurons of the per-neuron mean / std.
    pub mean_act: f64,
    pub std_act: f64,
    pub act_max: f64,
    /// Mean of every neuron over the rows ("barras de medias").
    pub neuron_means: Vec<f64>,
    /// Population standard deviation of every neuron over the rows.
    pub neuron_stds: Vec<f64>,
}

/// `activation_stats` of one hidden layer: `acts` is row-major `rows × units`
/// (after the activation, before dropout).
pub fn activation_stats(
    acts: &[f32],
    units: usize,
    dead_threshold: f64,
    constant_std: f64,
) -> LayerActivationStats {
    let rows = acts.len().checked_div(units).unwrap_or(0);
    let col = |j: usize| (0..rows).map(move |i| f64::from(acts[i * units + j]));
    let means: Vec<f64> = (0..units).map(|j| mean(col(j))).collect();
    let stds: Vec<f64> = (0..units).map(|j| std(col(j))).collect();
    let dead: Vec<usize> = (0..units)
        .filter(|&j| col(j).all(|a| a <= dead_threshold))
        .collect();
    let constant: Vec<usize> = (0..units)
        .filter(|j| !dead.contains(j) && stds[*j] < constant_std)
        .collect();
    let act_max = acts
        .iter()
        .map(|&a| f64::from(a))
        .fold(f64::NEG_INFINITY, f64::max);
    let saturation_pct = if act_max <= 1.2 && !acts.is_empty() {
        acts.iter().filter(|&&a| a >= 0.99).count() as f64 / acts.len() as f64 * 100.0
    } else {
        0.0
    };
    LayerActivationStats {
        units,
        saturation_pct,
        mean_act: mean(means.iter().copied()),
        std_act: mean(stds.iter().copied()),
        act_max,
        neuron_means: means,
        neuron_stds: stds,
        dead,
        constant,
    }
}

/// `% of dead neurons` over all the hidden layers (aviso global: > 50 error, > 20
/// warning).
pub fn dead_pct(layers: &[LayerActivationStats]) -> f64 {
    let total: usize = layers.iter().map(|l| l.units).sum();
    let dead: usize = layers.iter().map(|l| l.dead.len()).sum();
    if total == 0 {
        0.0
    } else {
        dead as f64 / total as f64 * 100.0
    }
}

/// State of a neuron in the map (`estado` of the Python tooltip).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuronState {
    Active,
    Dead,
    Constant,
    Output,
}

/// Tooltip of a neuron of the map: statistics of its incoming weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuronInfo {
    pub bias: f64,
    pub mean_w: f64,
    pub std_w: f64,
    pub max_abs_w: f64,
    pub state: NeuronState,
}

/// "🗺️ Mapa de Neuronas Muertas (💀) y Constantes (😴)" (D80; drawn by
/// `NetworkDiagram.svelte`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkMapView {
    /// `[input, hidden…, output]`.
    pub layers: Vec<usize>,
    /// Per dense layer (not the input), one entry per neuron.
    pub neurons: Vec<Vec<NeuronInfo>>,
}

/// The map of a network from its dense layers and the stats of its hidden ones.
pub fn network_map(dense: &[DenseView<'_>], hidden: &[LayerActivationStats]) -> NetworkMapView {
    let mut layers = vec![dense.first().map_or(0, |d| d.inputs)];
    layers.extend(dense.iter().map(|d| d.units));
    let neurons = dense
        .iter()
        .enumerate()
        .map(|(k, d)| {
            (0..d.units)
                .map(|j| {
                    let w = || (0..d.inputs).map(move |i| f64::from(d.kernel[i * d.units + j]));
                    let state = if k + 1 == dense.len() {
                        NeuronState::Output
                    } else {
                        match hidden.get(k) {
                            Some(h) if h.dead.contains(&j) => NeuronState::Dead,
                            Some(h) if h.constant.contains(&j) => NeuronState::Constant,
                            _ => NeuronState::Active,
                        }
                    };
                    NeuronInfo {
                        bias: d.bias.get(j).map_or(f64::NAN, |&b| f64::from(b)),
                        mean_w: mean(w()),
                        std_w: std(w()),
                        max_abs_w: w().map(f64::abs).fold(0.0, f64::max),
                        state,
                    }
                })
                .collect()
        })
        .collect();
    NetworkMapView { layers, neurons }
}

// ---------------------------------------------------------------------------
// Binned views (K2)
// ---------------------------------------------------------------------------

/// "Bins pesos" slider: 10..=120, 50 by default.
pub const DEFAULT_WEIGHT_BINS: usize = 50;
pub const WEIGHT_BINS_RANGE: (usize, usize) = (10, 120);
/// "Bins sesgos" slider: 5..=60, 20 by default.
pub const DEFAULT_BIAS_BINS: usize = 20;
pub const BIAS_BINS_RANGE: (usize, usize) = (5, 60);
/// "Mostrar histograma activaciones": `go.Histogram(nbinsx=50)` (no control).
pub const ACTIVATION_BINS: usize = 50;

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// The two histograms of one dense layer (🧠 Pesos, "#### 📊 Capa i").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerHistograms {
    pub weights: Histogram,
    pub biases: Histogram,
}

/// Everything 🧠 Análisis de Pesos draws: the statistics and, per dense layer, the
/// binned histograms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightsView {
    pub stats: WeightStats,
    /// One per dense layer, in order.
    pub histograms: Vec<LayerHistograms>,
}

/// [`weight_stats`] plus the histograms with `weight_bins`/`bias_bins` (`nbinsx`;
/// clamped to the sliders' ranges).
pub fn weights_view(layers: &[DenseView<'_>], weight_bins: usize, bias_bins: usize) -> WeightsView {
    let wb = weight_bins.clamp(WEIGHT_BINS_RANGE.0, WEIGHT_BINS_RANGE.1);
    let bb = bias_bins.clamp(BIAS_BINS_RANGE.0, BIAS_BINS_RANGE.1);
    WeightsView {
        stats: weight_stats(layers),
        histograms: layers
            .iter()
            .map(|l| LayerHistograms {
                weights: histogram_plotly(&f64s(l.kernel), wb),
                biases: histogram_plotly(&f64s(l.bias), bb),
            })
            .collect(),
    }
}

/// The "⚙️ Parámetros de Análisis" that change the numbers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ActivationParams {
    /// "Muestras": the first rows of X_test (50).
    pub sample_size: usize,
    /// "Umbral muerta" (0.0).
    pub dead_threshold: f64,
    /// "Umbral std constante" (1e-3).
    pub constant_std: f64,
    /// Bins of the activation histogram ([`ACTIVATION_BINS`]).
    pub bins: usize,
}

impl Default for ActivationParams {
    fn default() -> Self {
        Self {
            sample_size: DEFAULT_SAMPLE_SIZE,
            dead_threshold: DEFAULT_DEAD_THRESHOLD,
            constant_std: DEFAULT_CONSTANT_STD,
            bins: ACTIVATION_BINS,
        }
    }
}

/// One hidden layer of 📉 Activaciones: statistics (per neuron: means and stds)
/// and the histogram of all its activations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerActivationsView {
    pub stats: LayerActivationStats,
    pub histogram: Histogram,
}

/// Everything 📉 Análisis de Activaciones shows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivationsView {
    /// `min(sample_size, rows of X_test)`.
    pub rows_used: usize,
    pub layers: Vec<LayerActivationsView>,
    /// [`dead_pct`] over the hidden layers.
    pub dead_pct: f64,
    /// "🗺️ Mapa de Neuronas Muertas y Constantes".
    pub map: NetworkMapView,
}

/// The view from the hidden activations: `hidden[i]` is the row-major `rows_used ×
/// dense[i].units` output of hidden layer `i` (after the activation, before
/// dropout); `dense` are all the dense layers (the last one is the output).
pub fn activations_view(
    dense: &[DenseView<'_>],
    hidden: &[Vec<f32>],
    rows_used: usize,
    params: &ActivationParams,
) -> ActivationsView {
    let stats: Vec<LayerActivationStats> = hidden
        .iter()
        .zip(dense)
        .map(|(a, l)| activation_stats(a, l.units, params.dead_threshold, params.constant_std))
        .collect();
    let map = network_map(dense, &stats);
    let dead = dead_pct(&stats);
    let layers = stats
        .into_iter()
        .zip(hidden)
        .map(|(stats, a)| LayerActivationsView {
            histogram: histogram_plotly(&f64s(a), params.bins.max(1)),
            stats,
        })
        .collect();
    ActivationsView {
        rows_used,
        layers,
        dead_pct: dead,
        map,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dead_and_constant() {
        // 3 rows × 3 units: unit 0 dead, unit 1 constant 0.5, unit 2 active.
        let acts = [0.0, 0.5, 1.0, 0.0, 0.5, 2.0, -0.0, 0.5, 3.0];
        let s = activation_stats(&acts, 3, 0.0, 1e-3);
        assert_eq!(s.dead, vec![0]);
        assert_eq!(s.constant, vec![1]);
        assert_eq!(s.saturation_pct, 0.0);
        assert_eq!(s.neuron_stds[1], 0.0);
        assert!((dead_pct(&[s]) - 100.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn binned_views_are_small_and_complete() {
        let kernel: Vec<f32> = (0..600).map(|i| (i as f32 - 300.0) / 100.0).collect();
        let bias = vec![0.5_f32; 20];
        let out_k = vec![0.1_f32; 20];
        let out_b = vec![0.0_f32];
        let layers = [
            DenseView {
                inputs: 30,
                units: 20,
                kernel: &kernel,
                bias: &bias,
            },
            DenseView {
                inputs: 20,
                units: 1,
                kernel: &out_k,
                bias: &out_b,
            },
        ];
        let w = weights_view(&layers, 50, 1000);
        assert_eq!(w.histograms.len(), 2);
        assert_eq!(w.histograms[0].weights.counts.iter().sum::<u32>(), 600);
        assert_eq!(w.histograms[0].biases.counts, vec![20]);
        assert!(w.histograms[0].weights.counts.len() <= 50);
        let acts: Vec<f32> = (0..200).map(|i| (i % 20) as f32 * 0.1).collect();
        let v = activations_view(&layers, &[acts], 10, &ActivationParams::default());
        assert_eq!(v.rows_used, 10);
        assert_eq!(v.layers[0].histogram.counts.iter().sum::<u32>(), 200);
        assert_eq!(v.layers[0].stats.dead, vec![0]);
        assert_eq!(v.map.layers, vec![30, 20, 1]);
        assert!((v.dead_pct - 5.0).abs() < 1e-12);
    }

    #[test]
    fn rounding_like_python() {
        assert_eq!(round1(12.25), 12.2);
        assert_eq!(round1(0.0), 0.0);
    }
}

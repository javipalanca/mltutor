//! The "📊 Datos" exploration of a [`Dataset`] (fase1-shell-datos.md §2.4–2.6).
//!
//! The display frame is the features with their friendly names, the label
//! columns (D8) and `target` (class names for classification). Unlike Python,
//! features and target are aligned by position (Penguins bug, §11).

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::dataset::{ColumnKind, Dataset, Target, TaskType};
use crate::error::DatasetError;
use crate::frame::{Cell, DescribeTable, TablePreview};
use crate::sampling::{groupby_sample, sample_rows};
use crate::stats::{self, gaussian_kde_scott, histogram_bin_edges_auto, histogram_counts};

/// Rows of "👀 Vista previa de datos" (`df_display.head(10)`).
pub const PREVIEW_ROWS: usize = 10;
/// Bins of the regression target histogram.
pub const TARGET_HIST_BINS: usize = 30;
/// A class ratio above this is "desbalanceado".
pub const BALANCE_RATIO_LIMIT: f64 = 3.0;
/// The pairplot samples down to this number of points.
pub const PAIRPLOT_MAX_POINTS: usize = 2000;
/// Maximum number of pairplot features (slider limit).
pub const PAIRPLOT_MAX_FEATURES: usize = 6;
/// Default number of pairplot features.
pub const PAIRPLOT_DEFAULT_FEATURES: usize = 4;
/// Seed of `DataFrame.sample` in the pairplot.
pub const PAIRPLOT_SEED: u32 = 42;
/// Level name of the single, un-hued diagonal of a regression pairplot (D11).
pub const ALL_LEVEL: &str = "(todos)";
/// Name of the target column in the display frame.
pub const TARGET_COLUMN: &str = "target";

/// `df_display.head(rows)`.
pub fn preview(ds: &Dataset, rows: usize) -> TablePreview {
    let n = rows.min(ds.n_rows());
    let mut columns = ds.feature_display.clone();
    columns.extend(ds.label_columns.iter().map(|c| c.name.clone()));
    columns.push(TARGET_COLUMN.to_owned());
    let labels = ds.target_labels();
    let target_values = ds.target_numeric();
    let rows = (0..n)
        .map(|r| {
            let mut row: Vec<Cell> = ds
                .feature_kinds
                .iter()
                .enumerate()
                .map(|(c, kind)| {
                    let v = ds.features[[r, c]];
                    if *kind == ColumnKind::Bool && !v.is_nan() {
                        Cell::Bool { v: v != 0.0 }
                    } else {
                        Cell::number(v)
                    }
                })
                .collect();
            row.extend(ds.label_columns.iter().map(|c| Cell::Text {
                v: c.values.get(r).cloned().unwrap_or_default(),
            }));
            row.push(match &labels {
                Some(l) => Cell::Text {
                    v: l.get(r).map_or_else(String::new, |s| (*s).to_owned()),
                },
                None => Cell::number(target_values.get(r).copied().unwrap_or(f64::NAN)),
            });
            row
        })
        .collect();
    TablePreview {
        columns,
        index: (0..n).map(|i| i.to_string()).collect(),
        rows,
        total_rows: ds.n_rows(),
    }
}

/// `df_display.describe()`: numeric features (bool excluded) and, in regression,
/// the target. Bool features, label columns and a class target are listed in
/// `excluded_columns`.
pub fn describe(ds: &Dataset) -> DescribeTable {
    let mut included: Vec<(String, Vec<f64>)> = Vec::new();
    let mut excluded = Vec::new();
    for (c, (name, kind)) in ds.feature_display.iter().zip(&ds.feature_kinds).enumerate() {
        if kind.is_numeric() {
            included.push((name.clone(), ds.features.column(c).to_vec()));
        } else {
            excluded.push(name.clone());
        }
    }
    excluded.extend(ds.label_columns.iter().map(|c| c.name.clone()));
    match &ds.target {
        Target::Values { values } => included.push((TARGET_COLUMN.to_owned(), values.clone())),
        Target::Classes { .. } => excluded.push(TARGET_COLUMN.to_owned()),
    }
    DescribeTable::from_columns(
        included.iter().map(|(n, v)| (n.clone(), v.as_slice())),
        excluded,
    )
}

/// "🎯 Distribución del Objetivo".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TargetDistribution {
    /// Classification: `value_counts().sort_index()` over the class names
    /// (alphabetical), with the max/min ratio.
    Bar {
        labels: Vec<String>,
        counts: Vec<u32>,
        balance_ratio: f64,
        /// `balance_ratio <= 3`.
        balanced: bool,
    },
    /// Regression: `np.histogram(y, 30)` and the statistics of the expander.
    Histogram {
        edges: Vec<f64>,
        counts: Vec<u32>,
        mean: f64,
        /// ddof = 1.
        std: f64,
        min: f64,
        max: f64,
    },
}

/// Distribution of the target (§2.4 step 5).
pub fn target_distribution(ds: &Dataset) -> TargetDistribution {
    match &ds.target {
        Target::Classes { .. } => {
            let labels = ds.target_labels().unwrap_or_default();
            let vc = stats::value_counts_sorted(labels);
            let max = vc.iter().map(|(_, c)| *c).max().unwrap_or(0);
            let min = vc.iter().map(|(_, c)| *c).min().unwrap_or(0);
            let balance_ratio = if min == 0 {
                f64::NAN
            } else {
                f64::from(max) / f64::from(min)
            };
            TargetDistribution::Bar {
                labels: vc.iter().map(|(l, _)| l.clone()).collect(),
                counts: vc.iter().map(|(_, c)| *c).collect(),
                balance_ratio,
                balanced: balance_ratio <= BALANCE_RATIO_LIMIT,
            }
        }
        Target::Values { values } => {
            let y: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
            let d = stats::describe_column(&y);
            let h = stats::histogram(&y, TARGET_HIST_BINS);
            TargetDistribution::Histogram {
                edges: h.edges,
                counts: h.counts,
                mean: d[1],
                std: d[2],
                min: d[3],
                max: d[7],
            }
        }
    }
}

/// The correlation heatmap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrelationMatrix {
    /// `feature_columns` (original names, not the friendly ones) + `"target"`.
    pub labels: Vec<String>,
    /// `n × n`, symmetric; NaN where a column is constant.
    pub values: Array2<f64>,
    /// Minimum of the finite cells of the strict lower triangle (what Python shows).
    pub visible_min: Option<f64>,
    /// Maximum of the finite cells of the strict lower triangle.
    pub visible_max: Option<f64>,
    /// Columns whose whole row/column is NaN (constant columns: Digits 0, 32, 39).
    pub nan_columns: Vec<String>,
}

/// `pd.concat([X, y_codes], axis=1).corr()` (Pearson).
pub fn correlation_matrix(ds: &Dataset) -> CorrelationMatrix {
    let n = ds.n_rows();
    let k = ds.n_features();
    let target = ds.target_numeric();
    let data = Array2::from_shape_fn((n, k + 1), |(r, c)| {
        if c < k {
            ds.features[[r, c]]
        } else {
            target[r]
        }
    });
    let values = stats::pearson_matrix(data.view());
    let mut labels = ds.feature_columns.clone();
    labels.push(TARGET_COLUMN.to_owned());

    let mut visible_min: Option<f64> = None;
    let mut visible_max: Option<f64> = None;
    for i in 0..=k {
        for j in 0..i {
            let v = values[[i, j]];
            if v.is_finite() {
                visible_min = Some(visible_min.map_or(v, |m| m.min(v)));
                visible_max = Some(visible_max.map_or(v, |m| m.max(v)));
            }
        }
    }
    let nan_columns = (0..=k)
        .filter(|&c| values.column(c).iter().all(|v| v.is_nan()))
        .map(|c| labels[c].clone())
        .collect();
    CorrelationMatrix {
        labels,
        values,
        visible_min,
        visible_max,
        nan_columns,
    }
}

/// Diagonal of the pairplot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagKind {
    Kde,
    Hist,
}

/// Slider of "Número máximo de características:".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SliderSpec {
    pub min: usize,
    pub max: usize,
    pub default: usize,
}

/// Defaults of the pairplot options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairplotDefaults {
    pub n_features: usize,
    /// `None` when there are 2 features or fewer (Python shows an info instead).
    pub slider: Option<SliderSpec>,
    /// The first `min(4, n)` friendly names.
    pub default_selected: Vec<String>,
}

/// Slider limits and default selection (§2.6).
pub fn pairplot_defaults(ds: &Dataset) -> PairplotDefaults {
    let n = ds.n_features();
    let k = n.min(PAIRPLOT_DEFAULT_FEATURES);
    PairplotDefaults {
        n_features: n,
        slider: (n > 2).then(|| SliderSpec {
            min: 2,
            max: n.min(PAIRPLOT_MAX_FEATURES),
            default: k,
        }),
        default_selected: ds.feature_display[..k].to_vec(),
    }
}

/// Colour of the scatter points.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Hue {
    /// Classification: levels in order of appearance (seaborn `categorical_order`)
    /// and the level of each point; tab10 palette.
    Categorical {
        levels: Vec<String>,
        codes: Vec<u16>,
    },
    /// Regression (D11): the target value of each point.
    Continuous {
        values: Vec<f64>,
        min: f64,
        max: f64,
    },
}

/// KDE of one level of one feature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KdeCurve {
    pub level: String,
    pub n: usize,
    /// `None` when seaborn skips the level (n < 2 or zero variance).
    pub bandwidth: Option<f64>,
    /// Scott's factor, `None` when skipped.
    pub scott_factor: Option<f64>,
    /// First grid point; for a skipped level, the minimum of its values (0 if none).
    pub grid_min: f64,
    /// Last grid point; for a skipped level, the maximum of its values (0 if none).
    pub grid_max: f64,
    /// 200 densities already scaled by `n_level / n_points` (`common_norm`);
    /// empty when skipped.
    pub density: Vec<f64>,
}

/// Histogram counts of one level, on the common edges of its panel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistSeries {
    pub level: String,
    pub counts: Vec<u32>,
}

/// One diagonal panel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DiagPanel {
    Kde {
        curves: Vec<KdeCurve>,
    },
    /// `np.histogram_bin_edges(all values, "auto")`, one series per level.
    Hist {
        edges: Vec<f64>,
        series: Vec<HistSeries>,
    },
}

/// Everything the UI needs to draw the pairplot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairplotData {
    /// Selected friendly names, in the requested order.
    pub features: Vec<String>,
    pub n_points: usize,
    /// Rows of the dataset.
    pub n_total: usize,
    pub sampled: bool,
    /// Positional dataset rows of the points (the sample, in pandas order).
    pub rows: Vec<usize>,
    /// `n_points × k`.
    pub points: Array2<f64>,
    pub hue: Hue,
    /// D8: the first label column (car name in MPG) of each point.
    pub tooltips: Option<Vec<String>>,
    /// One panel per feature.
    pub diagonal: Vec<DiagPanel>,
    /// Style of §2.6 for datasets with more than 2000 rows.
    pub high_quality: bool,
    /// "Matriz de Dispersión de Características" (+ " (Muestra: n/N puntos)").
    pub title: String,
}

/// Rows kept by the pairplot (§2.6): all if `n <= 2000`; otherwise
/// `groupby(target).sample(min(len, 2000 // n_classes), random_state=42)` for
/// classification and `sample(n=2000, random_state=42)` for regression.
pub fn pairplot_rows(ds: &Dataset) -> Vec<usize> {
    let n = ds.n_rows();
    if n <= PAIRPLOT_MAX_POINTS {
        return (0..n).collect();
    }
    match ds.target_labels() {
        Some(labels) => {
            let mut distinct: Vec<&str> = labels.clone();
            distinct.sort_unstable();
            distinct.dedup();
            let per_class = PAIRPLOT_MAX_POINTS / distinct.len().max(1);
            groupby_sample(&labels, per_class, PAIRPLOT_SEED)
        }
        None => sample_rows(n, PAIRPLOT_MAX_POINTS, PAIRPLOT_SEED),
    }
}

/// Validates the selection: 1 to 6 distinct, known friendly names.
fn feature_indices(ds: &Dataset, features: &[String]) -> Result<Vec<usize>, DatasetError> {
    if features.is_empty() || features.len() > PAIRPLOT_MAX_FEATURES {
        return Err(DatasetError::InvalidSelection {
            reason: format!(
                "hay que elegir entre 1 y {PAIRPLOT_MAX_FEATURES} características (se han elegido {})",
                features.len()
            ),
        });
    }
    let mut out = Vec::with_capacity(features.len());
    for name in features {
        let idx =
            ds.feature_index_by_display(name)
                .ok_or_else(|| DatasetError::InvalidSelection {
                    reason: format!("la característica '{name}' no existe"),
                })?;
        if out.contains(&idx) {
            return Err(DatasetError::InvalidSelection {
                reason: format!("la característica '{name}' está repetida"),
            });
        }
        out.push(idx);
    }
    Ok(out)
}

fn kde_curve(level: &str, values: &[f64], weight: f64) -> KdeCurve {
    match gaussian_kde_scott(values, weight) {
        Some(k) => KdeCurve {
            level: level.to_owned(),
            n: k.n,
            bandwidth: Some(k.bandwidth),
            scott_factor: Some(k.scott_factor),
            grid_min: k.grid_min,
            grid_max: k.grid_max,
            density: k.density,
        },
        None => {
            let finite: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
            let lo = finite.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            KdeCurve {
                level: level.to_owned(),
                n: finite.len(),
                bandwidth: None,
                scott_factor: None,
                grid_min: if lo.is_finite() { lo } else { 0.0 },
                grid_max: if hi.is_finite() { hi } else { 0.0 },
                density: Vec::new(),
            }
        }
    }
}

/// Builds the pairplot of the selected features (friendly names), with sampling,
/// hue and the requested diagonal. Classification: one KDE/histogram per class in
/// order of appearance. Regression (D11): continuous hue and one un-hued diagonal
/// (level [`ALL_LEVEL`]).
pub fn pairplot_data(
    ds: &Dataset,
    features: &[String],
    diag: DiagKind,
) -> Result<PairplotData, DatasetError> {
    let idx = feature_indices(ds, features)?;
    let rows = pairplot_rows(ds);
    let n_points = rows.len();
    let n_total = ds.n_rows();
    let sampled = n_points < n_total;

    let points = Array2::from_shape_fn((n_points, idx.len()), |(r, c)| {
        ds.features[[rows[r], idx[c]]]
    });

    // Levels: (name, mask over the points).
    let (hue, levels): (Hue, Vec<(String, Vec<bool>)>) = match ds.task() {
        TaskType::Classification => {
            let labels = ds.target_labels().unwrap_or_default();
            let mut order: Vec<String> = Vec::new();
            let mut codes = Vec::with_capacity(n_points);
            for &r in &rows {
                let l = labels.get(r).copied().unwrap_or("");
                let code = match order.iter().position(|o| o == l) {
                    Some(p) => p,
                    None => {
                        order.push(l.to_owned());
                        order.len() - 1
                    }
                };
                codes.push(u16::try_from(code).unwrap_or(u16::MAX));
            }
            let levels = order
                .iter()
                .enumerate()
                .map(|(k, name)| {
                    (
                        name.clone(),
                        codes.iter().map(|&c| usize::from(c) == k).collect(),
                    )
                })
                .collect();
            (
                Hue::Categorical {
                    levels: order,
                    codes,
                },
                levels,
            )
        }
        TaskType::Regression => {
            let target = ds.target_numeric();
            let values: Vec<f64> = rows.iter().map(|&r| target[r]).collect();
            let min = values
                .iter()
                .copied()
                .filter(|v| !v.is_nan())
                .fold(f64::INFINITY, f64::min);
            let max = values
                .iter()
                .copied()
                .filter(|v| !v.is_nan())
                .fold(f64::NEG_INFINITY, f64::max);
            (
                Hue::Continuous { values, min, max },
                vec![(ALL_LEVEL.to_owned(), vec![true; n_points])],
            )
        }
    };

    let diagonal = (0..idx.len())
        .map(|c| {
            let col = points.column(c).to_vec();
            match diag {
                DiagKind::Kde => DiagPanel::Kde {
                    curves: levels
                        .iter()
                        .map(|(name, mask)| {
                            let vals: Vec<f64> = col
                                .iter()
                                .zip(mask)
                                .filter(|(_, &m)| m)
                                .map(|(&v, _)| v)
                                .collect();
                            let weight = vals.len() as f64 / n_points as f64;
                            kde_curve(name, &vals, weight)
                        })
                        .collect(),
                },
                DiagKind::Hist => {
                    let edges = histogram_bin_edges_auto(&col);
                    let series = levels
                        .iter()
                        .map(|(name, mask)| {
                            let vals: Vec<f64> = col
                                .iter()
                                .zip(mask)
                                .filter(|(_, &m)| m)
                                .map(|(&v, _)| v)
                                .collect();
                            HistSeries {
                                level: name.clone(),
                                counts: histogram_counts(&vals, &edges),
                            }
                        })
                        .collect();
                    DiagPanel::Hist { edges, series }
                }
            }
        })
        .collect();

    let tooltips = ds.label_columns.first().map(|lc| {
        rows.iter()
            .map(|&r| lc.values.get(r).cloned().unwrap_or_default())
            .collect()
    });

    let mut title = "Matriz de Dispersión de Características".to_owned();
    if sampled {
        title.push_str(&format!(" (Muestra: {n_points}/{n_total} puntos)"));
    }
    Ok(PairplotData {
        features: features.to_vec(),
        n_points,
        n_total,
        sampled,
        rows,
        points,
        hue,
        tooltips,
        diagonal,
        high_quality: n_total > PAIRPLOT_MAX_POINTS,
        title,
    })
}

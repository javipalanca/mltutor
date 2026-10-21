//! In-memory exploration frame shared by the fixture datasets and user CSVs: features,
//! label columns and target aligned by position (not by index: see the Penguins bug).

use super::stats;
use crate::ipc::{
    finite, to_u32, Cell, CorrelationMatrix, DiagKind, DiagPanel, HistSeries, Hue, Matrix,
    PairplotData, TablePreview, TargetDistribution,
};

pub const PAIRPLOT_MAX_POINTS: usize = 2000;
/// Single hue level of the regression diagonal (D11), as named in the fixtures.
pub const ALL_LEVEL: &str = "(todos)";
const BALANCE_LIMIT: f64 = 3.0;
const TARGET_BINS: usize = 30;

#[derive(Debug, Clone)]
pub enum Target {
    Classes { codes: Vec<u16>, names: Vec<String> },
    Numeric(Vec<f64>),
}

impl Target {
    pub fn len(&self) -> usize {
        match self {
            Self::Classes { codes, .. } => codes.len(),
            Self::Numeric(values) => values.len(),
        }
    }

    pub fn as_f64(&self) -> Vec<f64> {
        match self {
            Self::Classes { codes, .. } => codes.iter().map(|c| f64::from(*c)).collect(),
            Self::Numeric(values) => values.clone(),
        }
    }

    fn class_name(names: &[String], code: u16) -> String {
        names
            .get(usize::from(code))
            .cloned()
            .unwrap_or_else(|| code.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct Frame {
    pub feature_columns: Vec<String>,
    pub feature_display: Vec<String>,
    /// Column-major.
    pub features: Vec<Vec<f64>>,
    /// Bool columns: shown as True/False and left out of `describe`.
    pub bool_features: Vec<bool>,
    /// Columns that are not features (D8), e.g. MPG `name`.
    pub labels: Vec<(String, Vec<String>)>,
    pub target: Target,
}

impl Frame {
    pub fn n_rows(&self) -> usize {
        self.target.len()
    }

    pub fn preview(&self, rows: usize) -> TablePreview {
        let n = rows.min(self.n_rows());
        let mut columns = self.feature_display.clone();
        columns.extend(self.labels.iter().map(|(name, _)| name.clone()));
        columns.push("target".into());
        let rows = (0..n)
            .map(|row| {
                let mut cells: Vec<Cell> = self
                    .features
                    .iter()
                    .zip(&self.bool_features)
                    .map(|(col, is_bool)| {
                        if *is_bool && col[row].is_finite() {
                            Cell::Bool { v: col[row] != 0.0 }
                        } else {
                            Cell::number(col[row])
                        }
                    })
                    .collect();
                cells.extend(self.labels.iter().map(|(_, values)| Cell::Text {
                    v: values[row].clone(),
                }));
                cells.push(match &self.target {
                    Target::Classes { codes, names } => Cell::Text {
                        v: Target::class_name(names, codes[row]),
                    },
                    Target::Numeric(values) => Cell::number(values[row]),
                });
                cells
            })
            .collect();
        TablePreview {
            columns,
            index: (0..n).map(|i| i.to_string()).collect(),
            rows,
            total_rows: to_u32(self.n_rows()),
        }
    }

    pub fn describe(&self) -> crate::ipc::DescribeTable {
        let mut columns: Vec<(String, &[f64])> = Vec::new();
        let mut excluded = Vec::new();
        for ((name, values), is_bool) in self
            .feature_display
            .iter()
            .zip(&self.features)
            .zip(&self.bool_features)
        {
            if *is_bool {
                excluded.push(name.clone());
            } else {
                columns.push((name.clone(), values));
            }
        }
        excluded.extend(self.labels.iter().map(|(name, _)| name.clone()));
        match &self.target {
            Target::Numeric(values) => columns.push(("target".into(), values)),
            Target::Classes { .. } => excluded.push("target".into()),
        }
        stats::describe(&columns, excluded)
    }

    pub fn target_distribution(&self) -> TargetDistribution {
        match &self.target {
            Target::Classes { codes, names } => {
                let mut counts: Vec<(String, u32)> = Vec::new();
                for code in codes {
                    let name = Target::class_name(names, *code);
                    match counts.iter_mut().find(|(label, _)| *label == name) {
                        Some((_, count)) => *count += 1,
                        None => counts.push((name, 1)),
                    }
                }
                counts.sort_by(|a, b| a.0.cmp(&b.0));
                let max = counts.iter().map(|c| c.1).max().unwrap_or(0);
                let min = counts.iter().map(|c| c.1).min().unwrap_or(0);
                let ratio = if min > 0 {
                    f64::from(max) / f64::from(min)
                } else {
                    f64::INFINITY
                };
                TargetDistribution::Bar {
                    labels: counts.iter().map(|c| c.0.clone()).collect(),
                    counts: counts.iter().map(|c| c.1).collect(),
                    balance_ratio: finite(ratio),
                    balanced: ratio <= BALANCE_LIMIT,
                }
            }
            Target::Numeric(values) => {
                let (edges, counts) = stats::histogram(values, TARGET_BINS);
                let values_ok: Vec<f64> =
                    values.iter().copied().filter(|v| v.is_finite()).collect();
                TargetDistribution::Histogram {
                    edges,
                    counts,
                    mean: finite(stats::mean(&values_ok)),
                    std: finite(stats::std(&values_ok)),
                    min: finite(values_ok.iter().copied().fold(f64::INFINITY, f64::min)),
                    max: finite(values_ok.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
                }
            }
        }
    }

    pub fn correlation(&self) -> CorrelationMatrix {
        let mut labels = self.feature_columns.clone();
        labels.push("target".into());
        let target = self.target.as_f64();
        let mut columns: Vec<&[f64]> = self.features.iter().map(Vec::as_slice).collect();
        columns.push(&target);
        let n = columns.len();
        let mut values = vec![f64::NAN; n * n];
        let (mut lo, mut hi) = (None::<f64>, None::<f64>);
        for i in 0..n {
            for j in 0..=i {
                let r = stats::pearson(columns[i], columns[j]);
                values[i * n + j] = r;
                values[j * n + i] = r;
                if j < i && r.is_finite() {
                    lo = Some(lo.map_or(r, |m| m.min(r)));
                    hi = Some(hi.map_or(r, |m| m.max(r)));
                }
            }
        }
        CorrelationMatrix {
            labels,
            values: Matrix::from_f64(n, n, values),
            visible_min: lo,
            visible_max: hi,
            truncated: false,
            total_columns: to_u32(n),
            truncation_message: None,
        }
    }

    /// Rows used by the pairplot when no exact sample is supplied: every row up to 2000,
    /// otherwise evenly spaced rows (per class in classification). Python uses
    /// `DataFrame.sample`; the built-in datasets pass the exact rows from the fixture.
    pub fn default_sample(&self) -> Option<Vec<usize>> {
        let n = self.n_rows();
        if n <= PAIRPLOT_MAX_POINTS {
            return None;
        }
        let spaced = |rows: &[usize], k: usize| -> Vec<usize> {
            (0..k.min(rows.len()))
                .map(|i| rows[i * rows.len() / k.min(rows.len())])
                .collect()
        };
        Some(match &self.target {
            Target::Numeric(_) => spaced(&(0..n).collect::<Vec<_>>(), PAIRPLOT_MAX_POINTS),
            Target::Classes { codes, .. } => {
                let mut levels: Vec<u16> = codes.clone();
                levels.sort_unstable();
                levels.dedup();
                let per_class = PAIRPLOT_MAX_POINTS / levels.len().max(1);
                levels
                    .iter()
                    .flat_map(|level| {
                        let rows: Vec<usize> = (0..n).filter(|r| codes[*r] == *level).collect();
                        spaced(&rows, per_class)
                    })
                    .collect()
            }
        })
    }

    /// Builds the pairplot of `features` (friendly names, already validated).
    pub fn pairplot(
        &self,
        features: &[String],
        diag: DiagKind,
        sample: Option<Vec<usize>>,
    ) -> PairplotData {
        let n_total = self.n_rows();
        let sampled = sample.is_some();
        let rows: Vec<usize> = sample.unwrap_or_else(|| (0..n_total).collect());
        let n_points = rows.len();
        let columns: Vec<Vec<f64>> = features
            .iter()
            .filter_map(|f| self.feature_display.iter().position(|d| d == f))
            .map(|c| rows.iter().map(|r| self.features[c][*r]).collect())
            .collect();
        let points = Matrix::from_f64(
            n_points,
            columns.len(),
            (0..n_points)
                .flat_map(|r| columns.iter().map(move |col| col[r]))
                .collect::<Vec<_>>(),
        );

        // Hue levels in order of appearance (seaborn `categorical_order`).
        let (hue, level_of_point, levels): (Hue, Vec<usize>, Vec<String>) = match &self.target {
            Target::Classes { codes, names } => {
                let mut level_codes: Vec<u16> = Vec::new();
                let point_levels: Vec<usize> = rows
                    .iter()
                    .map(|r| {
                        let code = codes[*r];
                        match level_codes.iter().position(|c| *c == code) {
                            Some(index) => index,
                            None => {
                                level_codes.push(code);
                                level_codes.len() - 1
                            }
                        }
                    })
                    .collect();
                let levels: Vec<String> = level_codes
                    .iter()
                    .map(|c| Target::class_name(names, *c))
                    .collect();
                let hue = Hue::Categorical {
                    levels: levels.clone(),
                    codes: point_levels
                        .iter()
                        .map(|l| u16::try_from(*l).unwrap_or(u16::MAX))
                        .collect(),
                };
                (hue, point_levels, levels)
            }
            Target::Numeric(values) => {
                let picked: Vec<f64> = rows.iter().map(|r| values[*r]).collect();
                let values_ok = picked.iter().copied().filter(|v| v.is_finite());
                let min = values_ok.clone().fold(f64::INFINITY, f64::min);
                let max = values_ok.fold(f64::NEG_INFINITY, f64::max);
                (
                    Hue::Continuous {
                        values: picked.into_iter().map(finite).collect(),
                        min: finite(min),
                        max: finite(max),
                    },
                    vec![0; n_points],
                    vec![ALL_LEVEL.to_owned()],
                )
            }
        };

        let diagonal = columns
            .iter()
            .map(|col| {
                let per_level: Vec<Vec<f64>> = (0..levels.len())
                    .map(|level| {
                        col.iter()
                            .zip(&level_of_point)
                            .filter(|(_, l)| **l == level)
                            .map(|(v, _)| *v)
                            .collect()
                    })
                    .collect();
                match diag {
                    DiagKind::Kde => DiagPanel::Kde {
                        curves: levels
                            .iter()
                            .zip(&per_level)
                            .map(|(level, values)| {
                                stats::kde(level, values, values.len() as f64 / n_points as f64)
                            })
                            .collect(),
                    },
                    DiagKind::Hist => {
                        let edges = stats::bin_edges_auto(col);
                        DiagPanel::Hist {
                            series: levels
                                .iter()
                                .zip(&per_level)
                                .map(|(level, values)| HistSeries {
                                    level: level.clone(),
                                    counts: stats::count_in_bins(values, &edges),
                                })
                                .collect(),
                            edges,
                        }
                    }
                }
            })
            .collect();

        let tooltips = self
            .labels
            .first()
            .map(|(_, values)| rows.iter().map(|r| values[*r].clone()).collect());
        let mut title = "Matriz de Dispersión de Características".to_owned();
        if sampled {
            title.push_str(&format!(" (Muestra: {n_points}/{n_total} puntos)"));
        }
        PairplotData {
            features: features.to_vec(),
            n_points: to_u32(n_points),
            n_total: to_u32(n_total),
            sampled,
            points,
            hue,
            tooltips,
            diagonal,
            high_quality: n_total > PAIRPLOT_MAX_POINTS,
            title,
        }
    }
}

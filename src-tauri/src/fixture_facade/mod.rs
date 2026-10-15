//! [`CoreFacade`] over the golden fixtures of `tests/parity/` (fase1-shell-datos.md §8.2).
//!
//! Built-in datasets come from `datasets/<slug>.json` + `datasets/data/<slug>.csv`; the
//! preview, `describe`, target distribution and correlation are the exact values of
//! `stats/<slug>.json`, and the pairplot is computed from the CSV (with the fixture's
//! sample rows). User CSVs go through a small reader. It is meant for tests and for
//! development while `mltutor-core` is not ready: it reads the fixtures from disk.

pub mod csv;
pub mod frame;
pub mod model;
pub mod stats;
mod tree;

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::Value;

use self::frame::{Frame, Target};
use crate::core_facade::{
    check_cancelled, CoreFacade, CsvDraftHandle, DatasetHandle, DatasetView, ModelHandle, Progress,
    TrainSpec,
};
use crate::ipc::{
    finite, to_u32, Cell, CommandError, CorrelationMatrix, DatasetEntry, DatasetId, DatasetRef,
    DatasetSummary, DescribeTable, DiagKind, Matrix, PairplotData, TablePreview,
    TargetDistribution, TaskType,
};

/// Metadata of the selector, embedded from `content/datasets/<slug>.toml`.
const METADATA: [(DatasetId, &str); 12] = [
    (
        DatasetId::Iris,
        include_str!("../../../content/datasets/iris.toml"),
    ),
    (
        DatasetId::Wine,
        include_str!("../../../content/datasets/wine.toml"),
    ),
    (
        DatasetId::BreastCancer,
        include_str!("../../../content/datasets/breast_cancer.toml"),
    ),
    (
        DatasetId::Digits,
        include_str!("../../../content/datasets/digits.toml"),
    ),
    (
        DatasetId::Diabetes,
        include_str!("../../../content/datasets/diabetes.toml"),
    ),
    (
        DatasetId::Diamonds,
        include_str!("../../../content/datasets/diamonds.toml"),
    ),
    (
        DatasetId::Mpg,
        include_str!("../../../content/datasets/mpg.toml"),
    ),
    (
        DatasetId::Moons,
        include_str!("../../../content/datasets/moons.toml"),
    ),
    (
        DatasetId::Titanic,
        include_str!("../../../content/datasets/titanic.toml"),
    ),
    (
        DatasetId::Tips,
        include_str!("../../../content/datasets/tips.toml"),
    ),
    (
        DatasetId::CaliforniaHousing,
        include_str!("../../../content/datasets/california_housing.toml"),
    ),
    (
        DatasetId::Penguins,
        include_str!("../../../content/datasets/penguins.toml"),
    ),
];

#[derive(Debug, Deserialize)]
struct SelectorMetadata {
    label: String,
    task: String,
    #[serde(default)]
    selector_blurb: String,
    #[serde(default)]
    generator: Option<toml::Value>,
}

/// Selector entries of the built-in datasets (shared with `RealFacade`).
pub fn builtin_catalog() -> Vec<DatasetEntry> {
    METADATA
        .iter()
        .filter_map(
            |(id, text)| match toml::from_str::<SelectorMetadata>(text) {
                Ok(meta) => Some(DatasetEntry {
                    dataset: DatasetRef::Builtin { id: *id },
                    label: meta.label,
                    blurb: (!meta.selector_blurb.is_empty()).then_some(meta.selector_blurb),
                    task: if meta.task == "regression" {
                        TaskType::Regression
                    } else {
                        TaskType::Classification
                    },
                    synthetic: meta.generator.is_some(),
                }),
                Err(error) => {
                    log::error!("metadatos no válidos de {}: {error}", id.slug());
                    None
                }
            },
        )
        .collect()
}

pub struct FixtureFacade {
    root: PathBuf,
}

impl FixtureFacade {
    /// `root` is the `tests/parity` directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// `tests/parity` of this checkout (development builds and tests).
    pub fn from_checkout() -> Self {
        Self::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/parity"))
    }

    fn read_json(&self, relative: &str) -> Result<Value, CommandError> {
        let path = self.root.join(relative);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| CommandError::internal(format!("{}: {e}", path.display())))?;
        serde_json::from_str(&text)
            .map_err(|e| CommandError::internal(format!("{}: {e}", path.display())))
    }
}

impl CoreFacade for FixtureFacade {
    fn name(&self) -> &'static str {
        "FixtureFacade"
    }

    fn builtin_catalog(&self) -> Result<Vec<DatasetEntry>, CommandError> {
        Ok(builtin_catalog())
    }

    fn load_builtin(&self, id: DatasetId) -> Result<DatasetHandle, CommandError> {
        let slug = id.slug();
        let meta = self.read_json(&format!("datasets/{slug}.json"))?;
        let stats = self.read_json(&format!("stats/{slug}.json"))?;
        let csv_path = self.root.join(format!("datasets/data/{slug}.csv"));
        let csv_text = std::fs::read_to_string(&csv_path)
            .map_err(|e| CommandError::internal(format!("{}: {e}", csv_path.display())))?;
        let label = builtin_catalog()
            .into_iter()
            .find(|e| e.dataset == DatasetRef::Builtin { id })
            .map(|e| e.label)
            .unwrap_or_else(|| slug.to_owned());
        let mut dataset = FixtureDataset::parse(id, label, &meta["expected"], stats, &csv_text)
            .map_err(|e| CommandError::internal(format!("fixture {slug}: {e}")))?;
        dataset.root.clone_from(&self.root);
        Ok(Arc::new(dataset))
    }

    fn read_csv(
        &self,
        file_name: &str,
        bytes: &[u8],
        cancel: &AtomicBool,
    ) -> Result<CsvDraftHandle, CommandError> {
        check_cancelled(cancel)?;
        Ok(Arc::new(csv::read(file_name, bytes)?))
    }
}

pub(crate) fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .map(|a| {
            a.iter()
                .map(|v| match v {
                    Value::String(text) => text.clone(),
                    other => other.to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

pub(crate) fn floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect())
        .unwrap_or_default()
}

pub(crate) fn counts(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .map(|a| {
            a.iter()
                .map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok()).unwrap_or(0))
                .collect()
        })
        .unwrap_or_default()
}

pub struct FixtureDataset {
    id: DatasetId,
    root: PathBuf,
    frame: Frame,
    summary: DatasetSummary,
    stats: Value,
}

impl FixtureDataset {
    fn parse(
        id: DatasetId,
        label: String,
        meta: &Value,
        stats: Value,
        csv_text: &str,
    ) -> Result<Self, String> {
        let columns = strings(&meta["columns"]);
        let label_columns = strings(&meta["label_columns"]);
        let target_name = meta["target_name"]
            .as_str()
            .ok_or("falta target_name")?
            .to_owned();
        let class_names = meta["class_names"]
            .as_array()
            .map(|_| strings(&meta["class_names"]));
        let dtypes = &meta["dtypes"];

        let mut reader = ::csv::Reader::from_reader(csv_text.as_bytes());
        let header: Vec<String> = reader
            .headers()
            .map_err(|e| e.to_string())?
            .iter()
            .map(str::to_owned)
            .collect();
        let position = |name: &str| {
            header
                .iter()
                .position(|h| h == name)
                .ok_or_else(|| format!("columna {name} ausente del CSV"))
        };
        let feature_pos: Vec<usize> = columns
            .iter()
            .map(|c| position(c))
            .collect::<Result<_, _>>()?;
        let label_pos: Vec<usize> = label_columns
            .iter()
            .map(|c| position(c))
            .collect::<Result<_, _>>()?;
        let target_pos = position(&target_name)?;

        let mut features = vec![Vec::new(); columns.len()];
        let mut labels = vec![Vec::new(); label_columns.len()];
        let mut target = Vec::new();
        let number = |text: &str| match text {
            "True" => 1.0,
            "False" => 0.0,
            other => other.parse::<f64>().unwrap_or(f64::NAN),
        };
        for record in reader.records() {
            let record = record.map_err(|e| e.to_string())?;
            for (out, pos) in features.iter_mut().zip(&feature_pos) {
                out.push(number(record.get(*pos).unwrap_or("")));
            }
            for (out, pos) in labels.iter_mut().zip(&label_pos) {
                out.push(record.get(*pos).unwrap_or("").to_owned());
            }
            target.push(number(record.get(target_pos).unwrap_or("")));
        }
        let target = match &class_names {
            Some(names) => Target::Classes {
                codes: target
                    .iter()
                    .map(|v| u16::try_from(*v as i64).unwrap_or(u16::MAX))
                    .collect(),
                names: names.clone(),
            },
            None => Target::Numeric(target),
        };
        let feature_display = strings(&meta["feature_names"]);
        let task = if class_names.is_some() {
            TaskType::Classification
        } else {
            TaskType::Regression
        };
        let n_rows = target.len();
        let frame = Frame {
            bool_features: columns.iter().map(|c| dtypes[c] == "bool").collect(),
            feature_columns: columns.clone(),
            feature_display: feature_display.clone(),
            features,
            labels: label_columns.iter().cloned().zip(labels).collect(),
            target,
        };
        let summary = DatasetSummary {
            dataset: DatasetRef::Builtin { id },
            label,
            task,
            n_rows: to_u32(n_rows),
            n_features: to_u32(columns.len()),
            target_column: "target".into(),
            feature_columns: columns,
            feature_display,
            label_columns,
            class_names,
            description: meta["dataset_info"]["description"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
        };
        Ok(Self {
            id,
            root: PathBuf::new(),
            frame,
            summary,
            stats,
        })
    }

    fn expected(&self) -> &Value {
        &self.stats["expected"]
    }
}

impl DatasetView for FixtureDataset {
    fn summary(&self) -> DatasetSummary {
        self.summary.clone()
    }

    fn preview(&self, rows: usize) -> TablePreview {
        let expected = self.expected();
        let fixture_rows = expected["preview_rows"].as_array();
        let Some(fixture_rows) = fixture_rows.filter(|r| r.len() >= rows) else {
            return self.frame.preview(rows);
        };
        let columns = strings(&expected["display_columns"]);
        let rows: Vec<Vec<Cell>> = fixture_rows[..rows]
            .iter()
            .map(|row| {
                row.as_array()
                    .map(|cells| {
                        cells
                            .iter()
                            .map(|cell| match cell {
                                Value::Bool(v) => Cell::Bool { v: *v },
                                Value::String(v) => Cell::Text { v: v.clone() },
                                Value::Number(n) => Cell::number(n.as_f64().unwrap_or(f64::NAN)),
                                _ => Cell::Missing,
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect();
        TablePreview {
            columns,
            index: (0..rows.len()).map(|i| i.to_string()).collect(),
            rows,
            total_rows: self.summary.n_rows,
        }
    }

    fn describe(&self) -> DescribeTable {
        let d = &self.expected()["describe"];
        let columns = strings(&d["columns"]);
        let stats = strings(&d["index"]);
        DescribeTable {
            values: Matrix::from_f64(stats.len(), columns.len(), floats(&d["values"])),
            columns,
            stats,
            excluded_columns: strings(&self.expected()["describe_excluded_columns"]),
        }
    }

    fn target_distribution(&self) -> TargetDistribution {
        let t = &self.expected()["target_distribution"];
        if t["kind"] == "bar" {
            TargetDistribution::Bar {
                labels: strings(&t["labels"]),
                counts: counts(&t["counts"]),
                balance_ratio: t["balance_ratio"].as_f64().and_then(finite),
                balanced: t["balanced"].as_bool().unwrap_or(true),
            }
        } else {
            TargetDistribution::Histogram {
                edges: floats(&t["edges"]),
                counts: counts(&t["counts"]),
                mean: t["mean"].as_f64().and_then(finite),
                std: t["std_ddof1"].as_f64().and_then(finite),
                min: t["min"].as_f64().and_then(finite),
                max: t["max"].as_f64().and_then(finite),
            }
        }
    }

    fn correlation(&self, cancel: &AtomicBool) -> Result<CorrelationMatrix, CommandError> {
        check_cancelled(cancel)?;
        let c = &self.expected()["correlation"];
        let labels = strings(&c["columns"]);
        let n = labels.len();
        Ok(CorrelationMatrix {
            values: Matrix::from_f64(n, n, floats(&c["values"])),
            labels,
            visible_min: c["visible_min"].as_f64(),
            visible_max: c["visible_max"].as_f64(),
            truncated: false,
            total_columns: to_u32(n),
            truncation_message: None,
        })
    }

    fn pairplot(&self, features: &[String], diag: DiagKind) -> Result<PairplotData, CommandError> {
        let rows = self.expected()["pairplot"]["sample_rows"]
            .as_array()
            .map(|rows| {
                rows.iter()
                    .filter_map(|r| r.as_u64().and_then(|r| usize::try_from(r).ok()))
                    .filter(|r| *r < self.frame.n_rows())
                    .collect::<Vec<_>>()
            });
        let sample = rows.or_else(|| self.frame.default_sample());
        Ok(self.frame.pairplot(features, diag, sample))
    }

    fn train(
        &self,
        spec: &TrainSpec,
        progress: &Progress<'_>,
        cancel: &AtomicBool,
    ) -> Result<ModelHandle, CommandError> {
        model::train(
            &self.root,
            self.id,
            &self.frame,
            self.summary.task,
            spec,
            progress,
            cancel,
        )
    }
}

#[cfg(test)]
mod tests;

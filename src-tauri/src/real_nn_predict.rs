//! F5d of the real engine: 🔮 Predicciones (C29: 🎲 Exploración and 📊 Predicción por
//! lotes, D81) and 💾 Exportar (C28, D75) of a trained network, over
//! `mltutor_core::nn_predict`, `mltutor-nn` and `mltutor-export`.
//!
//! The checks of the batch CSV follow `[lotes]` of `content/redes/mensajes.toml`:
//! sin_filas → duplicadas → faltan → no_numericas → vacias; with any ❌ nothing is
//! predicted. The columns are matched by NAME (Python's `new_df[feature_cols]`).

use std::path::Path;
use std::sync::atomic::AtomicBool;

use mltutor_core as core;
use mltutor_core::nn_predict::{BatchPredictions, ExploreCurve, OutputKind, RowPrediction};

use crate::core_facade::BatchOutcome;
use crate::ipc::{
    BatchPredictView, BatchTemplate, Cell, ClassProbability, CodeTemplate, CommandError,
    ExploreKind, ExploreView, ExportFormat, Influence, ModelConfig, PredictionResult, TablePreview,
};
use crate::nn::nn_text;
use crate::nn_predict::{csv_field, number};
use crate::real_models::model_error;
use crate::real_nn::RealNn;

/// Rows of the results table on screen (the CSV has all of them).
pub const RESULT_ROWS: usize = 100;
/// Rows of `check` in `modelo.json` (the first test rows).
const CHECK_ROWS: usize = 5;

fn kind(k: OutputKind) -> ExploreKind {
    match k {
        OutputKind::Binary => ExploreKind::Binary,
        OutputKind::Multiclass => ExploreKind::Multiclass,
        OutputKind::Regression => ExploreKind::Regression,
    }
}

fn u32_of(v: usize) -> u32 {
    crate::ipc::to_u32(v)
}

impl RealNn {
    fn class_name(&self, code: usize) -> String {
        self.core
            .dataset
            .class_names()
            .and_then(|names| names.get(code).cloned())
            .unwrap_or_else(|| format!("Clase {code}"))
    }

    fn row_prediction(&self, p: &RowPrediction) -> PredictionResult {
        let classes = self.trained.network.class_codes();
        match p {
            RowPrediction::Binary {
                class, probability, ..
            } => PredictionResult::Class {
                index: u32_of(*class),
                name: self.class_name(*class),
                proba: Some(
                    classes
                        .iter()
                        .zip([1.0 - probability, *probability])
                        .map(|(&code, probability)| ClassProbability {
                            name: self.class_name(code),
                            probability,
                        })
                        .collect(),
                ),
                path: None,
                neighbors: None,
            },
            RowPrediction::Multiclass {
                class,
                probabilities,
                ..
            } => PredictionResult::Class {
                index: u32_of(*class),
                name: self.class_name(*class),
                proba: Some(
                    classes
                        .iter()
                        .zip(probabilities)
                        .map(|(&code, &probability)| ClassProbability {
                            name: self.class_name(code),
                            probability,
                        })
                        .collect(),
                ),
                path: None,
                neighbors: None,
            },
            RowPrediction::Regression { value } => PredictionResult::Value {
                value: *value,
                coefficients: None,
                path: None,
                neighbors: None,
            },
        }
    }

    /// C29 🎲: the axis spans the whole dataset (Python's `nn_df`), 50 values.
    pub fn explore_view(&self, row: usize, feature: usize) -> Result<ExploreView, CommandError> {
        let x = self.core.dataset.features.view();
        if row >= x.nrows() {
            return Err(CommandError::invalid_selection(format!(
                "La muestra {row} no existe: el dataset tiene {} filas.",
                x.nrows()
            )));
        }
        let view = self
            .trained
            .network
            .explore(
                x,
                row,
                feature,
                core::nn_predict::EXPLORE_DEFAULT_POINTS,
                &core::model::TrainControl::none(),
            )
            .map_err(model_error)?;
        let (ys, classes, confidence, base_class) = match view.curve {
            ExploreCurve::Binary { probability } => (probability, None, None, None),
            ExploreCurve::Multiclass {
                classes,
                confidence,
                base_class,
                base_class_probability,
            } => (
                base_class_probability,
                Some(classes.into_iter().map(u32_of).collect()),
                Some(confidence),
                Some(u32_of(base_class)),
            ),
            ExploreCurve::Regression { values } => (values, None, None, None),
        };
        Ok(ExploreView {
            kind: kind(view.kind),
            row: u32_of(view.sample_index),
            feature: u32_of(view.feature),
            n_rows: u32_of(x.nrows()),
            base_prediction: self.row_prediction(&view.base_prediction),
            base: view.base,
            base_value: view.base_value,
            xs: view.xs,
            ys,
            classes,
            confidence,
            base_class,
            range: view.stats.range,
            volatility: view.stats.std,
            relative_impact: view.stats.relative_impact_pct,
            influence: match view.influence {
                core::nn_predict::Influence::High => Influence::High,
                core::nn_predict::Influence::Moderate => Influence::Moderate,
                core::nn_predict::Influence::Low => Influence::Low,
            },
            output_is_probability: view.output_is_probability,
        })
    }

    /// Range (a): minimum and maximum of every feature over the unscaled training rows.
    fn train_ranges(&self) -> Vec<core::nn_predict::FeatureRange> {
        self.trained
            .training_ranges(self.core.dataset.features.view())
            .unwrap_or_default()
    }

    pub fn batch_template_view(&self, rows: usize) -> BatchTemplate {
        let dataset = &self.core.dataset;
        let n = rows.min(dataset.features.nrows());
        let ranges = self.train_ranges();
        let train_min = ranges.iter().map(|r| r.min).collect();
        let train_max = ranges.iter().map(|r| r.max).collect();
        BatchTemplate {
            columns: dataset.feature_columns.clone(),
            sample: TablePreview {
                columns: dataset.feature_columns.clone(),
                index: (0..n).map(|i| i.to_string()).collect(),
                rows: (0..n)
                    .map(|i| {
                        dataset
                            .features
                            .row(i)
                            .iter()
                            .map(|&v| Cell::number(v))
                            .collect()
                    })
                    .collect(),
                total_rows: u32_of(n),
            },
            train_min,
            train_max,
        }
    }

    pub fn predict_batch_view(
        &self,
        path: &Path,
        cancel: &AtomicBool,
    ) -> Result<BatchOutcome, CommandError> {
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let table = match core::csv::read_csv_file_cancellable(path, cancel) {
            Ok(table) => table,
            // A header without rows (or an empty file): `[lotes] sin_filas`, not an error.
            Err(core::CsvError::Empty) => core::csv::RawTable {
                columns: Vec::new(),
                n_rows: 0,
                separator: ',',
                notices: Vec::new(),
            },
            // The CSV reader needs two columns (features + target). A batch file with a
            // single column cannot hold the features of a model with more than one: it is
            // reported as `faltan` with the name of its header.
            Err(core::CsvError::TooFewColumns { .. }) => {
                let header = std::fs::read_to_string(path)
                    .ok()
                    .and_then(|text| text.lines().next().map(str::to_owned))
                    .unwrap_or_default();
                let name = header.trim().trim_matches('"').to_owned();
                if self.core.dataset.feature_columns.len() == 1 {
                    return Err(CommandError::invalid_input(
                        "Añade una columna más al CSV (por ejemplo, un identificador): el lector de CSV necesita al menos dos columnas.",
                    ));
                }
                core::csv::RawTable {
                    columns: vec![core::csv::RawColumn {
                        name,
                        data: core::csv::ColumnData::Float(Vec::new()),
                    }],
                    n_rows: 1,
                    separator: ',',
                    notices: Vec::new(),
                }
            }
            Err(error) => return Err(crate::real_facade::csv_error(error)),
        };
        let checked = check_table(&self.core.dataset, &table);
        let mut view = BatchPredictView {
            file_name,
            rows: u32_of(table.n_rows),
            cols: u32_of(table.columns.len()),
            missing: checked.missing.clone(),
            extra: checked.extra.clone(),
            errors: checked.errors.clone(),
            reordered: checked.reordered,
            kind: if self.core.dataset.class_names().is_none() {
                ExploreKind::Regression
            } else if self.trained.network.mlp().n_outputs() == 1 {
                ExploreKind::Binary
            } else {
                ExploreKind::Multiclass
            },
            output_is_probability: self.trained.network.output_is_probability(),
            out_of_range: None,
            results: None,
            predicted: 0,
        };
        let Some(x) = checked.matrix else {
            return Ok(BatchOutcome { view, csv: None });
        };
        let control = core::model::TrainControl::none().with_cancel(cancel);
        let batch = self
            .trained
            .network
            .predict_batch(x.view(), &control)
            .map_err(model_error)?;
        view.kind = kind(batch.kind);
        view.output_is_probability = batch.output_is_probability;
        view.predicted = u32_of(batch.rows);

        // Range (a): the training rows, original units.
        let columns = &self.core.dataset.feature_columns;
        let flagged = core::nn_predict::out_of_range(x.view(), &self.train_ranges());
        let mut outside = vec![false; x.nrows()];
        let mut outside_columns = vec![false; x.ncols()];
        for (i, cols) in &flagged {
            outside[*i] = true;
            for &j in cols {
                outside_columns[j] = true;
            }
        }
        let n_outside = outside.iter().filter(|&&o| o).count();
        if n_outside > 0 {
            let names: Vec<&str> = columns
                .iter()
                .zip(&outside_columns)
                .filter(|(_, &o)| o)
                .map(|(c, _)| c.as_str())
                .collect();
            view.out_of_range = Some(
                nn_text(
                    "lotes",
                    "fuera_rango",
                    "⚠️ {n} de {total} filas tienen algún valor fuera del rango de los datos de entrenamiento (en {columnas}).",
                )
                .replace("{n}", &n_outside.to_string())
                .replace("{total}", &x.nrows().to_string())
                .replace("{columnas}", &names.join(", ")),
            );
        }

        // Result columns (Python: Predicción, Probabilidad, Confianza / Confianza).
        let t = |key: &str, fallback: &str| nn_text("lotes", key, fallback);
        let yes = t("valor_si", "Sí");
        let no = t("valor_no", "No");
        let mut header: Vec<String> = columns.clone();
        header.push(t("columna_prediccion", "Predicción"));
        let n = batch.rows;
        // Each extra column: its value as a cell and as CSV text, by row.
        let mut extra: Vec<Vec<(Cell, String)>> = Vec::new();
        let f32_text = |v: f64| format!("{}", v as f32);
        match &batch.predictions {
            BatchPredictions::Binary {
                predicted,
                probability,
                confidence,
            } => {
                header.push(t("columna_probabilidad", "Probabilidad"));
                header.push(t("columna_confianza", "Confianza"));
                extra.push(
                    predicted
                        .iter()
                        .map(|&c| {
                            let name = self.class_name(c);
                            (Cell::Text { v: name.clone() }, name)
                        })
                        .collect(),
                );
                for series in [probability, confidence] {
                    extra.push(
                        series
                            .iter()
                            .map(|&v| (Cell::number(v), f32_text(v)))
                            .collect(),
                    );
                }
            }
            BatchPredictions::Multiclass {
                predicted,
                confidence,
                ..
            } => {
                header.push(t("columna_confianza", "Confianza"));
                extra.push(
                    predicted
                        .iter()
                        .map(|&c| {
                            let name = self.class_name(c);
                            (Cell::Text { v: name.clone() }, name)
                        })
                        .collect(),
                );
                extra.push(
                    confidence
                        .iter()
                        .map(|&v| (Cell::number(v), f32_text(v)))
                        .collect(),
                );
            }
            BatchPredictions::Regression { values } => {
                extra.push(
                    values
                        .iter()
                        .map(|&v| (Cell::number(v), f32_text(v)))
                        .collect(),
                );
            }
        }
        header.push(t("columna_fuera_rango", "Fuera de rango"));
        extra.push(
            outside
                .iter()
                .map(|&o| {
                    let text = if o { yes.clone() } else { no.clone() };
                    (Cell::Text { v: text.clone() }, text)
                })
                .collect(),
        );

        let shown = n.min(RESULT_ROWS);
        view.results = Some(TablePreview {
            columns: header.clone(),
            index: (0..shown).map(|i| i.to_string()).collect(),
            rows: (0..shown)
                .map(|i| {
                    x.row(i)
                        .iter()
                        .map(|&v| Cell::number(v))
                        .chain(extra.iter().map(|col| col[i].0.clone()))
                        .collect()
                })
                .collect(),
            total_rows: u32_of(n),
        });
        let mut csv = String::with_capacity(n * (header.len() * 10));
        let fields: Vec<String> = header.iter().map(|h| csv_field(h)).collect();
        csv.push_str(&fields.join(","));
        csv.push('\n');
        for i in 0..n {
            let fields: Vec<String> = x
                .row(i)
                .iter()
                .map(|&v| number(v))
                .chain(extra.iter().map(|col| csv_field(&col[i].1)))
                .collect();
            csv.push_str(&fields.join(","));
            csv.push('\n');
        }
        Ok(BatchOutcome {
            view,
            csv: Some(csv.into_bytes()),
        })
    }

    fn keras(&self) -> Result<mltutor_export::package::keras::KerasPackage, CommandError> {
        let ModelConfig::NeuralNet { arch, training } = &self.core.spec.config else {
            return Err(CommandError::not_nn());
        };
        let dataset = &self.core.dataset;
        let test = &self.trained.split.test;
        let check: Vec<usize> = test.iter().copied().take(CHECK_ROWS).collect();
        let x_check = dataset.features.select(ndarray::Axis(0), &check);
        let final_lr = self
            .trained
            .history
            .epochs
            .last()
            .and_then(|e| e.learning_rate)
            .map_or(training.learning_rate, |lr| lr as f32);
        let core_arch = crate::nn::core_arch(arch);
        let input = mltutor_export::package::keras::KerasPackageInput {
            network: &self.trained.network,
            arch: &core_arch,
            dataset_label: &self.core.spec.dataset_label,
            feature_names: &dataset.feature_display,
            class_names: dataset.class_names(),
            final_learning_rate: final_lr,
            x_check: x_check.view(),
            n_train: self.trained.split.train.len(),
            n_test: test.len(),
        };
        mltutor_export::package::keras::keras_package(&input).map_err(CommandError::export)
    }

    pub fn export_bytes(&self, format: ExportFormat) -> Result<Vec<u8>, CommandError> {
        match format {
            ExportFormat::Onnx => mltutor_export::onnx::mlp_model(&self.trained.network)
                .map(|m| m.to_bytes())
                .map_err(CommandError::export),
            ExportFormat::KerasPackage => self.keras()?.to_zip().map_err(CommandError::export),
            // Decision (F5d): the script with the `modelo.json` it reads.
            ExportFormat::NnNumpyPredictor => {
                self.keras()?.numpy_zip().map_err(CommandError::export)
            }
            ExportFormat::NnMetadata => Err(CommandError::invalid_selection(
                "Los metadatos necesitan la fecha del entrenamiento (export_model con trained_at).",
            )),
            _ => Err(CommandError::not_nn()),
        }
    }

    /// 📋 Metadatos (L3): `mltutor_export::metadata::nn_metadata` as text.
    pub fn metadata_text(&self, trained_at: &str) -> Result<String, CommandError> {
        let ModelConfig::NeuralNet { arch, training } = &self.core.spec.config else {
            return Err(CommandError::not_nn());
        };
        let doc = mltutor_export::metadata::nn_metadata(
            &self.trained,
            &self.core.dataset,
            &crate::nn::core_arch(arch),
            &crate::nn::core_training(training),
            trained_at,
        )
        .map_err(CommandError::export)?;
        mltutor_export::metadata::metadata_json_text(&doc).map_err(CommandError::export)
    }

    pub fn code_text(&self, template: CodeTemplate) -> Result<String, CommandError> {
        match template {
            CodeTemplate::NnTraining => {
                let ModelConfig::NeuralNet { arch, training } = &self.core.spec.config else {
                    return Err(CommandError::not_nn());
                };
                let base = crate::real_models::script_base(
                    &self.core.dataset,
                    self.core.id,
                    &self.core.spec.dataset_label,
                );
                let summary = crate::real_facade::summary_of(&self.core.dataset);
                // L1: the seed of the trained model (D70).
                let context = crate::nn::training_context(base, &summary, arch, training)?;
                crate::nn::render_training(&context)
            }
            CodeTemplate::NnNumpyPredictor => Ok(self.keras()?.numpy_script),
            _ => Err(CommandError::not_nn()),
        }
    }
}

/// The outcome of the checks of a batch CSV.
struct Checked {
    missing: Vec<String>,
    extra: Vec<String>,
    errors: Vec<String>,
    reordered: bool,
    /// The features in model order, when every check passed.
    matrix: Option<ndarray::Array2<f64>>,
}

/// A cell of a text column as a number: numbers, and true/false as 1/0 (as when a CSV
/// is loaded in 📊 Datos). `Err` with the text otherwise; `Ok(NaN)` for an empty cell.
fn text_number(value: Option<&str>) -> Result<f64, String> {
    let Some(text) = value else {
        return Ok(f64::NAN);
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(f64::NAN);
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "true" => return Ok(1.0),
        "false" => return Ok(0.0),
        _ => {}
    }
    match trimmed.parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => Err(text.to_owned()),
    }
}

fn check_table(dataset: &core::Dataset, table: &core::csv::RawTable) -> Checked {
    let t = |key: &str, fallback: &str| nn_text("lotes", key, fallback);
    let file_columns: Vec<String> = table.columns.iter().map(|c| c.name.clone()).collect();
    let mut checked = Checked {
        missing: Vec::new(),
        extra: Vec::new(),
        errors: Vec::new(),
        reordered: false,
        matrix: None,
    };
    if table.n_rows == 0 {
        checked.errors.push(t(
            "sin_filas",
            "❌ El archivo no tiene filas de datos, solo la cabecera.",
        ));
        return checked;
    }
    // The reader renames repeated headers as pandas does (`a, a.1`): `x.N` next to `x`
    // is a repeated column.
    let mut duplicated: Vec<String> = Vec::new();
    for (i, name) in file_columns.iter().enumerate() {
        let base = name
            .rsplit_once('.')
            .filter(|(_, n)| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
            .map(|(base, _)| base)
            .filter(|base| file_columns.iter().any(|c| c == base));
        let repeated = file_columns[..i].contains(name).then_some(name.as_str());
        if let Some(base) = repeated.or(base) {
            if !duplicated.iter().any(|d| d == base) {
                duplicated.push(base.to_owned());
            }
        }
    }
    if !duplicated.is_empty() {
        checked.errors.push(
            t("duplicadas", "❌ Hay columnas repetidas: {columnas}.")
                .replace("{columnas}", &duplicated.join(", ")),
        );
        return checked;
    }
    // By name: the `X.columns` of the model (the "Formato esperado" list) or, if that
    // finds fewer, the names of 📊 Datos.
    let by_column = core::nn_predict::batch_columns(&dataset.feature_columns, &file_columns);
    let by_display = core::nn_predict::batch_columns(&dataset.feature_display, &file_columns);
    let columns = if by_display.missing.len() < by_column.missing.len() {
        by_display
    } else {
        by_column
    };
    checked.missing = columns.missing.clone();
    checked.extra = columns.extra.clone();
    if !columns.complete() {
        checked.errors.push(
            t("faltan", "❌ Faltan características: {columnas}")
                .replace("{columnas}", &columns.missing.join(", ")),
        );
        return checked;
    }
    let positions: Vec<usize> = columns.positions.iter().flatten().copied().collect();
    checked.reordered = positions.iter().enumerate().any(|(i, &p)| i != p);
    let n = table.n_rows;
    let mut matrix = ndarray::Array2::<f64>::zeros((n, positions.len()));
    let mut bad: Vec<String> = Vec::new();
    for (j, &p) in positions.iter().enumerate() {
        let column = &table.columns[p];
        let values: Vec<f64> = match column.data.to_f64() {
            Some(values) => values,
            None => {
                let labels = column.data.labels();
                let mut out = Vec::with_capacity(n);
                let mut first_bad = None;
                for (row, label) in labels.iter().enumerate() {
                    match text_number(label.as_deref()) {
                        Ok(v) => out.push(v),
                        Err(text) => {
                            first_bad.get_or_insert((row, text));
                            out.push(f64::NAN);
                        }
                    }
                }
                if let Some((row, text)) = first_bad {
                    bad.push(
                        nn_text(
                            "lotes",
                            "no_numericas_item",
                            "{columna} (fila {fila}: «{valor}»)",
                        )
                        .replace("{columna}", &column.name)
                        .replace("{fila}", &(row + 1).to_string())
                        .replace("{valor}", &text),
                    );
                }
                out
            }
        };
        // ±inf is not a value the network can use either.
        if let Some(row) = values.iter().position(|v| v.is_infinite()) {
            bad.push(
                nn_text(
                    "lotes",
                    "no_numericas_item",
                    "{columna} (fila {fila}: «{valor}»)",
                )
                .replace("{columna}", &column.name)
                .replace("{fila}", &(row + 1).to_string())
                .replace("{valor}", &values[row].to_string()),
            );
        }
        matrix.column_mut(j).assign(&ndarray::aview1(&values));
    }
    if !bad.is_empty() {
        checked.errors.push(
            t(
                "no_numericas",
                "❌ Estas columnas tienen valores que no son números: {detalles}.",
            )
            .replace("{detalles}", &bad.join(", ")),
        );
        return checked;
    }
    let mut empty_columns: Vec<String> = Vec::new();
    let mut empty_rows = 0;
    for (i, row) in matrix.rows().into_iter().enumerate() {
        let mut any = false;
        for (j, v) in row.iter().enumerate() {
            if v.is_nan() {
                any = true;
                let name = &table.columns[positions[j]].name;
                if !empty_columns.contains(name) {
                    empty_columns.push(name.clone());
                }
            }
        }
        let _ = i;
        empty_rows += usize::from(any);
    }
    if !empty_columns.is_empty() {
        checked.errors.push(
            t(
                "vacias",
                "❌ Hay celdas vacías en: {columnas} ({filas} filas).",
            )
            .replace("{columnas}", &empty_columns.join(", "))
            .replace("{filas}", &empty_rows.to_string()),
        );
        return checked;
    }
    checked.matrix = Some(matrix);
    checked
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use super::*;
    use crate::core_facade::{CoreFacade, ModelHandle, TrainSpec};
    use crate::ipc::{DatasetId, NnActivation, NnArchitecture, NnOptimizer, NnTraining};
    use crate::real_facade::RealFacade;

    fn trained(id: DatasetId, output: NnActivation, seed: u32) -> ModelHandle {
        let spec = TrainSpec {
            config: ModelConfig::NeuralNet {
                arch: NnArchitecture {
                    hidden: vec![10, 10],
                    activation: NnActivation::Relu,
                    output_activation: output,
                    dropout: 0.2,
                    batch_size: 64,
                    optimizer: NnOptimizer::Adam,
                },
                training: NnTraining {
                    learning_rate: 0.01,
                    epochs: 20,
                    validation_pct: 20,
                    patience: Some(10),
                    lr_factor: Some(0.5),
                    seed,
                },
            },
            test_size: 0.2,
            dataset_label: "🌸 Iris - Clasificación de flores".into(),
        };
        RealFacade
            .load_builtin(id)
            .unwrap()
            .train_observed(&spec, &|_, _| {}, &|_| {}, &AtomicBool::new(false))
            .unwrap()
    }

    fn write(dir: &tempfile::TempDir, name: &str, text: &str) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, text).unwrap();
        path
    }

    /// C29 🎲 on the three kinds of output: 50 values from the dataset's min to max.
    #[test]
    fn exploration_follows_python() {
        let iris = trained(DatasetId::Iris, NnActivation::Softmax, 42);
        let view = iris.nn_explore(10, 2).unwrap();
        assert_eq!(view.kind, ExploreKind::Multiclass);
        assert_eq!(view.xs.len(), 50);
        assert_eq!((view.xs[0], view.xs[49]), (1.0, 6.9), "petal length range");
        assert_eq!(view.classes.as_ref().map(Vec::len), Some(50));
        assert!(view.base_class.is_some() && view.relative_impact.is_none());
        assert_eq!(view.n_rows, 150);
        assert!(iris.nn_explore(150, 0).is_err());
        let cancer = trained(DatasetId::BreastCancer, NnActivation::Sigmoid, 42);
        let view = cancer.nn_explore(0, 0).unwrap();
        assert_eq!(view.kind, ExploreKind::Binary);
        let PredictionResult::Class { proba: Some(p), .. } = &view.base_prediction else {
            panic!("{view:?}");
        };
        assert_eq!(p.len(), 2);
        assert!((p[0].probability + p[1].probability - 1.0).abs() < 1e-9);
        let diabetes = trained(DatasetId::Diabetes, NnActivation::Linear, 42);
        let view = diabetes.nn_explore(3, 2).unwrap();
        assert_eq!(view.kind, ExploreKind::Regression);
        assert!(view.relative_impact.is_some());
    }

    /// C29 📊: the checks of `[lotes]` in order and the CSV of the results.
    #[test]
    fn batch_csv_checks_and_results() {
        let iris = trained(DatasetId::Iris, NnActivation::Softmax, 42);
        let template = iris.nn_batch_template(3).unwrap();
        assert_eq!(template.columns.len(), 4);
        assert_eq!(template.sample.rows.len(), 3);
        assert!(template.train_min[2] <= 1.1 && template.train_max[2] >= 6.0);
        let dir = tempfile::tempdir().unwrap();
        let cancel = AtomicBool::new(false);
        // Other order, an extra column, a boolean-free numeric file, one row out of range.
        let path = write(
            &dir,
            "lote.csv",
            "petal width (cm),comentario,sepal length (cm),sepal width (cm),petal length (cm)\n\
             0.2,a,5.1,3.5,1.4\n1.8,b,6.3,2.9,5.6\n9.0,c,5.0,3.0,4.0\n",
        );
        let out = iris.nn_predict_batch(&path, &cancel).unwrap();
        let view = out.view;
        assert!(view.errors.is_empty(), "{view:?}");
        assert!(view.reordered);
        assert_eq!(view.extra, vec!["comentario".to_owned()]);
        assert_eq!(view.predicted, 3);
        assert!(view
            .out_of_range
            .as_deref()
            .unwrap()
            .contains("petal width"));
        let results = view.results.unwrap();
        assert_eq!(
            results.columns.len(),
            4 + 3,
            "features + Predicción + Confianza + Fuera de rango"
        );
        let csv = String::from_utf8(out.csv.unwrap()).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 4);
        assert!(lines[0].starts_with("sepal length (cm),sepal width (cm)"));
        assert!(lines[1].starts_with("5.1,3.5,1.4,0.2,setosa,"), "{csv}");
        assert!(lines[3].ends_with(",Sí"), "{csv}");
        // The checks, in order.
        let cases = [
            ("vacio.csv", "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)\n", "cabecera"),
            ("dup.csv", "a,a\n1,2\n", "repetidas"),
            ("faltan.csv", "sepal length (cm)\n1\n", "Faltan"),
            (
                "texto.csv",
                "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)\n1,2,3,4\n1,dos,3,4\n",
                "dos",
            ),
            (
                "vacias.csv",
                "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)\n1,,3,4\n",
                "vacías",
            ),
        ];
        for (name, text, needle) in cases {
            let path = write(&dir, name, text);
            let out = iris.nn_predict_batch(&path, &cancel).unwrap();
            assert!(out.csv.is_none() && out.view.results.is_none(), "{name}");
            assert_eq!(out.view.errors.len(), 1, "{name}: {:?}", out.view.errors);
            assert!(
                out.view.errors[0].contains(needle),
                "{name}: {:?}",
                out.view.errors
            );
        }
        // Booleans count as 1/0 (as in 📊 Datos).
        let path = write(
            &dir,
            "bool.csv",
            "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)\n5,3,true,false\n5,3,1.5,0.2\n",
        );
        assert!(iris
            .nn_predict_batch(&path, &cancel)
            .unwrap()
            .view
            .errors
            .is_empty());
    }

    /// C28 / D75 / L1: the four formats and the scripts of 💾.
    #[test]
    fn network_exports_and_scripts() {
        let iris = trained(DatasetId::Iris, NnActivation::Softmax, 7);
        let onnx = iris.export(ExportFormat::Onnx).unwrap();
        assert!(mltutor_export::onnx::Model::from_bytes(&onnx).is_ok());
        let zip = iris.export(ExportFormat::KerasPackage).unwrap();
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(zip)).unwrap();
        let mut json = String::new();
        archive
            .by_name("modelo.json")
            .unwrap()
            .read_to_string(&mut json)
            .unwrap();
        assert!(json.contains("mltutor-red-neuronal/1"));
        for file in ["reconstruir_modelo.py", "predecir_numpy.py", "LEEME.txt"] {
            assert!(archive.by_name(file).is_ok(), "{file}");
        }
        let zip = iris.export(ExportFormat::NnNumpyPredictor).unwrap();
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(zip)).unwrap();
        let mut numpy = String::new();
        archive
            .by_name("predecir_numpy.py")
            .unwrap()
            .read_to_string(&mut numpy)
            .unwrap();
        assert!(archive.by_name("modelo.json").is_ok());
        assert_eq!(
            numpy,
            iris.code(CodeTemplate::NnNumpyPredictor, None).unwrap()
        );
        let script = iris.code(CodeTemplate::NnTraining, None).unwrap();
        assert!(
            script.contains("set_random_seed(7)"),
            "L1: the seed of the model"
        );
        assert!(iris.export(ExportFormat::PythonPackage).is_err());
        let json: serde_json::Value =
            serde_json::from_str(&iris.nn_metadata("2026-10-16 10:30").unwrap()).unwrap();
        assert_eq!(json["data_info"]["num_classes"], 3);
        assert!(json.to_string().contains("2026-10-16 10:30"));
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn text_cells_accept_numbers_and_booleans() {
        assert_eq!(text_number(Some(" 2.5 ")), Ok(2.5));
        assert_eq!(text_number(Some("TRUE")), Ok(1.0));
        assert!(text_number(None).unwrap().is_nan());
        assert_eq!(text_number(Some("rojo")), Err("rojo".to_owned()));
        assert!(text_number(Some("inf")).is_err());
    }
}

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
use crate::nn_predict::{csv_field, number, UTF8_BOM};
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
                            code: crate::ipc::to_u32(code),
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
                            code: crate::ipc::to_u32(code),
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
            .unwrap_or_else(|error| {
                // Without ranges no row is flagged as out of range: logged, not hidden.
                log::error!("rangos de entrenamiento de la red: {error}");
                Vec::new()
            })
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

    fn batch_kind(&self) -> ExploreKind {
        if self.core.dataset.class_names().is_none() {
            ExploreKind::Regression
        } else if self.trained.network.mlp().n_outputs() == 1 {
            ExploreKind::Binary
        } else {
            ExploreKind::Multiclass
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
        let table = match read_batch_csv(path, cancel) {
            Ok(table) => table,
            // A header without rows (or an empty file): `[lotes] sin_filas`, not an error.
            Err(core::CsvError::Empty) => core::csv::RawTable {
                columns: Vec::new(),
                n_rows: 0,
                separator: ',',
                notices: Vec::new(),
            },
            Err(error) => return Err(crate::real_facade::csv_error(error)),
        };
        let mut view = BatchPredictView {
            file_name,
            rows: u32_of(table.n_rows),
            cols: u32_of(table.columns.len()),
            missing: Vec::new(),
            extra: Vec::new(),
            errors: Vec::new(),
            reordered: false,
            kind: self.batch_kind(),
            output_is_probability: self.trained.network.output_is_probability(),
            out_of_range: None,
            results: None,
            predicted: 0,
            non_finite: 0,
        };
        // Checked before anything else: nothing of a too big file is predicted.
        if table.n_rows > core::nn_predict::BATCH_MAX_ROWS {
            view.errors.push(
                nn_text(
                    "lotes",
                    "demasiadas_filas",
                    "❌ El archivo tiene demasiadas filas: el máximo para predecir por lotes es {max}.",
                )
                .replace("{max}", &core::nn_predict::BATCH_MAX_ROWS.to_string()),
            );
            return Ok(BatchOutcome { view, csv: None });
        }
        let checked = check_table(&self.core.dataset, &table);
        view.missing = checked.missing;
        view.extra = checked.extra;
        view.errors = checked.errors;
        view.reordered = checked.reordered;
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
        let n = batch.rows;
        let n_valid = batch.finite.iter().filter(|&&f| f).count();
        view.predicted = u32_of(n_valid);
        view.non_finite = u32_of(n - n_valid);

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

        let outputs = BatchOutputs::new(self, &batch, &outside);
        let mut header: Vec<String> = columns.clone();
        header.extend(outputs.header());

        // Only the rows on screen become cells.
        let shown = n.min(RESULT_ROWS);
        view.results = Some(TablePreview {
            columns: header.clone(),
            index: (0..shown).map(|i| i.to_string()).collect(),
            rows: (0..shown)
                .map(|i| {
                    x.row(i)
                        .iter()
                        .map(|&v| Cell::number(v))
                        .chain(outputs.row(i).into_iter().map(|(cell, _)| cell))
                        .collect()
                })
                .collect(),
            total_rows: u32_of(n),
        });
        // The CSV, row by row (BOM for Excel; text fields neutralised, see csv_field).
        let mut csv: Vec<u8> = Vec::with_capacity(UTF8_BOM.len() + n * header.len() * 10);
        csv.extend_from_slice(UTF8_BOM);
        let mut line = String::new();
        let mut push_line = |fields: &mut dyn Iterator<Item = String>, csv: &mut Vec<u8>| {
            line.clear();
            for (j, field) in fields.enumerate() {
                if j > 0 {
                    line.push(',');
                }
                line.push_str(&field);
            }
            line.push('\n');
            csv.extend_from_slice(line.as_bytes());
        };
        push_line(&mut header.iter().map(|h| csv_field(h)), &mut csv);
        for i in 0..n {
            if i % core::csv::CANCEL_CHECK_ROWS == 0 {
                crate::core_facade::check_cancelled(cancel)?;
            }
            let mut fields = x
                .row(i)
                .iter()
                .map(|&v| number(v))
                .chain(outputs.row(i).into_iter().map(|(_, text)| csv_field(&text)))
                .collect::<Vec<_>>()
                .into_iter();
            push_line(&mut fields, &mut csv);
        }
        Ok(BatchOutcome {
            view,
            csv: Some(csv),
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

/// The batch CSV: one column is enough (a model of one feature, M2); the size is
/// checked before reading, as `read_csv_file_cancellable`.
fn read_batch_csv(path: &Path, cancel: &AtomicBool) -> Result<core::csv::RawTable, core::CsvError> {
    let io = |e: std::io::Error| core::CsvError::Io {
        message: e.to_string(),
    };
    let len = std::fs::metadata(path).map_err(io)?.len();
    if len > core::csv::MAX_CSV_BYTES {
        return Err(core::CsvError::TooLarge {
            bytes: len,
            limit: core::csv::MAX_CSV_BYTES,
        });
    }
    let bytes = std::fs::read(path).map_err(io)?;
    core::csv::read_csv_cancellable_with(&bytes, cancel, core::csv::ReadOptions { min_columns: 1 })
}

/// The result columns after the features, built row by row (only the rows on screen
/// become cells; the CSV is written as it goes).
struct BatchOutputs<'a> {
    nn: &'a RealNn,
    batch: &'a core::nn_predict::BatchView,
    outside: &'a [bool],
    yes: String,
    no: String,
    no_prediction: String,
}

impl<'a> BatchOutputs<'a> {
    fn new(nn: &'a RealNn, batch: &'a core::nn_predict::BatchView, outside: &'a [bool]) -> Self {
        let t = |key: &str, fallback: &str| nn_text("lotes", key, fallback);
        Self {
            nn,
            batch,
            outside,
            yes: t("valor_si", "Sí"),
            no: t("valor_no", "No"),
            no_prediction: t(
                "fila_no_finita",
                "Sin predicción: la red dio una salida no numérica (NaN o infinito) para esta fila",
            ),
        }
    }

    fn header(&self) -> Vec<String> {
        let t = |key: &str, fallback: &str| nn_text("lotes", key, fallback);
        let probability = self.batch.output_is_probability;
        let mut header = vec![t("columna_prediccion", "Predicción")];
        match &self.batch.predictions {
            BatchPredictions::Binary { .. } => {
                // D87: a tanh output is the neuron's output, not a probability, and its
                // "confidence" max(p, 1 − p) would not be one either (it can exceed 1).
                header.push(if probability {
                    t("columna_probabilidad", "Probabilidad")
                } else {
                    nn_text("exploracion", "salida_neurona", "Salida de la neurona")
                });
                if probability {
                    header.push(t("columna_confianza", "Confianza"));
                }
            }
            BatchPredictions::Multiclass { .. } => {
                if probability {
                    header.push(t("columna_confianza", "Confianza"));
                }
            }
            BatchPredictions::Regression { .. } => {}
        }
        header.push(t("columna_fuera_rango", "Fuera de rango"));
        header
    }

    /// A number cell and its CSV text (float32, as the network computes; empty if NaN).
    fn value(v: f64) -> (Cell, String) {
        if v.is_finite() {
            (Cell::number(v), format!("{}", v as f32))
        } else {
            (Cell::Missing, String::new())
        }
    }

    fn class(&self, class: Option<usize>) -> (Cell, String) {
        let name = class.map_or_else(|| self.no_prediction.clone(), |c| self.nn.class_name(c));
        (Cell::Text { v: name.clone() }, name)
    }

    fn row(&self, i: usize) -> Vec<(Cell, String)> {
        let probability = self.batch.output_is_probability;
        let finite = self.batch.finite.get(i).copied().unwrap_or(false);
        let at = |series: &[f64]| series.get(i).copied().unwrap_or(f64::NAN);
        let mut out = Vec::with_capacity(4);
        match &self.batch.predictions {
            BatchPredictions::Binary {
                predicted,
                probability: p,
                confidence,
            } => {
                out.push(self.class(predicted.get(i).copied().flatten()));
                out.push(Self::value(at(p)));
                if probability {
                    out.push(Self::value(at(confidence)));
                }
            }
            BatchPredictions::Multiclass {
                predicted,
                confidence,
                ..
            } => {
                out.push(self.class(predicted.get(i).copied().flatten()));
                if probability {
                    out.push(Self::value(at(confidence)));
                }
            }
            BatchPredictions::Regression { values } => {
                if finite {
                    out.push(Self::value(at(values)));
                } else {
                    // A numeric column: the table says why, the CSV leaves it empty.
                    let (cell, _) = self.class(None);
                    out.push((cell, String::new()));
                }
            }
        }
        let o = self.outside.get(i).copied().unwrap_or(false);
        let text = if o { self.yes.clone() } else { self.no.clone() };
        out.push((Cell::Text { v: text.clone() }, text));
        out
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
    // By name: the `X.columns` of the model (the "Formato esperado" list) or, if that
    // finds fewer, the names of 📊 Datos.
    let by_column = core::nn_predict::batch_columns(&dataset.feature_columns, &file_columns);
    let by_display = core::nn_predict::batch_columns(&dataset.feature_display, &file_columns);
    let (columns, features) = if by_display.missing.len() < by_column.missing.len() {
        (by_display, &dataset.feature_display)
    } else {
        (by_column, &dataset.feature_columns)
    };
    // M3: the reader renamed the repeated headers (`a, a` → `a, a.1`). Only a repeated
    // FEATURE is ambiguous; a repeated extra column is ignored like any other.
    let mut duplicated: Vec<String> = Vec::new();
    for notice in &table.notices {
        if let core::csv::CsvNotice::RenamedHeaders { renamed } = notice {
            for (original, _) in renamed {
                if features.contains(original) && !duplicated.contains(original) {
                    duplicated.push(original.clone());
                }
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
    // In the file's order, not just at other indices (an extra column before them
    // shifts the indices without changing the order).
    checked.reordered = positions.windows(2).any(|w| w[0] > w[1]);
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
    // M1: a finite value beyond float32 would become ±∞ in the network.
    if let Err(core::model::ModelError::NonFiniteCell { row, column }) =
        core::nn_predict::check_f32_inputs(matrix.view())
    {
        let name = &table.columns[positions[column]].name;
        checked.errors.push(
            t(
                "no_numericas",
                "❌ Estas columnas tienen valores que no son números: {detalles}.",
            )
            .replace(
                "{detalles}",
                &nn_text(
                    "lotes",
                    "no_numericas_item",
                    "{columna} (fila {fila}: «{valor}»)",
                )
                .replace("{columna}", name)
                .replace("{fila}", &(row + 1).to_string())
                .replace("{valor}", &matrix[[row, column]].to_string()),
            ),
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
        let bytes = out.csv.unwrap();
        assert!(bytes.starts_with(UTF8_BOM), "BOM for Excel");
        let csv = String::from_utf8(bytes[UTF8_BOM.len()..].to_vec()).unwrap();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 4);
        assert!(lines[0].starts_with("sepal length (cm),sepal width (cm)"));
        assert!(lines[1].starts_with("5.1,3.5,1.4,0.2,setosa,"), "{csv}");
        assert!(lines[3].ends_with(",Sí"), "{csv}");
        // The checks, in order.
        let cases = [
            ("vacio.csv", "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)\n", "cabecera"),
            ("dup.csv", "sepal length (cm),sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)\n1,1,2,3,4\n", "repetidas"),
            ("grande.csv", "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)\n1,2,3,1e39\n", "petal width (cm) (fila 1"),
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

    /// M3: a repeated column that is not a feature does not block the batch; one-column
    /// files are read (M2) and only fail as `faltan`.
    #[test]
    fn repeated_extra_columns_and_one_column_files() {
        let iris = trained(DatasetId::Iris, NnActivation::Softmax, 42);
        let dir = tempfile::tempdir().unwrap();
        let cancel = AtomicBool::new(false);
        let path = write(
            &dir,
            "extra.csv",
            "id,sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),id\n1,5.1,3.5,1.4,0.2,1\n",
        );
        let view = iris.nn_predict_batch(&path, &cancel).unwrap().view;
        assert!(view.errors.is_empty(), "{view:?}");
        assert!(!view.reordered, "an extra column first is not a new order");
        assert_eq!(view.extra, vec!["id".to_owned(), "id.1".to_owned()]);
        let path = write(&dir, "una.csv", "sepal length (cm)\n5.1\n");
        let view = iris.nn_predict_batch(&path, &cancel).unwrap().view;
        assert_eq!(view.missing.len(), 3, "{view:?}");
        assert_eq!(view.cols, 1);
    }

    /// M1: a row whose output overflows float32 has no prediction; the others do.
    #[test]
    fn non_finite_rows_have_no_prediction() {
        let mut model = trained(DatasetId::Diabetes, NnActivation::Linear, 42);
        let template = model.nn_batch_template(2).unwrap();
        let header = template.columns.join(",");
        let ok: Vec<String> = template.sample.rows[0]
            .iter()
            .map(|c| match c {
                Cell::Number { v } => v.to_string(),
                _ => String::new(),
            })
            .collect();
        let mut huge = ok.clone();
        huge[0] = "3e38".into();
        let dir = tempfile::tempdir().unwrap();
        let path = write(
            &dir,
            "enorme.csv",
            &format!("{header}\n{}\n{}\n", ok.join(","), huge.join(",")),
        );
        let out = model
            .nn_predict_batch(&path, &AtomicBool::new(false))
            .unwrap();
        assert!(out.view.errors.is_empty(), "{:?}", out.view.errors);
        assert_eq!((out.view.predicted, out.view.non_finite), (1, 1));
        let results = out.view.results.unwrap();
        let p = template.columns.len();
        assert!(matches!(results.rows[0][p], Cell::Number { .. }));
        let Cell::Text { v } = &results.rows[1][p] else {
            panic!("{:?}", results.rows[1]);
        };
        assert!(v.starts_with("Sin predicción"), "{v}");
        let csv = String::from_utf8(out.csv.unwrap()).unwrap();
        let last = csv.lines().last().unwrap();
        let fields: Vec<&str> = last.split(',').collect();
        assert_eq!(fields[p], "", "empty numeric cell in regression: {last}");
        // Classification: the class cell explains it in the CSV too.
        model = trained(DatasetId::Iris, NnActivation::Softmax, 42);
        let path = write(
            &dir,
            "enorme_iris.csv",
            "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm)\n5,3,1.5,0.2\n3e38,3,1.5,0.2\n",
        );
        let out = model
            .nn_predict_batch(&path, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(
            (out.view.predicted, out.view.non_finite),
            (1, 1),
            "{:?}",
            out.view
        );
        let csv = String::from_utf8(out.csv.unwrap()).unwrap();
        assert!(
            csv.lines().last().unwrap().contains("Sin predicción"),
            "{csv}"
        );
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

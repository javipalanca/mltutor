//! [`CoreFacade`] over `mltutor-core` (the implementation the app uses). It only converts
//! core types into the IPC DTOs and core errors into Spanish `CommandError`s.

use std::sync::Arc;

use mltutor_core as core;
use mltutor_core::explore;

use crate::core_facade::{
    CoreFacade, CsvDraft, CsvDraftHandle, CsvFinal, DatasetHandle, DatasetView,
};
use crate::ipc::{
    self, to_u32, CommandError, CorrelationMatrix, CsvPreview, DatasetEntry, DatasetId, DatasetRef,
    DatasetSummary, DescribeTable, DiagKind, Matrix, PairplotData, TablePreview,
    TargetDistribution, TargetInfo, TaskChoice, TaskType,
};

/// Moons is generated once per session with this seed (D9; equals the fixture).
const MOONS_SEED: u32 = 42;

pub struct RealFacade;

// --- conversions core → IPC ---

fn dataset_id(id: DatasetId) -> core::DatasetId {
    match id {
        DatasetId::Iris => core::DatasetId::Iris,
        DatasetId::Wine => core::DatasetId::Wine,
        DatasetId::BreastCancer => core::DatasetId::BreastCancer,
        DatasetId::Digits => core::DatasetId::Digits,
        DatasetId::Diabetes => core::DatasetId::Diabetes,
        DatasetId::Diamonds => core::DatasetId::Diamonds,
        DatasetId::Mpg => core::DatasetId::Mpg,
        DatasetId::Moons => core::DatasetId::Moons,
        DatasetId::Titanic => core::DatasetId::Titanic,
        DatasetId::Tips => core::DatasetId::Tips,
        DatasetId::CaliforniaHousing => core::DatasetId::CaliforniaHousing,
        DatasetId::Penguins => core::DatasetId::Penguins,
    }
}

fn ipc_dataset_id(id: core::DatasetId) -> DatasetId {
    match id {
        core::DatasetId::Iris => DatasetId::Iris,
        core::DatasetId::Wine => DatasetId::Wine,
        core::DatasetId::BreastCancer => DatasetId::BreastCancer,
        core::DatasetId::Digits => DatasetId::Digits,
        core::DatasetId::Diabetes => DatasetId::Diabetes,
        core::DatasetId::Diamonds => DatasetId::Diamonds,
        core::DatasetId::Mpg => DatasetId::Mpg,
        core::DatasetId::Moons => DatasetId::Moons,
        core::DatasetId::Titanic => DatasetId::Titanic,
        core::DatasetId::Tips => DatasetId::Tips,
        core::DatasetId::CaliforniaHousing => DatasetId::CaliforniaHousing,
        core::DatasetId::Penguins => DatasetId::Penguins,
    }
}

fn task(task: core::TaskType) -> TaskType {
    match task {
        core::TaskType::Classification => TaskType::Classification,
        core::TaskType::Regression => TaskType::Regression,
    }
}

fn task_choice(choice: TaskChoice) -> core::TaskChoice {
    match choice {
        TaskChoice::Auto => core::TaskChoice::Auto,
        TaskChoice::Classification => core::TaskChoice::Classification,
        TaskChoice::Regression => core::TaskChoice::Regression,
    }
}

fn matrix(values: &ndarray::Array2<f64>) -> Matrix {
    let (rows, cols) = values.dim();
    Matrix::from_f64(rows, cols, values.iter().copied().collect::<Vec<_>>())
}

fn cell(cell: core::Cell) -> ipc::Cell {
    match cell {
        core::Cell::Number { v } => ipc::Cell::number(v),
        core::Cell::Text { v } => ipc::Cell::Text { v },
        core::Cell::Bool { v } => ipc::Cell::Bool { v },
        core::Cell::Missing => ipc::Cell::Missing,
    }
}

fn table(preview: core::TablePreview) -> TablePreview {
    TablePreview {
        columns: preview.columns,
        index: preview.index,
        rows: preview
            .rows
            .into_iter()
            .map(|row| row.into_iter().map(cell).collect())
            .collect(),
        total_rows: to_u32(preview.total_rows),
    }
}

fn describe(table: core::DescribeTable) -> DescribeTable {
    DescribeTable {
        values: matrix(&table.values),
        columns: table.columns,
        stats: table.stats,
        excluded_columns: table.excluded_columns,
    }
}

fn summary(summary: core::DatasetSummary) -> DatasetSummary {
    DatasetSummary {
        // Filled in by the session (it knows the reference of the dataset).
        dataset: DatasetRef::default(),
        label: summary.label,
        task: task(summary.task),
        n_rows: to_u32(summary.n_rows),
        n_features: to_u32(summary.n_features),
        target_column: summary.target_column,
        feature_columns: summary.feature_columns,
        feature_display: summary.feature_display,
        label_columns: summary.label_columns,
        class_names: summary.class_names,
        description: summary.description,
    }
}

fn target_info(info: core::csv::TargetInfo) -> TargetInfo {
    TargetInfo {
        column: info.column,
        nunique: to_u32(info.nunique),
        dtype: info.dtype,
        value_counts: info.value_counts,
        suggested_task: task(info.suggested_task),
    }
}

fn kde_curve(curve: explore::KdeCurve) -> ipc::KdeCurve {
    ipc::KdeCurve {
        level: curve.level,
        n: to_u32(curve.n),
        bandwidth: curve.bandwidth,
        grid_min: curve.grid_min,
        grid_max: curve.grid_max,
        density: curve.density,
    }
}

fn pairplot(plot: explore::PairplotData) -> PairplotData {
    PairplotData {
        features: plot.features,
        n_points: to_u32(plot.n_points),
        n_total: to_u32(plot.n_total),
        sampled: plot.sampled,
        points: matrix(&plot.points),
        hue: match plot.hue {
            explore::Hue::Categorical { levels, codes } => ipc::Hue::Categorical { levels, codes },
            explore::Hue::Continuous { values, min, max } => {
                ipc::Hue::Continuous { values, min, max }
            }
        },
        tooltips: plot.tooltips,
        diagonal: plot
            .diagonal
            .into_iter()
            .map(|panel| match panel {
                explore::DiagPanel::Kde { curves } => ipc::DiagPanel::Kde {
                    curves: curves.into_iter().map(kde_curve).collect(),
                },
                explore::DiagPanel::Hist { edges, series } => ipc::DiagPanel::Hist {
                    edges,
                    series: series
                        .into_iter()
                        .map(|s| ipc::HistSeries {
                            level: s.level,
                            counts: s.counts,
                        })
                        .collect(),
                },
            })
            .collect(),
        high_quality: plot.high_quality,
        title: plot.title,
    }
}

// --- errors core → IPC (messages of fase1-shell-datos.md §5.1) ---

pub fn csv_error(error: core::CsvError) -> CommandError {
    match error {
        core::CsvError::Empty => CommandError::csv_empty(),
        core::CsvError::TooFewColumns { .. } => CommandError::csv_too_few_columns(),
        core::CsvError::Encoding => CommandError::csv_encoding(),
        core::CsvError::Malformed {
            line,
            found,
            expected,
        } => CommandError::csv_malformed(line, found, expected),
        core::CsvError::TooLarge { bytes, limit } => CommandError::csv_too_large(bytes, limit),
        core::CsvError::Io { message } => CommandError::csv_io(message),
        core::CsvError::UnknownColumn { column } => CommandError::unknown_column(&column),
    }
}

pub fn dataset_error(error: core::DatasetError) -> CommandError {
    match error {
        core::DatasetError::UnknownDataset { name } => CommandError::dataset_not_found(name),
        core::DatasetError::UnknownColumn { column } => CommandError::unknown_column(&column),
        core::DatasetError::InvalidSelection { reason } => {
            CommandError::invalid_selection(format!("Selección no válida: {reason}"))
        }
        core::DatasetError::Csv(error) => csv_error(error),
        other @ (core::DatasetError::Metadata { .. } | core::DatasetError::Data { .. }) => {
            CommandError::internal(other)
        }
    }
}

// --- views ---

struct CoreDataset(core::Dataset);

impl DatasetView for CoreDataset {
    fn summary(&self) -> DatasetSummary {
        summary(self.0.summary())
    }

    fn preview(&self, rows: usize) -> TablePreview {
        table(explore::preview(&self.0, rows))
    }

    fn describe(&self) -> DescribeTable {
        describe(explore::describe(&self.0))
    }

    fn target_distribution(&self) -> TargetDistribution {
        match explore::target_distribution(&self.0) {
            explore::TargetDistribution::Bar {
                labels,
                counts,
                balance_ratio,
                balanced,
            } => TargetDistribution::Bar {
                labels,
                counts,
                balance_ratio,
                balanced,
            },
            explore::TargetDistribution::Histogram {
                edges,
                counts,
                mean,
                std,
                min,
                max,
            } => TargetDistribution::Histogram {
                edges,
                counts,
                mean,
                std,
                min,
                max,
            },
        }
    }

    fn correlation(&self) -> CorrelationMatrix {
        let corr = explore::correlation_matrix(&self.0);
        CorrelationMatrix {
            values: matrix(&corr.values),
            labels: corr.labels,
            visible_min: corr.visible_min,
            visible_max: corr.visible_max,
        }
    }

    fn pairplot(&self, features: &[String], diag: DiagKind) -> Result<PairplotData, CommandError> {
        let diag = match diag {
            DiagKind::Kde => explore::DiagKind::Kde,
            DiagKind::Hist => explore::DiagKind::Hist,
        };
        explore::pairplot_data(&self.0, features, diag)
            .map(pairplot)
            .map_err(dataset_error)
    }
}

struct CoreCsvDraft {
    file_name: String,
    table: core::csv::RawTable,
}

impl CsvDraft for CoreCsvDraft {
    fn preview(&self) -> CsvPreview {
        match self.table.preview() {
            Ok(preview) => CsvPreview {
                draft_id: 0,
                file_name: self.file_name.clone(),
                n_rows: to_u32(preview.n_rows),
                n_cols: to_u32(preview.n_cols),
                missing_values: to_u32(preview.missing_values),
                columns: preview
                    .columns
                    .into_iter()
                    .map(|c| ipc::CsvColumnInfo {
                        name: c.name,
                        dtype: c.dtype,
                        missing: to_u32(c.missing),
                        nunique: to_u32(c.nunique),
                    })
                    .collect(),
                head: table(preview.head),
                default_target: preview.default_target,
                target: target_info(preview.target),
                notices: preview
                    .notices
                    .iter()
                    .map(core::csv::CsvNotice::message)
                    .collect(),
            },
            // `read_csv` already validated the table, so this is not expected.
            Err(error) => {
                log::error!("vista previa del CSV no disponible: {error}");
                CsvPreview {
                    draft_id: 0,
                    file_name: self.file_name.clone(),
                    n_rows: to_u32(self.table.n_rows),
                    n_cols: to_u32(self.table.n_cols()),
                    missing_values: 0,
                    columns: Vec::new(),
                    head: table(self.table.head(0)),
                    default_target: self.table.default_target().to_owned(),
                    target: TargetInfo {
                        column: self.table.default_target().to_owned(),
                        nunique: 0,
                        dtype: "object".into(),
                        value_counts: None,
                        suggested_task: TaskType::Regression,
                    },
                    notices: Vec::new(),
                }
            }
        }
    }

    fn target_info(&self, target: &str) -> Result<TargetInfo, CommandError> {
        self.table
            .target_info(target)
            .map(target_info)
            .map_err(csv_error)
    }

    fn finalize(&self, target: &str, choice: TaskChoice) -> Result<CsvFinal, CommandError> {
        let choice = task_choice(choice);
        let dataset = self
            .table
            .to_dataset(target, choice, &self.file_name)
            .map_err(csv_error)?;
        let stats = match self.table.target_stats(target, choice).map_err(csv_error)? {
            core::csv::CsvTargetStats::Classes { value_counts } => {
                ipc::CsvTargetStats::Classes { value_counts }
            }
            core::csv::CsvTargetStats::Numeric {
                min,
                max,
                mean,
                median,
                std,
                hist_edges,
                hist_counts,
            } => ipc::CsvTargetStats::Numeric {
                min,
                max,
                mean,
                median,
                std,
                hist_edges,
                hist_counts,
            },
        };
        Ok(CsvFinal {
            resolved_task: task(dataset.task()),
            n_rows: self.table.n_rows,
            n_cols: self.table.n_cols(),
            describe: describe(self.table.describe()),
            target_stats: stats,
            dataset: Arc::new(CoreDataset(dataset)),
        })
    }
}

impl CoreFacade for RealFacade {
    fn name(&self) -> &'static str {
        "RealFacade (mltutor-core)"
    }

    fn builtin_catalog(&self) -> Vec<DatasetEntry> {
        match core::dataset::list_builtin() {
            Ok(entries) => entries
                .into_iter()
                .map(|e| DatasetEntry {
                    dataset: DatasetRef::Builtin {
                        id: ipc_dataset_id(e.id),
                    },
                    label: e.label,
                    blurb: e.blurb,
                    task: task(e.task),
                    synthetic: e.synthetic,
                })
                .collect(),
            Err(error) => {
                log::error!("no se pudo leer el catálogo de datasets: {error}");
                Vec::new()
            }
        }
    }

    fn load_builtin(&self, id: DatasetId) -> Result<DatasetHandle, CommandError> {
        let dataset = match id {
            DatasetId::Moons => core::dataset::moons_dataset(MOONS_SEED),
            other => core::dataset::load_builtin(dataset_id(other)),
        }
        .map_err(dataset_error)?;
        Ok(Arc::new(CoreDataset(dataset)))
    }

    fn read_csv(&self, file_name: &str, bytes: &[u8]) -> Result<CsvDraftHandle, CommandError> {
        let table = core::csv::read_csv(bytes).map_err(csv_error)?;
        Ok(Arc::new(CoreCsvDraft {
            file_name: file_name.to_owned(),
            table,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_facade::FixtureFacade;

    /// Both facades must agree on what the UI shows (the fixture one is exact).
    #[test]
    fn real_facade_matches_the_fixtures() {
        let real = RealFacade;
        let fixture = FixtureFacade::from_checkout();
        assert_eq!(real.builtin_catalog(), fixture.builtin_catalog());
        for entry in fixture.builtin_catalog() {
            let DatasetRef::Builtin { id } = entry.dataset else {
                unreachable!()
            };
            let a = real.load_builtin(id).unwrap();
            let b = fixture.load_builtin(id).unwrap();
            let (sa, sb) = (a.summary(), b.summary());
            assert_eq!(
                (
                    sa.n_rows,
                    sa.n_features,
                    &sa.feature_columns,
                    &sa.feature_display
                ),
                (
                    sb.n_rows,
                    sb.n_features,
                    &sb.feature_columns,
                    &sb.feature_display
                ),
                "{id:?}"
            );
            assert_eq!(sa.class_names, sb.class_names, "{id:?}");
            assert_eq!(a.preview(10).columns, b.preview(10).columns, "{id:?}");
            assert_eq!(a.describe().columns, b.describe().columns, "{id:?}");
            assert_eq!(a.correlation().labels, b.correlation().labels, "{id:?}");
            let features: Vec<String> = sa.feature_display.iter().take(3).cloned().collect();
            let pa = a.pairplot(&features, DiagKind::Kde).unwrap();
            let pb = b.pairplot(&features, DiagKind::Kde).unwrap();
            assert_eq!(
                (pa.n_points, pa.sampled),
                (pb.n_points, pb.sampled),
                "{id:?}"
            );
        }
    }

    #[test]
    fn csv_errors_keep_the_python_messages() {
        let real = RealFacade;
        let err = |bytes: &[u8]| real.read_csv("x.csv", bytes).err().unwrap();
        assert_eq!(err(b"").message(), "❌ El archivo está vacío");
        assert!(matches!(
            err(b"a\n1\n"),
            CommandError::CsvTooFewColumns { .. }
        ));
        assert!(matches!(
            err(b"a,b\n1,2\n1,2,3\n"),
            CommandError::CsvMalformed { .. }
        ));
        let draft = real.read_csv("t.csv", b"a,b,c\n1,2,x\n3,4,y\n").unwrap();
        assert!(matches!(
            draft.target_info("zz"),
            Err(CommandError::UnknownColumn { .. })
        ));
        let done = draft.finalize("c", TaskChoice::Classification).unwrap();
        assert_eq!(done.resolved_task, TaskType::Classification);
        assert_eq!(
            done.dataset.summary().class_names,
            Some(vec!["x".to_owned(), "y".to_owned()])
        );
    }
}

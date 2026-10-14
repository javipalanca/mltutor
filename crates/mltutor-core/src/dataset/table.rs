//! Builds a built-in [`Dataset`] from its embedded CSV and its metadata.

use ndarray::Array2;

use super::metadata::DatasetMetadata;
use super::{ColumnKind, Dataset, LabelColumn, Target, TaskType};
use crate::csv::{ColumnData, RawTable};
use crate::error::DatasetError;

fn data_err(dataset: &str, message: String) -> DatasetError {
    DatasetError::Data {
        dataset: dataset.to_owned(),
        message,
    }
}

/// Splits the golden CSV into features (metadata `[[features]]` order), label
/// columns (`label_columns`, D8) and the target (`target`), and attaches the
/// friendly names and class names of the metadata.
///
/// `bool_columns` are stored as 0/1 but have pandas dtype `bool`.
pub fn from_raw_table(
    slug: &str,
    meta: &DatasetMetadata,
    raw: &RawTable,
    bool_columns: &[&str],
) -> Result<Dataset, DatasetError> {
    let n = raw.n_rows;
    let column = |name: &str| {
        raw.column(name)
            .ok_or_else(|| data_err(slug, format!("falta la columna '{name}' en el CSV")))
    };

    let expected = meta.features.len() + meta.label_columns.len() + 1;
    if raw.n_cols() != expected {
        return Err(data_err(
            slug,
            format!(
                "el CSV tiene {} columnas y los metadatos {expected}",
                raw.n_cols()
            ),
        ));
    }

    let mut features = Array2::<f64>::zeros((n, meta.features.len()));
    let mut feature_kinds = Vec::with_capacity(meta.features.len());
    for (j, feat) in meta.features.iter().enumerate() {
        let col = column(&feat.column)?;
        let values = col.data.to_f64().ok_or_else(|| {
            data_err(slug, format!("la columna '{}' no es numérica", feat.column))
        })?;
        for (r, v) in values.into_iter().enumerate() {
            features[[r, j]] = v;
        }
        feature_kinds.push(if bool_columns.contains(&feat.column.as_str()) {
            ColumnKind::Bool
        } else {
            col.data.kind()
        });
    }

    let label_columns = meta
        .label_columns
        .iter()
        .map(|name| {
            let col = column(name)?;
            let values = col
                .data
                .labels()
                .into_iter()
                .map(Option::unwrap_or_default)
                .collect();
            Ok(LabelColumn {
                name: name.clone(),
                values,
            })
        })
        .collect::<Result<Vec<_>, DatasetError>>()?;

    let target_col = column(&meta.target)?;
    let target = match meta.task {
        TaskType::Regression => Target::Values {
            values: target_col
                .data
                .to_f64()
                .ok_or_else(|| data_err(slug, "objetivo no numérico".to_owned()))?,
        },
        TaskType::Classification => {
            let class_names = meta
                .classes
                .clone()
                .ok_or_else(|| data_err(slug, "faltan los nombres de clase".to_owned()))?;
            let ColumnData::Int(raw_codes) = &target_col.data else {
                return Err(data_err(
                    slug,
                    "el objetivo de clasificación no es entero".to_owned(),
                ));
            };
            let codes = raw_codes
                .iter()
                .map(|&c| {
                    u32::try_from(c)
                        .ok()
                        .filter(|&c| (c as usize) < class_names.len())
                        .ok_or_else(|| {
                            data_err(slug, format!("código de clase {c} fuera de rango"))
                        })
                })
                .collect::<Result<Vec<u32>, DatasetError>>()?;
            Target::Classes {
                codes,
                class_values: (0..class_names.len()).map(|c| c as f64).collect(),
                class_names,
            }
        }
    };

    Ok(Dataset {
        label: meta.label.clone(),
        features,
        feature_columns: meta.features.iter().map(|f| f.column.clone()).collect(),
        feature_display: meta.display_names(),
        feature_kinds,
        label_columns,
        target_column: meta.target.clone(),
        target,
        description: meta.description.clone(),
        csv_dialect: None,
    })
}

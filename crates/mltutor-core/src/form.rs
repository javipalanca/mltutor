//! "🔮 Predicciones" form and prediction (fase2-regresion.md C9,
//! `ui.py::create_prediction_interface`).
//!
//! Controls are derived from the **training** rows, column by column:
//!
//! * `nunique ≤ 2` → binary: a select with the metadata texts (first option), a
//!   checkbox if the values are {0, 1} (unchecked), else a select of the values.
//! * `nunique ≤ 10` and (integer column **or** categorical in the metadata) →
//!   categorical: a select of the texts or values, option `len // 2`.
//! * otherwise → continuous: a slider from min to max, step `(max − min)/100`
//!   (0.1 if constant), starting at the training mean.
//!
//! The values sent back are the original codes, in column order. Python rounds
//! them to float32 before predicting (D31); we keep f64.

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::dataset::{ColumnKind, Dataset, DatasetMetadata, TaskType};
use crate::model::{ModelError, Prediction, Predictor};
use crate::pyfmt::py_float_repr;
use crate::stats::mean;

/// Binary features have at most this many distinct values.
pub const BINARY_MAX_UNIQUE: usize = 2;
/// Categorical features have at most this many distinct values.
pub const CATEGORICAL_MAX_UNIQUE: usize = 10;

/// One option of a select: the text shown and the code sent to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectOption {
    pub label: String,
    pub code: f64,
}

/// Kind of control.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ControlKind {
    /// Binary 0/1 feature without texts: unchecked = 0, checked = 1.
    Checkbox { default: bool },
    Select {
        options: Vec<SelectOption>,
        default_index: usize,
    },
    Slider {
        min: f64,
        max: f64,
        step: f64,
        default: f64,
    },
}

/// One control of the form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormControl {
    /// Column index (`feature_{index}` in Python).
    pub index: usize,
    /// Column name (`X.columns`).
    pub column: String,
    /// Friendly name.
    pub name: String,
    /// Description from the metadata.
    pub description: Option<String>,
    /// Markdown label: `**{name}**\n\n*{description}*`, or the name alone.
    pub label: String,
    pub control: ControlKind,
}

impl FormControl {
    /// The code this control sends with its default value.
    pub fn default_code(&self) -> f64 {
        match &self.control {
            ControlKind::Checkbox { default } => f64::from(u8::from(*default)),
            ControlKind::Select {
                options,
                default_index,
            } => options.get(*default_index).map_or(f64::NAN, |o| o.code),
            ControlKind::Slider { default, .. } => *default,
        }
    }
}

/// The form: controls in column order; the first `left_column` go to the left
/// column (`⌈n/2⌉`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictionForm {
    pub controls: Vec<FormControl>,
    pub left_column: usize,
}

impl PredictionForm {
    /// The default values, ready for [`predict_one`].
    pub fn default_values(&self) -> Vec<f64> {
        self.controls
            .iter()
            .map(FormControl::default_code)
            .collect()
    }
}

/// `str(value)` of a pandas value: integers without decimals for integer columns.
fn value_text(v: f64, integer: bool) -> String {
    if integer && v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        py_float_repr(v)
    }
}

/// Builds the form from the training rows of `ds` (`x_train`, same columns) and
/// the dataset's metadata (`None` for CSV files).
pub fn build(
    x_train: ArrayView2<'_, f64>,
    ds: &Dataset,
    metadata: Option<&DatasetMetadata>,
) -> PredictionForm {
    let n = x_train.ncols();
    let mut controls = Vec::with_capacity(n);
    for i in 0..n {
        let column = ds
            .feature_columns
            .get(i)
            .cloned()
            .unwrap_or_else(|| i.to_string());
        let name = ds
            .feature_display
            .get(i)
            .cloned()
            .unwrap_or_else(|| column.clone());
        let kind = ds
            .feature_kinds
            .get(i)
            .copied()
            .unwrap_or(ColumnKind::Float);
        let meta = metadata.and_then(|m| m.feature(&column));
        let description = meta
            .and_then(|f| f.description.clone())
            .filter(|d| !d.is_empty());
        let label = match &description {
            Some(d) => format!("**{name}**\n\n*{d}*"),
            None => name.clone(),
        };

        let col: Vec<f64> = x_train
            .column(i)
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .collect();
        let mut unique = col.clone();
        unique.sort_by(f64::total_cmp);
        unique.dedup();
        let integer = kind == ColumnKind::Int;
        let categorical_meta = meta.is_some_and(|f| f.categorical_in_metadata);
        let mapping = meta.and_then(|f| f.values.as_ref());
        let text_of = |v: f64| -> Option<String> {
            mapping.and_then(|m| {
                m.iter()
                    .find(|(k, _)| k.parse::<f64>().is_ok_and(|kv| kv == v))
                    .map(|(_, t)| t.clone())
            })
        };
        let options = |unique: &[f64]| -> Vec<SelectOption> {
            unique
                .iter()
                .map(|&v| SelectOption {
                    label: text_of(v).unwrap_or_else(|| value_text(v, integer)),
                    code: v,
                })
                .collect()
        };

        let control = if unique.len() <= BINARY_MAX_UNIQUE {
            if mapping.is_some() {
                ControlKind::Select {
                    options: options(&unique),
                    default_index: 0,
                }
            } else if unique.len() == 2 && unique[0] == 0.0 && unique[1] == 1.0 {
                ControlKind::Checkbox { default: false }
            } else {
                ControlKind::Select {
                    options: options(&unique),
                    default_index: 0,
                }
            }
        } else if unique.len() <= CATEGORICAL_MAX_UNIQUE && (integer || categorical_meta) {
            let opts = options(&unique);
            let default_index = if opts.len() > 1 { opts.len() / 2 } else { 0 };
            ControlKind::Select {
                options: opts,
                default_index,
            }
        } else {
            let (min, max) = (
                unique.first().copied().unwrap_or(0.0),
                unique.last().copied().unwrap_or(0.0),
            );
            let step = if max != min { (max - min) / 100.0 } else { 0.1 };
            ControlKind::Slider {
                min,
                max,
                step,
                default: mean(&col),
            }
        };
        controls.push(FormControl {
            index: i,
            column,
            name,
            description,
            label,
            control,
        });
    }
    PredictionForm {
        left_column: n / 2 + n % 2,
        controls,
    }
}

/// Result of "Realizar predicción".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PredictionOutcome {
    /// "Clase: {name}" and, for models with `predict_proba`, the table
    /// Clase/Probabilidad (class code, probability) in `classes_` order.
    Class {
        class: usize,
        proba: Option<Vec<(usize, f64)>>,
    },
    /// "Valor: {value:.4f}". Also for a linear regression on a classification
    /// dataset (D25: Python truncated it with `int()` to a class).
    Value { value: f64 },
}

/// Predicts one sample (`values` = codes in column order).
pub fn predict_one(model: &dyn Predictor, values: &[f64]) -> Result<PredictionOutcome, ModelError> {
    if values.len() != model.n_features() {
        return Err(ModelError::InvalidInput {
            reason: format!(
                "se esperaban {} valores y hay {}",
                model.n_features(),
                values.len()
            ),
        });
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(ModelError::InvalidInput {
            reason: "hay valores vacíos o no numéricos".to_owned(),
        });
    }
    let x = Array2::from_shape_vec((1, values.len()), values.to_vec()).map_err(|e| {
        ModelError::InvalidInput {
            reason: e.to_string(),
        }
    })?;
    match (model.task(), model.predict(x.view())?) {
        (TaskType::Classification, Prediction::Classes(c)) => {
            let class = c.first().copied().ok_or_else(|| ModelError::InvalidInput {
                reason: "sin predicción".to_owned(),
            })?;
            let proba = match model.predict_proba(x.view()) {
                Some(p) => {
                    let p = p?;
                    let classes = model.classes().unwrap_or(&[]);
                    Some(
                        classes
                            .iter()
                            .copied()
                            .zip(p.row(0).iter().copied())
                            .collect(),
                    )
                }
                None => None,
            };
            Ok(PredictionOutcome::Class { class, proba })
        }
        (_, prediction) => Ok(PredictionOutcome::Value {
            value: prediction.to_f64().first().copied().unwrap_or(f64::NAN),
        }),
    }
}

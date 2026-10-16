//! 🔍 Diagnóstico del Entrenamiento (fase5-redes.md C23), pure logic over the history.
//!
//! A line-by-line port of `mltutor/algorithms/model_evaluation.py`
//! (`neural_network_diagnostics`, `analyze_training_diagnostics`,
//! `show_general_health`, `show_trend_analysis`, `show_training_alerts`,
//! `generate_training_recommendations`) and of the four cards of
//! `viz/nn.py::show_training_history_tab`. The output is the view tree the Python
//! lesson draws ([`ViewNode`]), with the texts of `content/redes/mensajes.toml
//! [diagnostico]`; `nn/diagnostics.json` checks it node by node.
//!
//! Details copied on purpose (they decide the texts):
//! - `loss[-n//4:]` is `loss[(-n)//4:]` in Python: the last `ceil(n/4)` epochs.
//! - `x[-0:]` is the whole list (window `min(5, n // 2)` with one epoch).
//! - NumPy means are pairwise sums ([`crate::stats::sum::pairwise_sum`]); `np.std` is
//!   the population standard deviation; `np.polyfit(…, 1)[0]` the least-squares slope.
//! - Stability is `inf` with fewer than 5 epochs (printed "inf%").
//! - The "plateau" and "no patterns" branches are unreachable in Python and are not
//!   ported (D84).
//!
//! Approved deviations (fase5-redes.md §8): with a flat loss and a poor model the
//! "unstable" texts become `estable_rendimiento_bajo`, `patron_rendimiento_bajo` and
//! `alerta_rendimiento_bajo` (same score); `nota_validacion_pequena` follows an
//! accuracy-overfitting alarm with fewer than [`SMALL_VALIDATION`] validation rows;
//! and the texts marked CORREGIDO in `mensajes.toml` replace Python's literals.
//!
//! D84 changes: in regression the critical recommendation says that the validation
//! error does not improve (Python spoke of accuracy, `rec_critico_regresion`), and
//! [`r2_negative_notice`] is the NEW warning of 📈 Evaluación when the test R² is
//! negative.

use serde::{Deserialize, Serialize};

use crate::content::{
    markdown_body, Messages, NN_DIAG_CONTEXT_MD, NN_DIAG_SHORT_MD, NN_DIAG_TRENDS_MD, NN_HISTORY_MD,
};
use crate::dataset::TaskType;
use crate::error::ContentError;
use crate::nn_data::{HistorySeries, Severity};
use crate::pyfmt::{format_fixed, py_format, FmtArg};
use crate::stats::sum::pairwise_sum;
use crate::view::ViewNode;

/// `np.mean` of a list of floats (pairwise sum; NaN for an empty list).
fn np_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    pairwise_sum(values) / values.len() as f64
}

/// `np.std` (population).
fn np_std(values: &[f64]) -> f64 {
    let m = np_mean(values);
    let dev: Vec<f64> = values.iter().map(|v| (v - m) * (v - m)).collect();
    np_mean(&dev).sqrt()
}

/// `values[start:]` with a Python start index (negative counts from the end, `-0` is
/// 0).
fn py_tail(values: &[f64], start: i64) -> &[f64] {
    let n = values.len() as i64;
    let s = if start < 0 {
        (n + start).max(0)
    } else {
        start.min(n)
    };
    &values[usize::try_from(s).unwrap_or(0)..]
}

/// `values[:end]` with a non-negative Python end index.
fn py_head(values: &[f64], end: usize) -> &[f64] {
    &values[..end.min(values.len())]
}

/// Python's floor division of a negative numerator (`-n // d`).
fn neg_floor_div(n: usize, d: usize) -> i64 {
    let n = i64::try_from(n).unwrap_or(i64::MAX);
    let d = i64::try_from(d).unwrap_or(1);
    (-n).div_euclid(d)
}

/// `np.polyfit(range(len(y)), y, 1)[0]`.
fn slope(y: &[f64]) -> f64 {
    let n = y.len();
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mx = np_mean(&xs);
    let my = np_mean(y);
    let sxy: Vec<f64> = xs.iter().zip(y).map(|(x, v)| (x - mx) * (v - my)).collect();
    let sxx: Vec<f64> = xs.iter().map(|x| (x - mx) * (x - mx)).collect();
    pairwise_sum(&sxy) / pairwise_sum(&sxx)
}

/// Overall quality of the training (`overall_quality`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quality {
    Excellent,
    Good,
    Moderate,
    Poor,
}

/// `analyze_training_diagnostics` (keys of the Python dict; `None` = key absent).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diagnostics {
    pub convergence_rate: f64,
    pub loss_converged: bool,
    pub overfitting_gap: f64,
    pub is_overfitting: bool,
    /// `inf` with fewer than 5 epochs.
    pub stability: f64,
    pub loss_stable: bool,
    pub final_train_accuracy: Option<f64>,
    pub final_val_accuracy: Option<f64>,
    pub good_train_accuracy: Option<bool>,
    pub good_val_accuracy: Option<bool>,
    pub excellent_val_accuracy: Option<bool>,
    pub accuracy_overfitting: Option<bool>,
    pub final_mae: Option<f64>,
    pub final_val_mae: Option<f64>,
    pub good_mae: Option<bool>,
    pub overall_quality: Quality,
    pub converged: bool,
    pub is_stable: bool,
}

/// `analyze_training_diagnostics(history, config)`.
pub fn analyze(h: &HistorySeries, task: TaskType) -> Diagnostics {
    let loss = &h.loss;
    let n = loss.len();
    let convergence_rate = if n >= 10 {
        let early = np_mean(py_head(loss, n / 4));
        let late = np_mean(py_tail(loss, neg_floor_div(n, 4)));
        (early - late) / early
    } else if n > 0 {
        (loss[0] - loss[n - 1]) / loss[0]
    } else {
        f64::NAN
    };
    let loss_converged = convergence_rate > 0.01;

    let (overfitting_gap, is_overfitting) = match &h.val_loss {
        Some(val) => {
            let window = (n / 2).min(5);
            let start = -(i64::try_from(window).unwrap_or(0));
            let train_mean = np_mean(py_tail(loss, start));
            let gap = np_mean(py_tail(val, start)) - train_mean;
            let rel = gap / train_mean;
            (rel, rel > 0.15)
        }
        None => (0.0, false),
    };

    let (stability, loss_stable) = if n >= 5 {
        let recent = py_tail(loss, -5);
        let s = np_std(recent) / np_mean(recent);
        (s, s < 0.05)
    } else {
        (f64::INFINITY, false)
    };

    let mut d = Diagnostics {
        convergence_rate,
        loss_converged,
        overfitting_gap,
        is_overfitting,
        stability,
        loss_stable,
        final_train_accuracy: None,
        final_val_accuracy: None,
        good_train_accuracy: None,
        good_val_accuracy: None,
        excellent_val_accuracy: None,
        accuracy_overfitting: None,
        final_mae: None,
        final_val_mae: None,
        good_mae: None,
        overall_quality: Quality::Poor,
        converged: false,
        is_stable: false,
    };
    let last = |v: &Option<Vec<f64>>| v.as_ref().and_then(|v| v.last().copied());
    match task {
        TaskType::Classification => {
            let train_acc = if h.metric_key == "accuracy" {
                last(&h.metric)
            } else {
                None
            };
            match train_acc {
                Some(train_acc) => {
                    d.final_train_accuracy = Some(train_acc);
                    d.good_train_accuracy = Some(train_acc > 0.7);
                    match last(&h.val_metric) {
                        Some(val_acc) => {
                            d.final_val_accuracy = Some(val_acc);
                            d.good_val_accuracy = Some(val_acc > 0.7);
                            d.excellent_val_accuracy = Some(val_acc > 0.85);
                            d.accuracy_overfitting = Some(train_acc - val_acc > 0.15);
                        }
                        None => {
                            d.good_val_accuracy = Some(false);
                            d.excellent_val_accuracy = Some(false);
                            d.accuracy_overfitting = Some(false);
                        }
                    }
                }
                None => {
                    d.good_train_accuracy = Some(false);
                    d.good_val_accuracy = Some(false);
                    d.excellent_val_accuracy = Some(false);
                    d.accuracy_overfitting = Some(false);
                }
            }
            d.overall_quality = if d.good_val_accuracy == Some(true) {
                if d.excellent_val_accuracy == Some(true) {
                    Quality::Excellent
                } else {
                    Quality::Good
                }
            } else if d.good_train_accuracy == Some(true) {
                Quality::Moderate
            } else {
                Quality::Poor
            };
        }
        TaskType::Regression => {
            let mae = if h.metric_key == "mae" {
                h.metric.as_ref()
            } else {
                None
            };
            match mae.and_then(|m| m.last().copied().map(|l| (m, l))) {
                Some((mae, final_mae)) => {
                    d.final_mae = Some(final_mae);
                    let first = np_mean(py_head(mae, 5));
                    match last(&h.val_metric) {
                        Some(val_mae) => {
                            d.final_val_mae = Some(val_mae);
                            d.good_mae = Some(val_mae < first);
                        }
                        None => d.good_mae = Some(final_mae < first),
                    }
                }
                None => d.good_mae = Some(false),
            }
            d.overall_quality = if d.good_mae == Some(true) {
                Quality::Good
            } else {
                Quality::Poor
            };
        }
    }
    d.converged =
        d.loss_converged && matches!(d.overall_quality, Quality::Excellent | Quality::Good);
    d.is_stable = d.loss_stable && d.overall_quality != Quality::Poor;
    d
}

/// `generate_training_recommendations`: the level and its text, and the actions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recommendations {
    pub severity: Severity,
    /// The text without its emoji prefix (`rec_*` of `[diagnostico]`).
    pub text: String,
    pub actions: Vec<String>,
}

/// Slopes of the three thirds of the loss (`[:n//3]`, `[n//3:2n//3]`, `[2n//3:]`)
/// and the population std of the three; `None` with fewer than 10 epochs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trends {
    pub slopes: [f64; 3],
    pub consistency_std: f64,
}

/// [`Trends`] of a loss history.
pub fn trends(loss: &[f64]) -> Option<Trends> {
    let n = loss.len();
    if n < 10 {
        return None;
    }
    let slopes = [
        slope(&loss[..n / 3]),
        slope(&loss[n / 3..2 * n / 3]),
        slope(&loss[2 * n / 3..]),
    ];
    Some(Trends {
        slopes,
        consistency_std: np_std(&slopes),
    })
}

/// [`recommendations`] for `d`.
pub fn recommendations(
    d: &Diagnostics,
    task: TaskType,
    m: &Messages,
) -> Result<Recommendations, ContentError> {
    const S: &str = "diagnostico";
    let quality = d.overall_quality;
    let overfit = d.is_overfitting || d.accuracy_overfitting == Some(true);
    let mut issues = Vec::new();
    if !d.converged {
        issues.push("convergencia");
    }
    if overfit {
        issues.push("sobreajuste");
    }
    if !d.is_stable {
        issues.push("estabilidad");
    }
    if quality == Quality::Poor {
        issues.push("rendimiento_bajo");
    }
    let lines = |key: &str| -> Result<Vec<String>, ContentError> {
        Ok(m.lines(S, key)?.into_iter().map(str::to_owned).collect())
    };
    let (severity, text, mut actions) = if quality == Quality::Excellent && issues.is_empty() {
        (
            Severity::Success,
            m.text(S, "rec_excelente")?.to_owned(),
            lines("rec_excelente_acciones")?,
        )
    } else if quality == Quality::Good && issues.len() <= 1 {
        let actions = if issues.contains(&"sobreajuste") {
            lines("rec_bueno_sobreajuste")?
        } else if issues.contains(&"estabilidad") {
            lines("rec_bueno_estabilidad")?
        } else if task == TaskType::Regression {
            lines("rec_bueno_otro_regresion")?
        } else {
            lines("rec_bueno_otro")?
        };
        // Approved deviation (2026-10-16): in regression "good" only means that the
        // validation MAE went down, so the regression texts say that.
        let key = match task {
            TaskType::Regression => "rec_bueno_regresion",
            TaskType::Classification => "rec_bueno",
        };
        (Severity::Info, m.text(S, key)?.to_owned(), actions)
    } else if quality == Quality::Moderate || issues.len() == 2 {
        (
            Severity::Warning,
            m.text(S, "rec_moderado")?.to_owned(),
            lines("rec_moderado_acciones")?,
        )
    } else {
        let key = match (quality, task) {
            (Quality::Poor, TaskType::Regression) => "rec_critico_regresion",
            (Quality::Poor, TaskType::Classification) => "rec_critico",
            _ => "rec_multiples",
        };
        (
            Severity::Error,
            m.text(S, key)?.to_owned(),
            lines("rec_critico_acciones")?,
        )
    };
    if quality == Quality::Poor {
        actions.insert(0, m.text(S, "rec_pobre_primera")?.to_owned());
    }
    Ok(Recommendations {
        severity,
        text,
        actions,
    })
}

/// Everything 📈 Evaluación shows under "### 🔍 Diagnóstico del Entrenamiento".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticsView {
    pub diagnostics: Diagnostics,
    pub trends: Option<Trends>,
    pub recommendations: Recommendations,
    /// The view tree (children of the page, in order).
    pub blocks: Vec<ViewNode>,
}

fn fmt1(template: &str, name: &str, v: f64) -> String {
    py_format(template, &[(name, FmtArg::Float(v))])
}

/// Below this many validation rows an accuracy-based overfitting alarm gets
/// `nota_validacion_pequena` (each row moves the accuracy by `100 / n` points).
pub const SMALL_VALIDATION: usize = 30;

/// `neural_network_diagnostics(history, config)`: the numbers and the view tree.
/// `n_val` = validation rows (for `nota_validacion_pequena`; `None` = unknown).
pub fn diagnostics_view(
    h: &HistorySeries,
    task: TaskType,
    n_val: Option<usize>,
) -> Result<DiagnosticsView, ContentError> {
    const S: &str = "diagnostico";
    let m = Messages::neural_network()?;
    let t = |k: &str| -> Result<String, ContentError> { Ok(m.text(S, k)?.to_owned()) };
    let d = analyze(h, task);
    // D84 (approved F5a): Python's `is_stable` is `loss_stable ∧ quality ≠ poor`, so a
    // flat loss with a poor model was reported as "inestable" with a tiny CV and the
    // advice to lower the learning rate. Then the `*_rendimiento_bajo` texts are used
    // (same score).
    let stalled = d.loss_stable && d.overall_quality == Quality::Poor;
    let mut blocks = vec![
        ViewNode::text(t("titulo")?),
        ViewNode::text(t("estado_titulo")?),
    ];

    // show_general_health
    let mut score = 0u32;
    let mut col1 = vec![ViewNode::text(t("checklist")?)];
    if d.converged {
        // Regression: "good" performance only means a lower validation MAE than at the
        // start (approved deviation, 2026-10-16).
        let key = match task {
            TaskType::Regression => "convergio_regresion",
            TaskType::Classification => "convergio",
        };
        col1.push(ViewNode::notice(Severity::Success, t(key)?));
        score += 1;
    } else if d.loss_converged {
        col1.push(ViewNode::notice(Severity::Warning, t("convergio_bajo")?));
    } else {
        col1.push(ViewNode::notice(Severity::Error, t("no_convergio")?));
    }
    let acc_overfit = d.accuracy_overfitting == Some(true);
    if !(d.is_overfitting || acc_overfit) {
        col1.push(ViewNode::notice(Severity::Success, t("sin_sobreajuste")?));
        score += 1;
    } else {
        if d.is_overfitting {
            col1.push(ViewNode::notice(
                Severity::Error,
                fmt1(
                    m.text(S, "sobreajuste_perdida")?,
                    "gap",
                    d.overfitting_gap * 100.0,
                ),
            ));
        }
        if acc_overfit {
            col1.push(ViewNode::notice(
                Severity::Error,
                py_format(
                    m.text(S, "sobreajuste_accuracy")?,
                    &[
                        (
                            "train",
                            FmtArg::Float(d.final_train_accuracy.unwrap_or(0.0)),
                        ),
                        ("val", FmtArg::Float(d.final_val_accuracy.unwrap_or(0.0))),
                    ],
                ),
            ));
            if let Some(n) = n_val.filter(|&n| n > 0 && n < SMALL_VALIDATION) {
                col1.push(ViewNode::notice(
                    Severity::Info,
                    py_format(
                        m.text(S, "nota_validacion_pequena")?,
                        &[
                            ("n", FmtArg::Int(i64::try_from(n).unwrap_or(i64::MAX))),
                            ("pp", FmtArg::Float(100.0 / n as f64)),
                        ],
                    ),
                ));
            }
        }
    }
    if d.is_stable {
        col1.push(ViewNode::notice(Severity::Success, t("estable")?));
        score += 1;
    } else {
        let key = if stalled {
            "estable_rendimiento_bajo"
        } else {
            "inestable"
        };
        col1.push(ViewNode::notice(
            Severity::Warning,
            fmt1(m.text(S, key)?, "cv", d.stability * 100.0),
        ));
    }
    match task {
        TaskType::Classification => {
            if d.excellent_val_accuracy == Some(true) {
                col1.push(ViewNode::notice(Severity::Success, t("val_excelente")?));
                score += 1;
            } else if d.good_val_accuracy == Some(true) {
                col1.push(ViewNode::notice(Severity::Success, t("val_bueno")?));
                score += 1;
            } else if d.good_train_accuracy == Some(true) {
                col1.push(ViewNode::notice(Severity::Warning, t("solo_train")?));
            } else {
                col1.push(ViewNode::notice(Severity::Error, t("accuracy_bajo")?));
            }
        }
        TaskType::Regression => {
            if d.good_mae == Some(true) {
                col1.push(ViewNode::notice(
                    Severity::Success,
                    t("regresion_aceptable")?,
                ));
                score += 1;
            } else {
                col1.push(ViewNode::notice(Severity::Error, t("regresion_alto")?));
            }
        }
    }
    let pct = f64::from(score) / 4.0 * 100.0;
    let mut col2 = vec![ViewNode::text(t("puntuacion")?)];
    let (sev, key, status) = match d.overall_quality {
        Quality::Excellent if pct >= 75.0 => {
            (Severity::Success, "salud_excelente", "estado_excelente")
        }
        Quality::Good if pct >= 60.0 => match task {
            TaskType::Regression => (
                Severity::Info,
                "salud_bueno_regresion",
                "estado_bueno_regresion",
            ),
            TaskType::Classification => (Severity::Info, "salud_bueno", "estado_bueno"),
        },
        Quality::Moderate => (Severity::Warning, "salud_moderado", "estado_moderado"),
        _ => (Severity::Error, "salud_critico", "estado_critico"),
    };
    col2.push(ViewNode::notice(sev, fmt1(m.text(S, key)?, "pct", pct)));
    col2.push(ViewNode::notice(Severity::Info, t(status)?));
    match task {
        TaskType::Classification => {
            if let Some(val_acc) = d.final_val_accuracy {
                col2.push(ViewNode::Metric {
                    label: t("metrica_val_accuracy")?,
                    value: format_fixed(val_acc, 3, false),
                    delta: None,
                    delta_color: Some(if val_acc > 0.7 { "normal" } else { "inverse" }.to_owned()),
                    help: Some(t("metrica_val_accuracy_help")?),
                });
            } else if let Some(train_acc) = d.final_train_accuracy {
                col2.push(ViewNode::Metric {
                    label: t("metrica_train_accuracy")?,
                    value: format_fixed(train_acc, 3, false),
                    delta: None,
                    delta_color: None,
                    help: Some(t("metrica_train_accuracy_help")?),
                });
            }
        }
        TaskType::Regression => {
            if let Some(val_mae) = d.final_val_mae {
                col2.push(ViewNode::Metric {
                    label: t("metrica_val_mae")?,
                    value: format_fixed(val_mae, 4, false),
                    delta: None,
                    delta_color: None,
                    help: Some(t("metrica_val_mae_help")?),
                });
            }
        }
    }
    blocks.push(ViewNode::row(vec![col1, col2]));

    // show_trend_analysis
    blocks.push(ViewNode::text(t("tendencias_titulo")?));
    blocks.push(ViewNode::text(t("fases_titulo")?));
    blocks.push(ViewNode::expander(
        t("fases_expander")?,
        vec![ViewNode::text(markdown_body(NN_DIAG_TRENDS_MD))],
    ));
    let tr = trends(&h.loss);
    match &tr {
        Some(tr) => {
            let [early, middle, late] = tr.slopes;
            let direction = |v: f64| -> Result<String, ContentError> {
                t(if v < 0.0 { "descendente" } else { "ascendente" })
            };
            let pick = |key: &str, i: usize| -> Result<String, ContentError> {
                m.lines(S, key)?
                    .get(i)
                    .map(|s| (*s).to_owned())
                    .ok_or_else(|| ContentError {
                        file: "content/redes/mensajes.toml".to_owned(),
                        key: format!("{S}.{key}[{i}]"),
                        message: "falta el elemento".to_owned(),
                    })
            };
            let early_i = if early < -0.1 {
                0
            } else if early < -0.01 {
                1
            } else if early < 0.0 {
                2
            } else {
                3
            };
            let middle_i = if middle < -0.01 {
                0
            } else if middle < 0.0 {
                1
            } else if middle.abs() < 0.001 {
                2
            } else {
                3
            };
            let late_i = if late.abs() < 0.001 {
                0
            } else if late < -0.01 {
                1
            } else if late < 0.0 {
                2
            } else {
                3
            };
            let metric = |label: &str,
                          v: f64,
                          q: &str,
                          i: usize,
                          help: &str|
             -> Result<ViewNode, ContentError> {
                Ok(ViewNode::Metric {
                    label: t(label)?,
                    value: format_fixed(v, 6, false),
                    delta: Some(format!("{} - {}", direction(v)?, pick(q, i)?)),
                    delta_color: None,
                    help: Some(pick(help, i)?),
                })
            };
            blocks.push(ViewNode::row(vec![
                vec![metric(
                    "inicio",
                    early,
                    "inicio_calidad",
                    early_i,
                    "inicio_ayuda",
                )?],
                vec![metric(
                    "medio",
                    middle,
                    "medio_calidad",
                    middle_i,
                    "medio_ayuda",
                )?],
                vec![metric(
                    "final",
                    late,
                    "final_calidad",
                    late_i,
                    "final_ayuda",
                )?],
            ]));
            blocks.push(ViewNode::text(t("comparativo_titulo")?));
            let speeds = [early.abs(), middle.abs(), late.abs()];
            let mut fastest = 0;
            for (i, s) in speeds.iter().enumerate() {
                if *s > speeds[fastest] {
                    fastest = i;
                }
            }
            let mut c1 = vec![
                ViewNode::text(t("velocidad")?),
                ViewNode::notice(
                    Severity::Info,
                    py_format(
                        m.text(S, "fase_activa")?,
                        &[("fase", FmtArg::Str(&pick("fases_nombres", fastest)?))],
                    ),
                ),
            ];
            c1.push(if speeds[0] > speeds[1] && speeds[1] > speeds[2] {
                ViewNode::notice(Severity::Success, t("desaceleracion")?)
            } else if speeds[2] > speeds[0] {
                ViewNode::notice(Severity::Warning, t("aceleracion")?)
            } else {
                ViewNode::notice(Severity::Info, t("mixto")?)
            });
            let std = tr.consistency_std;
            let mut c2 = vec![
                ViewNode::text(t("consistencia")?),
                if std < 0.01 {
                    ViewNode::notice(Severity::Success, t("consistencia_alta")?)
                } else if std < 0.1 {
                    ViewNode::notice(Severity::Info, t("consistencia_media")?)
                } else {
                    ViewNode::notice(Severity::Warning, t("consistencia_baja")?)
                },
            ];
            if early < 0.0 && middle < 0.0 && late >= 0.0 {
                c2.push(ViewNode::notice(Severity::Error, t("patron_sobreajuste")?));
            } else if early < 0.0 && middle < 0.0 && late < 0.0 {
                c2.push(ViewNode::notice(Severity::Success, t("mejora_sostenida")?));
            }
            blocks.push(ViewNode::row(vec![c1, c2]));
        }
        None => {
            blocks.push(ViewNode::notice(Severity::Info, t("corto")?));
            blocks.push(ViewNode::text(markdown_body(NN_DIAG_SHORT_MD)));
        }
    }

    // Patterns (always at least the convergence one).
    blocks.push(ViewNode::text(t("patrones_titulo")?));
    let rate = d.convergence_rate;
    let conv = if rate > 0.5 {
        "convergencia_rapida"
    } else if rate > 0.1 {
        "convergencia_moderada"
    } else if rate > 0.01 {
        "convergencia_lenta"
    } else {
        "sin_convergencia"
    };
    let mut patterns = vec![t(conv)?];
    let mut recs = vec![t(&format!("{conv}_rec"))?];
    if d.is_overfitting {
        patterns.push(fmt1(
            m.text(S, "sobreajuste_progresivo")?,
            "gap",
            d.overfitting_gap * 100.0,
        ));
        recs.push(t("sobreajuste_progresivo_rec")?);
    }
    if !d.is_stable {
        let key = if stalled {
            "patron_rendimiento_bajo"
        } else {
            "inestable_patron"
        };
        patterns.push(fmt1(m.text(S, key)?, "cv", d.stability * 100.0));
        recs.push(t(&format!("{key}_rec"))?);
    }
    let bullet = |s: &String| ViewNode::text(format!("• {s}"));
    let mut p1 = vec![ViewNode::text(t("patrones")?)];
    p1.extend(patterns.iter().map(bullet));
    let mut p2 = vec![ViewNode::text(t("patrones_recomendaciones")?)];
    p2.extend(recs.iter().map(bullet));
    blocks.push(ViewNode::row(vec![p1, p2]));
    blocks.push(ViewNode::expander(
        t("contexto_expander")?,
        vec![ViewNode::text(markdown_body(NN_DIAG_CONTEXT_MD))],
    ));

    // show_training_alerts
    blocks.push(ViewNode::text(t("alertas_titulo")?));
    let mut alerts = Vec::new();
    let alert =
        |prefix: &str, msg: String, action: String| format!("{prefix} **{msg}**\n💡 {action}");
    if !d.converged {
        alerts.push(ViewNode::notice(
            Severity::Error,
            alert(
                "🚨",
                t("alerta_no_convergio")?,
                t("alerta_no_convergio_accion")?,
            ),
        ));
    }
    if d.is_overfitting {
        alerts.push(ViewNode::notice(
            Severity::Error,
            alert(
                "🚨",
                fmt1(
                    m.text(S, "alerta_sobreajuste")?,
                    "gap",
                    d.overfitting_gap * 100.0,
                ),
                t("alerta_sobreajuste_accion")?,
            ),
        ));
    }
    if !d.is_stable {
        alerts.push(ViewNode::notice(Severity::Warning, {
            let key = if stalled {
                "alerta_rendimiento_bajo"
            } else {
                "alerta_inestable"
            };
            alert(
                "⚠️",
                fmt1(m.text(S, key)?, "cv", d.stability * 100.0),
                t(&format!("{key}_accion"))?,
            )
        }));
    }
    if alerts.is_empty() {
        blocks.push(ViewNode::notice(Severity::Success, t("sin_alertas")?));
    } else {
        blocks.extend(alerts);
    }

    // Recommendations
    blocks.push(ViewNode::text(t("recomendaciones_titulo")?));
    let rec = recommendations(&d, task, &m)?;
    let prefix = match rec.severity {
        Severity::Success => "🌟",
        Severity::Info => "✅",
        Severity::Warning => "⚠️",
        Severity::Error => "🚨",
    };
    blocks.push(ViewNode::notice(
        rec.severity,
        format!("{prefix} {}", rec.text),
    ));
    if !rec.actions.is_empty() {
        blocks.push(ViewNode::expander(
            t("acciones_expander")?,
            rec.actions.iter().map(bullet).collect(),
        ));
    }
    Ok(DiagnosticsView {
        diagnostics: d,
        trends: tr,
        recommendations: rec,
        blocks,
    })
}

/// The four cards of 📊 Historial (`show_training_history_tab`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryCards {
    pub final_loss: f64,
    pub initial_loss: f64,
    /// `(initial − final) / initial · 100`.
    pub improvement_pct: f64,
    /// `(final_val_loss, gap, gap_pct)` when there is validation.
    pub validation: Option<(f64, f64, f64)>,
    pub epochs: usize,
    /// CV (%) of the last `min(5, epochs // 2)` losses.
    pub stability_cv: f64,
    /// The row of 4 metrics, as Python draws it.
    pub row: ViewNode,
}

/// [`HistoryCards`] of a history (errors only with broken embedded content).
pub fn history_cards(h: &HistorySeries) -> Result<HistoryCards, ContentError> {
    const S: &str = "historial";
    let m = Messages::neural_network()?;
    let t = |k: &str| -> Result<String, ContentError> { Ok(m.text(S, k)?.to_owned()) };
    let loss = &h.loss;
    let n = loss.len();
    let final_loss = loss.last().copied().unwrap_or(f64::NAN);
    let initial_loss = loss.first().copied().unwrap_or(f64::NAN);
    let improvement_pct = (initial_loss - final_loss) / initial_loss * 100.0;
    let window = (n / 2).min(5);
    let recent = py_tail(loss, -(i64::try_from(window).unwrap_or(0)));
    let cv = np_std(recent) / np_mean(recent) * 100.0;

    let c1 = vec![ViewNode::Metric {
        label: t("perdida_final")?,
        value: format_fixed(final_loss, 6, false),
        delta: Some(format!("-{}%", format_fixed(improvement_pct, 1, false))),
        delta_color: None,
        help: Some(t("perdida_final_help")?),
    }];
    let mut c2 = Vec::new();
    let validation = h
        .val_loss
        .as_ref()
        .and_then(|v| v.last().copied())
        .map(|fv| {
            let gap = fv - final_loss;
            (fv, gap, gap / final_loss * 100.0)
        });
    if let Some((fv, gap, gap_pct)) = validation {
        c2.push(ViewNode::Metric {
            label: t("perdida_val")?,
            value: format_fixed(fv, 6, false),
            delta: Some(py_format(
                m.text(S, "perdida_val_delta")?,
                &[
                    ("gap", FmtArg::Float(gap)),
                    ("gap_pct", FmtArg::Float(gap_pct)),
                ],
            )),
            delta_color: Some(
                if gap_pct.abs() < 10.0 {
                    "normal"
                } else {
                    "inverse"
                }
                .to_owned(),
            ),
            help: Some(t("perdida_val_help")?),
        });
    }
    let c3 = vec![ViewNode::Metric {
        label: t("epocas")?,
        value: n.to_string(),
        delta: None,
        delta_color: None,
        help: Some(t("epocas_help")?),
    }];
    let level = if cv < 1.0 {
        0
    } else if cv < 5.0 {
        1
    } else {
        2
    };
    let values = m
        .table_pairs(S, "estabilidad_valores")?
        .get(level)
        .cloned()
        .unwrap_or_default();
    let c4 = vec![ViewNode::Metric {
        label: py_format(
            m.text(S, "estabilidad")?,
            &[("emoji", FmtArg::Str(&values.0))],
        ),
        value: values.1,
        delta: Some(fmt1(m.text(S, "estabilidad_delta")?, "cv", cv)),
        delta_color: None,
        help: Some(t("estabilidad_help")?),
    }];
    Ok(HistoryCards {
        final_loss,
        initial_loss,
        improvement_pct,
        validation,
        epochs: n,
        stability_cv: cv,
        row: ViewNode::row(vec![c1, c2, c3, c4]),
    })
}

/// Body of the "💡 Interpretación del Entrenamiento" expander of 📊 Historial.
pub fn history_help_markdown() -> &'static str {
    markdown_body(NN_HISTORY_MD)
}

/// NEW (D84): `[evaluacion] aviso_r2_negativo` when the test R² of a regression
/// network is negative (with the default values Diabetes ends at R² ≈ −3 while the
/// Python diagnostics said "✅ Error de regresión aceptable").
pub fn r2_negative_notice(r2: f64) -> Result<Option<String>, ContentError> {
    if r2.is_nan() || r2 >= 0.0 {
        return Ok(None);
    }
    let m = Messages::neural_network()?;
    Ok(Some(fmt1(
        m.text("evaluacion", "aviso_r2_negativo")?,
        "r2",
        r2,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_slices() {
        let v: Vec<f64> = (0..10).map(f64::from).collect();
        // loss[-10//4:] = loss[-3:]
        assert_eq!(py_tail(&v, neg_floor_div(10, 4)), &[7.0, 8.0, 9.0]);
        assert_eq!(py_tail(&v, 0).len(), 10);
        assert!((slope(&[1.0, 3.0, 5.0]) - 2.0).abs() < 1e-15);
    }

    #[test]
    fn r2_notice_only_when_negative() {
        assert_eq!(r2_negative_notice(0.3).ok().flatten(), None);
        let text = r2_negative_notice(-3.4).ok().flatten().unwrap_or_default();
        assert!(text.contains("-3.40"), "{text}");
    }
}

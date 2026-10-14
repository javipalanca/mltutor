//! C3 "🔍 Interpretación de Resultados": the `st.info` Markdown of
//! `show_detailed_evaluation`, built from `content/componentes/mensajes.toml`
//! `[interpretacion]` with the same rules and number formats as Python.
//!
//! The text is returned without trailing whitespace (what the student sees).

use crate::content::Messages;
use crate::error::ContentError;
use crate::metrics::{ClassificationEvaluation, RegressionEvaluation};
use crate::pyfmt::{py_format, FmtArg};

/// Regression: R² in %, MAE and RMSE with 2 decimals and the warning when
/// `RMSE > 1.5·MAE`, plus `reg_r2_negativo` when `R² < 0` (new text). (The text says "regresión lineal" in every lesson, as in Python.)
pub fn regression(eval: &RegressionEvaluation) -> Result<String, ContentError> {
    let m = Messages::components()?;
    let args = [
        ("r2_pct", FmtArg::Float(eval.r2 * 100.0)),
        ("mae", FmtArg::Float(eval.mae)),
        ("rmse", FmtArg::Float(eval.rmse)),
    ];
    let mut out = String::new();
    for line in m.lines("interpretacion", "reg")? {
        out.push_str(&py_format(line, &args));
        out.push('\n');
    }
    if eval.rmse > eval.mae * 1.5 {
        out.push_str(m.text("interpretacion", "reg_rmse_mayor")?);
    }
    if eval.r2 < 0.0 {
        // NUEVO: Python only printed the negative percentage.
        out.push_str(m.text("interpretacion", "reg_r2_negativo")?);
    }
    Ok(out.trim_end().to_owned())
}

/// Classification: accuracy and the weighted precision/recall/F1.
pub fn classification(eval: &ClassificationEvaluation) -> Result<String, ContentError> {
    let m = Messages::components()?;
    let (acc, prec, rec, f1) = (
        eval.accuracy,
        eval.weighted_avg.precision,
        eval.weighted_avg.recall,
        eval.weighted_avg.f1,
    );
    let args = [
        ("acc_pct", FmtArg::Float(acc * 100.0)),
        ("prec_pct", FmtArg::Float(prec * 100.0)),
        ("rec_pct", FmtArg::Float(rec * 100.0)),
        ("f1", FmtArg::Float(f1)),
    ];
    let mut out = String::new();
    for line in m.lines("interpretacion", "cls")? {
        out.push_str(&py_format(line, &args));
        out.push('\n');
    }
    if f1 > 0.0 {
        out.push_str(&py_format(m.text("interpretacion", "cls_f1")?, &args));
    }
    let key = if (prec - rec).abs() > 0.1 {
        if prec > rec {
            "cls_mas_preciso"
        } else {
            "cls_mas_sensible"
        }
    } else {
        "cls_equilibrio"
    };
    out.push_str(m.text("interpretacion", key)?);
    if acc < 0.6 {
        out.push_str(m.text("interpretacion", "cls_mejorar")?);
    }
    Ok(out.trim_end().to_owned())
}

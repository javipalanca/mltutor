//! A small declarative view tree, with the shape that `v1.0.0:tools/golden_render.py`
//! records from the Python lessons (`expected.views` of the fixtures).
//!
//! Views that are pure logic over the model (the 🧠 training diagnostics, the
//! history cards) are built in the core as a tree of these nodes with the texts
//! already filled in from `content/`; the UI renders the tree as it comes. Their
//! parity tests compare the serialised tree with the fixture node by node.

use serde::{Deserialize, Serialize};

pub use crate::nn_data::Severity;

/// One node (`kind` = the Streamlit call that produced it in Python).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ViewNode {
    /// The root.
    Page { children: Vec<ViewNode> },
    /// `st.markdown` (`html` = `unsafe_allow_html`).
    Text { text: String, html: bool },
    /// `st.header` (1), `st.subheader` (2).
    Heading { text: String, level: u8 },
    /// `st.success` / `st.info` / `st.warning` / `st.error`.
    Notice { text: String, severity: Severity },
    /// `st.metric`.
    Metric {
        label: String,
        value: String,
        delta: Option<String>,
        /// Only when the Python call passes `delta_color` (`"normal"`/`"inverse"`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        delta_color: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        help: Option<String>,
    },
    /// `st.columns(weights)`; its children are [`ViewNode::Column`]s.
    Row {
        weights: Vec<u32>,
        children: Vec<ViewNode>,
    },
    /// One column of a [`ViewNode::Row`].
    Column { children: Vec<ViewNode> },
    /// `st.expander(label, expanded)`.
    Expander {
        label: String,
        expanded: bool,
        children: Vec<ViewNode>,
    },
}

impl ViewNode {
    /// Markdown text (no HTML).
    pub fn text(text: impl Into<String>) -> Self {
        ViewNode::Text {
            text: text.into(),
            html: false,
        }
    }

    /// A notice.
    pub fn notice(severity: Severity, text: impl Into<String>) -> Self {
        ViewNode::Notice {
            text: text.into(),
            severity,
        }
    }

    /// A row of equal columns.
    pub fn row(columns: Vec<Vec<ViewNode>>) -> Self {
        ViewNode::Row {
            weights: vec![1; columns.len()],
            children: columns
                .into_iter()
                .map(|children| ViewNode::Column { children })
                .collect(),
        }
    }

    /// A closed expander.
    pub fn expander(label: impl Into<String>, children: Vec<ViewNode>) -> Self {
        ViewNode::Expander {
            label: label.into(),
            expanded: false,
            children,
        }
    }
}

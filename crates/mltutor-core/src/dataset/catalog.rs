//! The 12 built-in datasets, in the order of the "Dataset:" selector.
//!
//! The data are the golden CSVs of `tests/parity/datasets/data/` (D17: a single
//! copy, verified by hash in the tests) and the metadata the TOML files of
//! `content/datasets/`. Both are compiled into the binary: no network at runtime.

use serde::{Deserialize, Serialize};

use super::metadata::DatasetMetadata;
use super::{table, Dataset, TaskType};
use crate::error::DatasetError;

/// Identifier of a built-in dataset. The declaration order is the selector order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetId {
    Iris,
    Wine,
    BreastCancer,
    Digits,
    Diabetes,
    Diamonds,
    Mpg,
    Moons,
    Titanic,
    Tips,
    CaliforniaHousing,
    Penguins,
}

macro_rules! embedded {
    ($slug:literal) => {
        (
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tests/parity/datasets/data/",
                $slug,
                ".csv"
            ))
            .as_slice(),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../content/datasets/",
                $slug,
                ".toml"
            )),
        )
    };
}

impl DatasetId {
    /// All built-in datasets, in selector order.
    pub const ALL: [DatasetId; 12] = [
        DatasetId::Iris,
        DatasetId::Wine,
        DatasetId::BreastCancer,
        DatasetId::Digits,
        DatasetId::Diabetes,
        DatasetId::Diamonds,
        DatasetId::Mpg,
        DatasetId::Moons,
        DatasetId::Titanic,
        DatasetId::Tips,
        DatasetId::CaliforniaHousing,
        DatasetId::Penguins,
    ];

    /// File stem used by the fixtures and the TOML files (`breast_cancer`).
    pub fn slug(self) -> &'static str {
        match self {
            DatasetId::Iris => "iris",
            DatasetId::Wine => "wine",
            DatasetId::BreastCancer => "breast_cancer",
            DatasetId::Digits => "digits",
            DatasetId::Diabetes => "diabetes",
            DatasetId::Diamonds => "diamonds",
            DatasetId::Mpg => "mpg",
            DatasetId::Moons => "moons",
            DatasetId::Titanic => "titanic",
            DatasetId::Tips => "tips",
            DatasetId::CaliforniaHousing => "california_housing",
            DatasetId::Penguins => "penguins",
        }
    }

    /// Inverse of [`DatasetId::slug`].
    pub fn from_slug(slug: &str) -> Result<Self, DatasetError> {
        Self::ALL
            .into_iter()
            .find(|id| id.slug() == slug)
            .ok_or_else(|| DatasetError::UnknownDataset {
                name: slug.to_owned(),
            })
    }

    /// Whether the dataset is synthetic (Moons: the UI shows "Dataset generado
    /// aleatoriamente.").
    pub fn is_synthetic(self) -> bool {
        self == DatasetId::Moons
    }

    /// The embedded CSV (golden data: features in `X.columns` order, then label
    /// columns, then the target).
    pub fn csv_bytes(self) -> &'static [u8] {
        self.embedded().0
    }

    /// The embedded `content/datasets/<slug>.toml`.
    pub fn metadata_toml(self) -> &'static str {
        self.embedded().1
    }

    /// Parsed metadata.
    pub fn metadata(self) -> Result<DatasetMetadata, DatasetError> {
        DatasetMetadata::from_toml(self.slug(), self.metadata_toml())
    }

    /// Columns whose pandas dtype is `bool` (stored as 0/1 in the CSV). pandas
    /// `describe()` leaves them out.
    pub fn bool_columns(self) -> &'static [&'static str] {
        match self {
            DatasetId::Titanic => &["adult_male", "alone"],
            _ => &[],
        }
    }

    fn embedded(self) -> (&'static [u8], &'static str) {
        match self {
            DatasetId::Iris => embedded!("iris"),
            DatasetId::Wine => embedded!("wine"),
            DatasetId::BreastCancer => embedded!("breast_cancer"),
            DatasetId::Digits => embedded!("digits"),
            DatasetId::Diabetes => embedded!("diabetes"),
            DatasetId::Diamonds => embedded!("diamonds"),
            DatasetId::Mpg => embedded!("mpg"),
            DatasetId::Moons => embedded!("moons"),
            DatasetId::Titanic => embedded!("titanic"),
            DatasetId::Tips => embedded!("tips"),
            DatasetId::CaliforniaHousing => embedded!("california_housing"),
            DatasetId::Penguins => embedded!("penguins"),
        }
    }
}

/// One option of the "Dataset:" selector (contract `DatasetEntry` without the
/// session reference).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetEntry {
    pub id: DatasetId,
    /// "🌸 Iris - Clasificación de flores".
    pub label: String,
    /// Markdown line under the selector (`selector_blurb`), if any.
    pub blurb: Option<String>,
    pub task: TaskType,
    /// Moons: the UI shows "Dataset generado aleatoriamente.".
    pub synthetic: bool,
}

/// The 12 built-in datasets in selector order.
pub fn list_builtin() -> Result<Vec<DatasetEntry>, DatasetError> {
    DatasetId::ALL
        .into_iter()
        .map(|id| {
            let meta = id.metadata()?;
            Ok(DatasetEntry {
                id,
                blurb: meta.blurb().map(str::to_owned),
                label: meta.label,
                task: meta.task,
                synthetic: id.is_synthetic(),
            })
        })
        .collect()
}

/// Parses an embedded dataset. Moons is the fixed seed-42 sample (D9), identical
/// to [`super::moons::moons_dataset`]`(42)`.
pub fn load_builtin(id: DatasetId) -> Result<Dataset, DatasetError> {
    let meta = id.metadata()?;
    let raw = crate::csv::read_csv(id.csv_bytes()).map_err(|e| DatasetError::Data {
        dataset: id.slug().to_owned(),
        message: e.to_string(),
    })?;
    table::from_raw_table(id.slug(), &meta, &raw, id.bool_columns())
}

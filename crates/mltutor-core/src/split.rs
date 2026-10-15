//! Train/test split identical to the Python app (D7).
//!
//! `preprocess_data` calls `train_test_split(X, y, test_size, random_state=42)`
//! without stratification. scikit-learn's `ShuffleSplit` takes
//! `RandomState(seed).permutation(n)`, puts the first `ceil(test_size * n)` rows in
//! the test set and the remaining `n - n_test` rows in the training set (sklearn's
//! `_validate_shuffle_split` with `train_size=None`; `floor((1 - t) * n)` would lose a row
//! to rounding, e.g. California: `0.7 * 20640 = 14447.999…`).

use serde::{Deserialize, Serialize};

use crate::error::DatasetError;
use crate::sampling::RandomState;

/// Positional row indices of a train/test split.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainTestSplit {
    /// Training rows, in the order scikit-learn returns them.
    pub train: Vec<usize>,
    /// Test rows, in the order scikit-learn returns them.
    pub test: Vec<usize>,
}

/// `train_test_split(range(n_rows), test_size=test_size, random_state=seed)`.
///
/// `test_size` must be in `(0, 1)` and leave at least one row on each side, as
/// scikit-learn requires.
pub fn train_test_split(
    n_rows: usize,
    test_size: f64,
    seed: u32,
) -> Result<TrainTestSplit, DatasetError> {
    if !(test_size > 0.0 && test_size < 1.0) {
        return Err(DatasetError::InvalidSelection {
            reason: format!("test_size debe estar entre 0 y 1 (es {test_size})"),
        });
    }
    let n = n_rows as f64;
    // Same float expressions as sklearn's `_validate_shuffle_split`.
    let n_test = (test_size * n).ceil() as usize;
    let n_train = n_rows.saturating_sub(n_test);
    if n_test == 0 || n_train == 0 || n_test + n_train > n_rows {
        return Err(DatasetError::InvalidSelection {
            reason: format!(
                "con {n_rows} filas y test_size={test_size} un conjunto quedaría vacío"
            ),
        });
    }
    let perm = RandomState::new(seed).permutation(n_rows);
    Ok(TrainTestSplit {
        test: perm[..n_test].to_vec(),
        train: perm[n_test..n_test + n_train].to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::train_test_split;

    #[test]
    fn rejects_degenerate_sizes() {
        assert!(train_test_split(10, 0.0, 42).is_err());
        assert!(train_test_split(10, 1.0, 42).is_err());
        assert!(train_test_split(1, 0.3, 42).is_err());
        let california = train_test_split(20_640, 0.3, 42).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            (california.test.len(), california.train.len()),
            (6192, 14_448)
        );
        let s = train_test_split(10, 0.3, 42).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((s.train.len(), s.test.len()), (7, 3));
    }
}

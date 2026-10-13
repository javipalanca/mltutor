//! Descriptive statistics with pandas/NumPy/SciPy semantics.
//!
//! Each function documents the Python call it reproduces. Summations follow NumPy's
//! pairwise algorithm so that results agree with Python to ~1e-15 relative
//! (fixtures `tests/parity/stats/*.json`).

pub mod correlation;
pub mod describe;
pub mod histogram;
pub mod kde;
pub mod sum;
pub mod value_counts;

pub use correlation::{pearson, pearson_matrix, pearson_matrix_cancellable};
pub use describe::{describe_column, mean, quantile_linear, std_ddof1, DESCRIBE_STATS};
pub use histogram::{
    histogram, histogram_bin_edges_auto, histogram_counts, linspace, Histogram, HIST_AUTO_MAX_BINS,
};
pub use kde::{gaussian_kde_scott, Kde, KDE_CUT, KDE_GRIDSIZE};
pub use sum::{pairwise_sum, pairwise_sum_by};
pub use value_counts::{value_counts_by_frequency, value_counts_sorted};

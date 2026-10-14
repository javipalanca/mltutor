//! `sklearn.datasets.make_moons` (D9: generated once per session with seed 42).

use std::f64::consts::PI;

use ndarray::Array2;

use super::{ColumnKind, Dataset, DatasetId, Target};
use crate::error::DatasetError;
use crate::sampling::RandomState;
use crate::stats::linspace;

/// `make_moons(n_samples, shuffle=True, noise, random_state=seed)` with
/// scikit-learn's algorithm and NumPy's legacy RNG: two half circles, shuffled
/// with `RandomState.shuffle`, plus `normal(scale=noise)` noise drawn row by row.
///
/// Returns the `n × 2` points and the labels (0 = outer moon, 1 = inner moon).
/// With `(500, 0.2, 42)` it reproduces `tests/parity/datasets/data/moons.csv`.
pub fn make_moons(n_samples: usize, noise: f64, seed: u32) -> (Array2<f64>, Vec<u32>) {
    let n_out = n_samples / 2;
    let n_in = n_samples - n_out;
    let t_out = linspace(0.0, PI, n_out);
    let t_in = linspace(0.0, PI, n_in);

    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);
    for &t in &t_out {
        x.push([t.cos(), t.sin()]);
        y.push(0_u32);
    }
    for &t in &t_in {
        x.push([1.0 - t.cos(), 1.0 - t.sin() - 0.5]);
        y.push(1_u32);
    }

    let mut rng = RandomState::new(seed);
    // sklearn.utils.shuffle → resample(replace=False): shuffle an index array.
    let mut order: Vec<usize> = (0..n_samples).collect();
    rng.shuffle(&mut order);

    let mut points = Array2::<f64>::zeros((n_samples, 2));
    let mut labels = Vec::with_capacity(n_samples);
    for (row, &src) in order.iter().enumerate() {
        points[[row, 0]] = x[src][0];
        points[[row, 1]] = x[src][1];
        labels.push(y[src]);
    }
    for row in 0..n_samples {
        for col in 0..2 {
            points[[row, col]] += rng.normal(0.0, noise);
        }
    }
    (points, labels)
}

/// The Moons dataset (500 points, noise 0.2) generated with `seed`, labelled with
/// the Moons metadata. `seed = 42` equals the embedded dataset.
pub fn moons_dataset(seed: u32) -> Result<Dataset, DatasetError> {
    let meta = DatasetId::Moons.metadata()?;
    let (n, noise) = meta
        .generator
        .as_ref()
        .map_or((500, 0.2), |g| (g.n_samples, g.noise));
    let (features, codes) = make_moons(n, noise, seed);
    let class_names = meta
        .classes
        .clone()
        .unwrap_or_else(|| vec!["class_0".into(), "class_1".into()]);
    Ok(Dataset {
        label: meta.label.clone(),
        features,
        feature_columns: meta.features.iter().map(|f| f.column.clone()).collect(),
        feature_display: meta.display_names(),
        feature_kinds: vec![ColumnKind::Float; 2],
        label_columns: Vec::new(),
        target_column: meta.target.clone(),
        target: Target::Classes {
            codes,
            class_values: vec![0.0, 1.0],
            class_names,
        },
        description: meta.description,
        csv_dialect: None,
    })
}

//! Reproduction of NumPy's legacy `RandomState` and pandas sampling.
//!
//! The Python app samples rows with `DataFrame.sample(random_state=42)` and splits
//! with `train_test_split(random_state=42)`. Both end in
//! `RandomState(42).permutation(n)`: MT19937 seeded with `init_genrand(seed)` and a
//! Fisher-Yates shuffle whose bounded integers use NumPy's masked rejection
//! sampling. Reproducing it exactly gives the same rows as Python (D7).

use rand_mt::Mt;

/// NumPy's legacy `numpy.random.RandomState` (only what MLTutor needs).
///
/// ```
/// use mltutor_core::sampling::RandomState;
/// let perm = RandomState::new(42).permutation(5);
/// assert_eq!(perm, vec![1, 4, 2, 0, 3]); // numpy: RandomState(42).permutation(5)
/// ```
#[derive(Debug, Clone)]
pub struct RandomState {
    mt: Mt,
    cached_gauss: Option<f64>,
}

impl RandomState {
    /// `RandomState(seed)` for an integer seed (`init_genrand`).
    pub fn new(seed: u32) -> Self {
        Self {
            mt: Mt::new(seed),
            cached_gauss: None,
        }
    }

    /// Uniform integer in `0..=max`, as NumPy's legacy `random_interval`.
    fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
        if max <= u64::from(u32::MAX) {
            loop {
                let value = u64::from(self.mt.next_u32()) & mask;
                if value <= max {
                    return value;
                }
            }
        }
        loop {
            let value = self.mt.next_u64() & mask;
            if value <= max {
                return value;
            }
        }
    }

    /// `RandomState.shuffle` of a 1-D sequence (in place).
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            // `i` fits in u64 and the result is <= i, so the conversions are lossless.
            let j = self.interval(i as u64) as usize;
            values.swap(i, j);
        }
    }

    /// `RandomState.permutation(n)`.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut perm: Vec<usize> = (0..n).collect();
        self.shuffle(&mut perm);
        perm
    }

    /// `RandomState.random_sample()`: a double in `[0, 1)` with 53 random bits.
    pub fn random_sample(&mut self) -> f64 {
        let a = self.mt.next_u32() >> 5;
        let b = self.mt.next_u32() >> 6;
        (f64::from(a) * 67_108_864.0 + f64::from(b)) / 9_007_199_254_740_992.0
    }

    /// `RandomState.standard_normal()`: NumPy's legacy polar Box-Muller, which
    /// produces two values per iteration and keeps the second for the next call.
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.cached_gauss.take() {
            return value;
        }
        loop {
            let x1 = 2.0 * self.random_sample() - 1.0;
            let x2 = 2.0 * self.random_sample() - 1.0;
            let r2 = x1 * x1 + x2 * x2;
            if r2 < 1.0 && r2 != 0.0 {
                let f = (-2.0 * r2.ln() / r2).sqrt();
                self.cached_gauss = Some(f * x1);
                return f * x2;
            }
        }
    }

    /// `RandomState.normal(loc, scale)`.
    pub fn normal(&mut self, loc: f64, scale: f64) -> f64 {
        loc + scale * self.standard_normal()
    }
}

/// `DataFrame.sample(n, random_state=seed)` without replacement: the positional
/// rows `permutation(n_rows)[..n]`, in that order. If `n >= n_rows` all rows are
/// returned, permuted.
pub fn sample_rows(n_rows: usize, n: usize, seed: u32) -> Vec<usize> {
    let mut perm = RandomState::new(seed).permutation(n_rows);
    perm.truncate(n.min(n_rows));
    perm
}

/// `df.groupby(key, group_keys=False).apply(lambda g: g.sample(min(len(g), per_group),
/// random_state=seed))`, as the pairplot does for classification.
///
/// `keys[i]` is the group key of row `i`. Groups are visited in ascending key order
/// (pandas sorts the group keys) and each one uses a **fresh** `RandomState(seed)`.
/// Returns positional rows, concatenated group by group.
pub fn groupby_sample<K: Ord + Clone>(keys: &[K], per_group: usize, seed: u32) -> Vec<usize> {
    let mut groups: std::collections::BTreeMap<K, Vec<usize>> = std::collections::BTreeMap::new();
    for (row, key) in keys.iter().enumerate() {
        groups.entry(key.clone()).or_default().push(row);
    }
    let mut out = Vec::new();
    for rows in groups.values() {
        let picked = sample_rows(rows.len(), per_group, seed);
        out.extend(picked.into_iter().map(|i| rows[i]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_matches_numpy() {
        // numpy 1.26: np.random.RandomState(42).permutation(10)
        assert_eq!(
            RandomState::new(42).permutation(10),
            vec![8, 1, 5, 0, 7, 2, 9, 4, 3, 6]
        );
    }

    #[test]
    fn random_sample_matches_numpy() {
        // np.random.RandomState(42).random_sample(2)
        let mut rs = RandomState::new(42);
        assert_eq!(rs.random_sample(), 0.374_540_118_847_362_5);
        assert_eq!(rs.random_sample(), 0.950_714_306_409_916_2);
    }

    #[test]
    fn groupby_sample_uses_a_fresh_state_per_group() {
        let keys = ["b", "a", "b", "a", "b", "a"];
        let rows = groupby_sample(&keys, 2, 42);
        assert_eq!(rows.len(), 4);
        // Group "a" = rows [1, 3, 5] first, then "b" = [0, 2, 4]; same picks in each.
        let perm = sample_rows(3, 2, 42);
        let a = [1, 3, 5];
        let b = [0, 2, 4];
        assert_eq!(rows, vec![a[perm[0]], a[perm[1]], b[perm[0]], b[perm[1]]]);
    }
}

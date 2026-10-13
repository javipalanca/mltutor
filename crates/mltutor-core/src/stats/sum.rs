//! NumPy's pairwise summation (`np.add.reduce` on a contiguous float64 array).

const PW_BLOCKSIZE: usize = 128;

/// Sum of `values` with NumPy's pairwise algorithm.
///
/// Below 8 elements it is a plain loop; up to 128 it uses 8 interleaved
/// accumulators; above that it splits the slice in two halves (rounded to a
/// multiple of 8) recursively. The rounding error is O(log n · ε) instead of O(n · ε)
/// and, more importantly, it is bit-for-bit what `ndarray.sum()` returns.
pub fn pairwise_sum(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in values {
            res += v;
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [0.0_f64; 8];
        r.copy_from_slice(&values[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (k, acc) in r.iter_mut().enumerate() {
                *acc += values[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &values[i..] {
            res += v;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&values[..n2]) + pairwise_sum(&values[n2..])
    }
}

/// [`pairwise_sum`] of `f(0), f(1), …, f(n - 1)` without materializing them: the
/// same additions in the same order, so the result is bit-identical to summing a
/// slice of those values. Used to skip missing values without copying the column.
pub fn pairwise_sum_by(n: usize, f: impl Fn(usize) -> f64) -> f64 {
    pairwise_sum_range(0, n, &f)
}

fn pairwise_sum_range(start: usize, n: usize, f: &impl Fn(usize) -> f64) -> f64 {
    if n < 8 {
        let mut res = 0.0;
        for i in 0..n {
            res += f(start + i);
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [0.0_f64; 8];
        for (k, acc) in r.iter_mut().enumerate() {
            *acc = f(start + k);
        }
        let mut i = 8;
        while i < n - (n % 8) {
            for (k, acc) in r.iter_mut().enumerate() {
                *acc += f(start + i + k);
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for j in i..n {
            res += f(start + j);
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_range(start, n2, f) + pairwise_sum_range(start + n2, n - n2, f)
    }
}

#[cfg(test)]
mod tests {
    use super::{pairwise_sum, pairwise_sum_by};

    #[test]
    fn sum_by_is_bit_identical_to_the_slice_version() {
        let mut rs = crate::sampling::RandomState::new(3);
        for n in [0, 1, 7, 8, 9, 127, 128, 129, 1000, 4097] {
            let x: Vec<f64> = (0..n).map(|_| rs.standard_normal()).collect();
            assert_eq!(
                pairwise_sum(&x).to_bits(),
                pairwise_sum_by(n, |i| x[i]).to_bits(),
                "{n}"
            );
        }
    }

    #[test]
    fn sums_exactly_small_integers() {
        let v: Vec<f64> = (1..=1000).map(f64::from).collect();
        assert_eq!(pairwise_sum(&v), 500_500.0);
        assert_eq!(pairwise_sum(&[]), 0.0);
        assert_eq!(pairwise_sum(&[1.5]), 1.5);
    }

    #[test]
    fn matches_numpy_rounding() {
        // np.full(10, 0.1).sum() == 1.0 and np.full(1000, 0.1).sum() == 100.00000000000001
        // (a naive left-to-right loop gives 0.9999999999999999 and 99.9999999999986).
        assert_eq!(pairwise_sum(&[0.1; 10]), 1.0);
        assert_eq!(pairwise_sum(&[0.1; 1000]), 100.000_000_000_000_01);
    }

    #[test]
    fn matches_numpy_on_1001_normals() {
        // x = np.random.RandomState(0).normal(size=1001); x.sum() == -44.70074481048559
        // (Python's built-in sum, left to right, gives -44.70074481048551).
        let mut rs = crate::sampling::RandomState::new(0);
        let x: Vec<f64> = (0..1001).map(|_| rs.standard_normal()).collect();
        assert_eq!(x[0], 1.764_052_345_967_664);
        assert_eq!(pairwise_sum(&x), -44.700_744_810_485_59);
    }
}

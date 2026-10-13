//! Peak memory of loading a large numeric CSV (Rust review of Phase 1, finding 1).
//!
//! Generates CSV files close to [`MAX_CSV_BYTES`] in a temporary directory and
//! measures the peak heap with a counting global allocator (exact and portable),
//! plus the process peak RSS reported by `getrusage`. Run in release:
//! `cargo test -p mltutor-core --release --test csv_memory -- --ignored --nocapture --test-threads=1`
//!
//! Goal: the load step (`read_csv_file` + `preview`, what the `load_csv` command
//! does) peaks below 4× the file size for numeric CSV files.
//!
//! Measured on 2026-10-13 (Mac arm64, release), peak heap of the load step:
//!
//! | File | Before (String per field) | After (streaming typed columns) |
//! |---|---|---|
//! | 95.4 MiB, 10 M rows `int,float,0/1` | 1506 MiB (15.8×), 4.9 s | 324 MiB (3.4×), 1.1 s |
//! | 93.0 MiB, 3.9 M rows of distinct floats | 511 MiB (5.5×), 3.7 s | 182 MiB (2.0×), 0.6 s |
//!
//! Setting the target afterwards (`to_dataset` + `target_stats` + `describe`, table
//! still alive) adds 624 → 382 MiB and 226 MiB (11.6 s, quadratic `class_index`) →
//! 150 MiB (0.44 s). Process peak RSS of the first case: 2034 → 670 MiB.

// `getrusage` (peak RSS) only exists on Unix; the allocator measurement is portable,
// but the test is a manual benchmark, so it is simply skipped on Windows.
#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use mltutor_core::csv::{read_csv_file, MAX_CSV_BYTES};
use mltutor_core::TaskChoice;

/// `System` plus a count of the live heap bytes and their peak.
struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grow(bytes: usize) {
    let now = CURRENT.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

// SAFETY: every method forwards to `System` with the same arguments; the counters
// are only bookkeeping.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grow(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: same contract as the caller's.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            // Count the worst case of a moving realloc: both blocks alive at once.
            grow(new_size);
            CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        p
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Starts a new measurement window and returns the live bytes at its start.
fn reset_peak() -> usize {
    let now = CURRENT.load(Ordering::Relaxed);
    PEAK.store(now, Ordering::Relaxed);
    now
}

/// Peak above `base` since the last [`reset_peak`].
fn peak_since(base: usize) -> usize {
    PEAK.load(Ordering::Relaxed).saturating_sub(base)
}

/// Process peak RSS in bytes (`ru_maxrss` is bytes on macOS and KiB on Linux).
fn max_rss() -> u64 {
    // SAFETY: `getrusage` only writes into the zeroed struct we pass.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: valid pointer to a `rusage`.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    assert_eq!(rc, 0);
    let raw = u64::try_from(usage.ru_maxrss).unwrap_or(0);
    if cfg!(target_os = "macos") {
        raw
    } else {
        raw * 1024
    }
}

fn write_csv(path: &Path, header: &str, rows: usize, row: impl Fn(usize) -> String) -> u64 {
    let mut out = BufWriter::new(std::fs::File::create(path).unwrap());
    writeln!(out, "{header}").unwrap();
    for i in 0..rows {
        out.write_all(row(i).as_bytes()).unwrap();
    }
    out.flush().unwrap();
    std::fs::metadata(path).unwrap().len()
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Loads `path` like the app and returns the peak/file ratio of the load step.
fn measure(name: &str, path: &Path, file_len: u64) -> f64 {
    assert!(file_len <= MAX_CSV_BYTES, "{file_len}");
    let base = reset_peak();
    let start = std::time::Instant::now();
    let table = read_csv_file(path).unwrap();
    let preview = table.preview().unwrap();
    let load_time = start.elapsed();
    let load_peak = peak_since(base);
    let retained = CURRENT.load(Ordering::Relaxed).saturating_sub(base);

    // What `set_csv_target` does next, with the table still alive.
    let base2 = reset_peak();
    let start = std::time::Instant::now();
    let target = preview.default_target.clone();
    let ds = table
        .to_dataset(&target, TaskChoice::Auto, "big.csv")
        .unwrap();
    let _ = table.target_stats(&target, TaskChoice::Auto).unwrap();
    let _ = table.describe();
    let final_time = start.elapsed();
    let final_peak = peak_since(base2);
    let ratio = load_peak as f64 / file_len as f64;
    println!(
        "{name}: file {:.1} MiB, {} rows × {} cols\n  load (read_csv_file + preview): peak heap {:.1} MiB = {:.2}× file, \
         retained {:.1} MiB, {:?}\n  set target (to_dataset + target_stats + describe): extra peak {:.1} MiB = {:.2}× file, {:?}\n  \
         process peak RSS {:.1} MiB",
        mib(file_len as usize),
        table.n_rows,
        table.n_cols(),
        mib(load_peak),
        ratio,
        mib(retained),
        load_time,
        mib(final_peak),
        final_peak as f64 / file_len as f64,
        final_time,
        max_rss() as f64 / (1024.0 * 1024.0),
    );
    assert_eq!(ds.n_rows(), table.n_rows);
    ratio
}

#[test]
#[ignore = "memory benchmark: run in release with --ignored --nocapture"]
fn large_numeric_csv_peaks_below_4x_file_size() {
    let dir = tempfile::tempdir().unwrap();

    // ~95 MiB, 10 M rows of short numbers (the review's case): int, float, 0/1 class.
    let low = dir.path().join("low_cardinality.csv");
    let len = write_csv(&low, "a,b,y", 10_000_000, |i| {
        format!("{},{}.{},{}\n", 100 + i % 900, (i / 7) % 10, i % 10, i % 2)
    });
    let r1 = measure("10 M short rows", &low, len);
    std::fs::remove_file(&low).unwrap();

    // ~95 MiB of distinct floats: the worst case of `nunique`.
    let high = dir.path().join("high_cardinality.csv");
    let len = write_csv(&high, "x,z,target", 3_900_000, |i| {
        let x = (i as f64 * 0.618_033_988_75).fract();
        let z = (i as f64 * 0.414_213_562_37).fract();
        format!("{x:.6},{z:.6},{:.4}\n", x + z)
    });
    let r2 = measure("3.9 M distinct floats", &high, len);

    assert!(r1 < 4.0, "low cardinality: {r1:.2}× file size");
    assert!(r2 < 4.0, "high cardinality: {r2:.2}× file size");
}

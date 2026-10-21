//! Peak memory of loading a large CSV (Rust review of Phase 1, findings 1 and,
//! for text columns, the last round).
//!
//! Generates CSV files close to [`MAX_CSV_BYTES`] in a temporary directory and
//! measures the peak heap with a counting global allocator (exact and portable),
//! plus the process peak RSS reported by `getrusage`. Run in release, one test
//! per process if you want a meaningful RSS (it is a process-wide maximum):
//! `cargo test -p mltutor-core --release --test csv_memory -- --ignored --nocapture --test-threads=1`
//!
//! Goals: the load step (`read_csv_file` + `preview`, what the `load_csv` command
//! does) peaks below 4× the file size, and the whole sequence with the target set
//! (`to_dataset` + `target_stats` + `describe`, table still alive) below 6× for the
//! categorical case. The heap counter counts requested bytes: before the
//! dictionary encoding, the RSS was higher still because every 2-byte `String`
//! takes a 16-byte malloc block.
//!
//! Numeric files, measured on 2026-10-13 (Mac arm64, release), peak heap:
//!
//! | File | Before (String per field) | After (streaming typed columns) |
//! |---|---|---|
//! | 95.4 MiB, 10 M rows `int,float,0/1` | 1506 MiB (15.8×), 4.9 s | 324 MiB (3.4×), 1.1 s |
//! | 93.0 MiB, 3.9 M rows of distinct floats | 511 MiB (5.5×), 3.7 s | 182 MiB (2.0×), 0.6 s |
//!
//! Setting the target afterwards added 624 → 382 MiB and 226 MiB (11.6 s, quadratic
//! `class_index`) → 150 MiB (0.44 s). On 2026-10-14 `describe`/`target_stats` sort
//! their owned copy in place: 382 → 305 MiB and 150 → 120 MiB (total 5.6× and 2.25×).
//!
//! Text files, measured on 2026-10-14 (Mac arm64, release), before → after the
//! dictionary encoding (`TextColumn`: a `u32` code per row, distinct values once):
//!
//! | File | Load peak | Set-target extra peak | Total peak | Process peak RSS |
//! |---|---|---|---|---|
//! | 94.4 MiB, 9 M rows `ab,cd,ef,1` (3 text columns, int target) | 833 → 266 MiB (8.8× → 2.8×) | 515 → 343 MiB, 1.6 → 0.22 s | 1253 → 515 MiB (13.3× → 5.5×) | 1815 → 578 MiB |
//! | 93.5 MiB, 7 M distinct ids `u0000123456,1` | 491 → 439 MiB (5.3× → 4.7×) | 527 → 160 MiB, 7.5 → 1.4 s | 814 → 439 MiB (8.7× → 4.7×) | n/a (second in the process) |
//!
//! The all-distinct case still peaks above 4× while reading (the hash table and
//! the growing string buffer of the dictionary); it is not asserted.

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

/// Peak/file ratios of [`measure`].
struct Ratios {
    /// `read_csv_file` + `preview`.
    load: f64,
    /// Whole sequence: load, then target set with the table still alive.
    total: f64,
}

/// Loads `path` like the app and returns the peak/file ratios.
fn measure(name: &str, path: &Path, file_len: u64) -> Ratios {
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
    let step_dataset = peak_since(base2);
    let base3 = reset_peak();
    let _ = table.target_stats(&target, TaskChoice::Auto).unwrap();
    let step_stats = peak_since(base3);
    let base4 = reset_peak();
    let _ = table.describe();
    let step_describe = peak_since(base4);
    // `PEAK` was reset by each step: rebuild the peak of the whole window.
    let dataset_bytes = base3.saturating_sub(base2);
    let set_peak = step_dataset
        .max(dataset_bytes + step_stats)
        .max(base4.saturating_sub(base2) + step_describe);
    println!(
        "  steps: to_dataset {:.1} MiB (keeps {:.1}), target_stats {:.1} MiB, describe {:.1} MiB",
        mib(step_dataset),
        mib(dataset_bytes),
        mib(step_stats),
        mib(step_describe)
    );
    let final_time = start.elapsed();
    let final_peak = set_peak;
    let ratio = load_peak as f64 / file_len as f64;
    let total_peak = load_peak.max(retained + final_peak);
    let total = total_peak as f64 / file_len as f64;
    println!(
        "{name}: file {:.1} MiB, {} rows × {} cols\n  load (read_csv_file + preview): peak heap {:.1} MiB = {:.2}× file, \
         retained {:.1} MiB, {:?}\n  set target (to_dataset + target_stats + describe): extra peak {:.1} MiB = {:.2}× file, {:?}\n  \
         total peak heap {:.1} MiB = {:.2}× file; process peak RSS {:.1} MiB",
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
        mib(total_peak),
        total,
        max_rss() as f64 / (1024.0 * 1024.0),
    );
    assert_eq!(ds.n_rows(), table.n_rows);
    Ratios { load: ratio, total }
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
    let r1 = measure("10 M short rows", &low, len).load;
    std::fs::remove_file(&low).unwrap();

    // ~95 MiB of distinct floats: the worst case of `nunique`.
    let high = dir.path().join("high_cardinality.csv");
    let len = write_csv(&high, "x,z,target", 3_900_000, |i| {
        let x = (i as f64 * 0.618_033_988_75).fract();
        let z = (i as f64 * 0.414_213_562_37).fract();
        format!("{x:.6},{z:.6},{:.4}\n", x + z)
    });
    let r2 = measure("3.9 M distinct floats", &high, len).load;

    assert!(r1 < 4.0, "low cardinality: {r1:.2}× file size");
    assert!(r2 < 4.0, "high cardinality: {r2:.2}× file size");
}

#[test]
#[ignore = "memory benchmark: run in release with --ignored --nocapture"]
fn large_categorical_csv_peaks_below_4x_and_6x_file_size() {
    let dir = tempfile::tempdir().unwrap();

    // ~95 MiB, 9 M rows of the review's shape `ab,cd,ef,1`: three low-cardinality
    // text columns and an integer class (the default target).
    const LEVELS: [&str; 6] = ["ab", "cd", "ef", "gh", "ij", "kl"];
    let path = dir.path().join("categorical.csv");
    let len = write_csv(&path, "a,b,c,y", 9_000_000, |i| {
        format!(
            "{},{},{},{}\n",
            LEVELS[i % 6],
            LEVELS[(i / 3) % 4],
            LEVELS[(i / 5) % 3],
            i % 2
        )
    });
    let r = measure("9 M categorical rows", &path, len);
    std::fs::remove_file(&path).unwrap();

    // ~95 MiB of distinct strings (an id column): the worst case of the dictionary.
    let ids = dir.path().join("distinct_text.csv");
    let len = write_csv(&ids, "id,y", 7_000_000, |i| {
        format!("u{:010},{}\n", i * 7919 % 10_000_019, i % 3)
    });
    let d = measure("7 M distinct strings", &ids, len);

    assert!(r.load < 4.0, "categorical load: {:.2}× file size", r.load);
    assert!(
        r.total < 6.0,
        "categorical total: {:.2}× file size",
        r.total
    );
    println!(
        "distinct strings: load {:.2}×, total {:.2}×",
        d.load, d.total
    );
}

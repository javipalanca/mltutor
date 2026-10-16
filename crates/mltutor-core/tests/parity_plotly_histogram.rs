//! `stats::histogram_plotly` against plotly.js itself (fase5-redes.md K2): the
//! histograms of 🧠 Pesos and 📉 Activaciones are binned in Rust and must be the
//! ones `go.Histogram(x, nbinsx)` drew in the Python app.
//!
//! `data/plotly_histogram.json` was produced by running plotly.js-strict-dist-min
//! 4.1.2 under jsdom (`Plotly.newPlot(gd, [{type: "histogram", x, nbinsx}])`, then
//! `gd._fullData[0].xbins` and the bar counts of `gd.calcdata[0]`) on random
//! float32 samples, ReLU outputs, the kernels, biases and activations of
//! `tests/parity/nn/exact` and single values. Bin counts must be identical; the bin
//! width and the first edge equal to 1e-9.
#![allow(clippy::unwrap_used)]

use mltutor_core::stats::histogram_plotly;
use serde_json::Value;

#[test]
fn bins_and_counts_are_plotlys() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/plotly_histogram.json"
    );
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = doc["cases"].as_array().unwrap();
    for (i, c) in cases.iter().enumerate() {
        let x: Vec<f64> = c["x"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let h = histogram_plotly(&x, c["nbinsx"].as_u64().unwrap() as usize);
        let p = &c["plotly"];
        let counts: Vec<u32> = p["counts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(h.counts, counts, "case {i}");
        assert_eq!(h.edges.len(), h.counts.len() + 1, "case {i}");
        let (start, size) = (p["start"].as_f64().unwrap(), p["size"].as_f64().unwrap());
        let width = h.edges[1] - h.edges[0];
        assert!(
            (width - size).abs() <= 1e-9 * size,
            "case {i}: {width} vs {size}"
        );
        // Our first edge is Plotly's first non-empty bin: start + k·size.
        let k = ((h.edges[0] - start) / size).round();
        assert!(
            k >= 0.0 && (h.edges[0] - (start + k * size)).abs() <= 1e-9 * size.max(start.abs()),
            "case {i}: first edge {} vs start {start}",
            h.edges[0]
        );
    }
    assert!(cases.len() >= 50);
}

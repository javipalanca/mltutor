//! Neural networks for MLTutor built on burn (architecture, training, history).
//!
//! Phase 0 skeleton: the crate only exposes its version.

/// Version of the MLTutor workspace this crate was built from.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::VERSION;

    #[test]
    fn version_matches_workspace() {
        assert!(VERSION.starts_with("0.4.0"));
    }
}

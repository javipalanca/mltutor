//! IPC data-transfer objects (fase1-shell-datos.md §5.3) and the command error type.
//!
//! These are the only types the frontend sees. They are produced from the core types by
//! the [`crate::core_facade`] implementations, so `mltutor-core` does not depend on ts-rs.

pub mod error;
pub mod types;

pub use error::CommandError;
pub use types::*;

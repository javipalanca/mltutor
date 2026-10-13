//! Cooperative cancellation of long operations (CSV reading, correlation…).
//!
//! The frontend tags each cancellable request with a `task_id` of its own (a counter of
//! the window) and can cancel it with `cancel_task` while it runs, e.g. from the
//! "Cancelar" button of the status bar or when a newer request supersedes it. The worker
//! receives the `Arc<AtomicBool>` and checks it in its loops; the registry entry is
//! removed when the [`TaskGuard`] is dropped, so finished ids cannot be cancelled.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// Shared token of one operation.
pub type CancelToken = Arc<AtomicBool>;

#[derive(Clone, Default)]
pub struct TaskRegistry {
    running: Arc<Mutex<HashMap<u32, CancelToken>>>,
}

impl TaskRegistry {
    fn lock(&self) -> MutexGuard<'_, HashMap<u32, CancelToken>> {
        // Every update is a single insertion or removal: a poisoned map is still valid.
        self.running
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Registers `id` and returns its token. A repeated id gets a fresh token (the
    /// previous operation keeps its own and can no longer be cancelled by id).
    pub fn start(&self, id: u32) -> TaskGuard {
        let token = CancelToken::default();
        self.lock().insert(id, token.clone());
        TaskGuard {
            registry: self.clone(),
            id,
            token,
        }
    }

    /// Sets the flag of `id`; `false` if it is not running (already finished).
    pub fn cancel(&self, id: u32) -> bool {
        match self.lock().get(&id) {
            Some(token) => {
                token.store(true, Ordering::SeqCst);
                true
            }
            None => false,
        }
    }

    #[cfg(test)]
    pub fn running(&self) -> usize {
        self.lock().len()
    }
}

/// Keeps an operation registered while it runs.
pub struct TaskGuard {
    registry: TaskRegistry,
    id: u32,
    token: CancelToken,
}

impl TaskGuard {
    pub fn token(&self) -> CancelToken {
        self.token.clone()
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        let mut running = self.registry.lock();
        // Only remove our own entry: the id may have been reused by a newer operation.
        if running
            .get(&self.id)
            .is_some_and(|token| Arc::ptr_eq(token, &self.token))
        {
            running.remove(&self.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_sets_the_flag_only_while_running() {
        let registry = TaskRegistry::default();
        let guard = registry.start(7);
        let token = guard.token();
        assert!(!token.load(Ordering::SeqCst));
        assert!(registry.cancel(7));
        assert!(token.load(Ordering::SeqCst));
        drop(guard);
        assert_eq!(registry.running(), 0);
        assert!(!registry.cancel(7));
    }

    #[test]
    fn a_reused_id_keeps_the_newer_entry() {
        let registry = TaskRegistry::default();
        let old = registry.start(1);
        let old_token = old.token();
        let new = registry.start(1);
        drop(old);
        assert_eq!(registry.running(), 1);
        assert!(registry.cancel(1));
        assert!(new.token().load(Ordering::SeqCst));
        assert!(!old_token.load(Ordering::SeqCst));
    }
}

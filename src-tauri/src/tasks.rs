//! Cooperative cancellation of long operations (CSV reading, correlation…).
//!
//! The frontend tags each cancellable request with a `task_id` of its own (a counter of
//! the window) and can cancel it with `cancel_task` while it runs, e.g. from the
//! "Cancelar" button of the status bar or when a newer request supersedes it. The worker
//! receives the `Arc<AtomicBool>` and checks it in its loops; the registry entry is
//! removed when the [`TaskGuard`] is dropped, so finished ids cannot be cancelled.
//!
//! Both commands are async and run concurrently: a `cancel_task` sent right after the
//! request (a drag cancels the previous frame at once) can arrive **before** the
//! command registers its id. Such an id is remembered as *pending* for
//! [`PENDING_TTL`] and the operation then starts already cancelled. Ids that already
//! finished are remembered too (the last [`FINISHED_MEMORY`]), so that a late cancel of
//! a finished operation does not become pending and cancel a reused id (the counter of
//! the window restarts when the page reloads).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Shared token of one operation.
pub type CancelToken = Arc<AtomicBool>;

/// How long a cancel of an id that has not started yet is kept.
pub const PENDING_TTL: Duration = Duration::from_secs(10);
/// Finished ids remembered (and pending cancels kept at most).
pub const FINISHED_MEMORY: usize = 256;

#[derive(Default)]
struct Registry {
    running: HashMap<u32, CancelToken>,
    /// Cancels that arrived before their operation started.
    pending: VecDeque<(u32, Instant)>,
    /// The last ids that finished.
    finished: VecDeque<u32>,
}

impl Registry {
    fn prune(&mut self, now: Instant) {
        self.pending
            .retain(|&(_, at)| now.saturating_duration_since(at) < PENDING_TTL);
        while self.pending.len() > FINISHED_MEMORY {
            self.pending.pop_front();
        }
    }
}

#[derive(Clone, Default)]
pub struct TaskRegistry {
    inner: Arc<Mutex<Registry>>,
}

impl TaskRegistry {
    fn lock(&self) -> MutexGuard<'_, Registry> {
        // Every update is a single insertion or removal: a poisoned map is still valid.
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Registers `id` and returns its token. A repeated id gets a fresh token (the
    /// previous operation keeps its own and can no longer be cancelled by id).
    /// If `id` was cancelled before it started, the token is already set.
    pub fn start(&self, id: u32) -> TaskGuard {
        self.start_at(id, Instant::now())
    }

    fn start_at(&self, id: u32, now: Instant) -> TaskGuard {
        let token = CancelToken::default();
        let mut registry = self.lock();
        registry.prune(now);
        if let Some(i) = registry.pending.iter().position(|&(p, _)| p == id) {
            registry.pending.remove(i);
            token.store(true, Ordering::SeqCst);
        }
        registry.finished.retain(|&f| f != id);
        registry.running.insert(id, token.clone());
        drop(registry);
        TaskGuard {
            registry: self.clone(),
            id,
            token,
        }
    }

    /// Sets the flag of `id`. `false` if it is not running: if it already finished,
    /// nothing happens; otherwise it is kept as pending (it has not started yet).
    pub fn cancel(&self, id: u32) -> bool {
        self.cancel_at(id, Instant::now())
    }

    fn cancel_at(&self, id: u32, now: Instant) -> bool {
        let mut registry = self.lock();
        if let Some(token) = registry.running.get(&id) {
            token.store(true, Ordering::SeqCst);
            return true;
        }
        if !registry.finished.contains(&id) && !registry.pending.iter().any(|&(p, _)| p == id) {
            registry.pending.push_back((id, now));
            registry.prune(now);
        }
        false
    }

    #[cfg(test)]
    pub fn running(&self) -> usize {
        self.lock().running.len()
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
        let mut registry = self.registry.lock();
        // Only remove our own entry: the id may have been reused by a newer operation.
        if registry
            .running
            .get(&self.id)
            .is_some_and(|token| Arc::ptr_eq(token, &self.token))
        {
            registry.running.remove(&self.id);
            registry.finished.push_back(self.id);
            if registry.finished.len() > FINISHED_MEMORY {
                registry.finished.pop_front();
            }
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

    /// The race of `cancel_task` arriving before the command registers its id.
    #[test]
    fn a_cancel_before_the_start_cancels_the_operation() {
        let registry = TaskRegistry::default();
        assert!(!registry.cancel(5));
        let guard = registry.start(5);
        assert!(guard.token().load(Ordering::SeqCst));
        drop(guard);
        // Consumed: a later operation with the same id is not cancelled.
        let again = registry.start(5);
        assert!(!again.token().load(Ordering::SeqCst));
    }

    #[test]
    fn a_late_cancel_of_a_finished_id_is_ignored() {
        let registry = TaskRegistry::default();
        drop(registry.start(3));
        assert!(!registry.cancel(3));
        // The page reloaded and the counter reuses the id: not cancelled.
        assert!(!registry.start(3).token().load(Ordering::SeqCst));
    }

    #[test]
    fn pending_cancels_expire() {
        let registry = TaskRegistry::default();
        let t0 = Instant::now();
        assert!(!registry.cancel_at(9, t0));
        let guard = registry.start_at(9, t0 + PENDING_TTL + Duration::from_millis(1));
        assert!(!guard.token().load(Ordering::SeqCst));
        for id in 100..100 + 2 * FINISHED_MEMORY as u32 {
            registry.cancel_at(id, t0);
        }
        assert!(registry.lock().pending.len() <= FINISHED_MEMORY);
    }
}

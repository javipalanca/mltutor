//! Per-session data state (fase1-shell-datos.md §5.2).
//!
//! As in Python, the selected dataset is a single value shared by the four lessons and
//! the CSV page. UI-only state (active tab, open expanders…) lives in the frontend.
//!
//! Requests that change the selection carry a `seq` from the frontend (a counter of the
//! window, increasing in the order the user acted). Commands run concurrently, so they
//! may reach the session in another order: the request with the highest `seq` wins and
//! older ones do not overwrite it.
//!
//! Methods that release a draft or a registered CSV hand it back instead of dropping it:
//! freeing a big table takes a while, and the caller does it after releasing the lock.

use std::collections::HashMap;

use indexmap::IndexMap;

use crate::core_facade::{CsvDraftHandle, DatasetHandle};
use crate::ipc::{DatasetEntry, DatasetId, DatasetRef, TaskChoice};

/// A CSV registered in the selector as "📄 {archivo}".
/// `file_name`, `target` and `task` are what the lessons (F2+) need to train on it.
#[derive(Clone)]
#[allow(dead_code)]
pub struct CsvEntry {
    pub entry: DatasetEntry,
    pub file_name: String,
    pub target: String,
    pub task: TaskChoice,
    pub dataset: DatasetHandle,
}

/// Why a CSV could not be registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterError {
    /// The draft was replaced (another file was read) or discarded meanwhile.
    StaleDraft,
    /// A newer registration has already been applied.
    Superseded,
}

#[derive(Default)]
pub struct Session {
    pub selected: DatasetRef,
    /// `seq` of the request that set `selected` (0 = the default).
    selection_seq: u64,
    /// `seq` of the last applied CSV registration.
    csv_seq: u64,
    /// In load order; the key is the `CsvId` of `DatasetRef::Csv`. At most one entry:
    /// registering another file releases the previous one (memory, finding 7).
    csv: IndexMap<u32, CsvEntry>,
    next_csv_id: u32,
    /// The file read last, waiting for its target (only one at a time, like the uploader).
    draft: Option<(u32, CsvDraftHandle)>,
    next_draft_id: u32,
    /// Parsed built-in datasets. Moons is generated once and stays fixed (D9).
    cache: HashMap<DatasetId, DatasetHandle>,
}

impl Session {
    pub fn cached(&self, id: DatasetId) -> Option<DatasetHandle> {
        self.cache.get(&id).cloned()
    }

    /// Keeps the first handle if another thread loaded the same dataset meanwhile.
    pub fn cache_builtin(&mut self, id: DatasetId, dataset: DatasetHandle) -> DatasetHandle {
        self.cache.entry(id).or_insert(dataset).clone()
    }

    pub fn csv_entry(&self, id: u32) -> Option<&CsvEntry> {
        self.csv.get(&id)
    }

    pub fn csv_entries(&self) -> impl Iterator<Item = &CsvEntry> {
        self.csv.values()
    }

    /// Replaces the current draft; returns its id and the previous draft, to be dropped
    /// outside the lock.
    #[must_use]
    pub fn store_draft(&mut self, draft: CsvDraftHandle) -> (u32, Option<CsvDraftHandle>) {
        self.next_draft_id = self.next_draft_id.wrapping_add(1);
        let id = self.next_draft_id;
        let previous = self.draft.replace((id, draft)).map(|(_, old)| old);
        (id, previous)
    }

    pub fn draft(&self, id: u32) -> Option<CsvDraftHandle> {
        match &self.draft {
            Some((draft_id, draft)) if *draft_id == id => Some(draft.clone()),
            _ => None,
        }
    }

    /// Takes the current draft out (another file is about to be read).
    #[must_use]
    pub fn clear_draft(&mut self) -> Option<CsvDraftHandle> {
        self.draft.take().map(|(_, draft)| draft)
    }

    /// Takes draft `id` out if it is still the current one ("Quitar archivo").
    #[must_use]
    pub fn discard_draft(&mut self, id: u32) -> Option<CsvDraftHandle> {
        if self
            .draft
            .as_ref()
            .is_some_and(|(draft_id, _)| *draft_id == id)
        {
            self.clear_draft()
        } else {
            None
        }
    }

    /// The highest `seq` applied so far (selection or CSV registration). A reloaded page
    /// continues from here even if the clock went backwards.
    pub fn last_seq(&self) -> u64 {
        self.selection_seq.max(self.csv_seq)
    }

    /// Registers the CSV of draft `draft_id` and selects it (unless a newer selection
    /// exists). A file with the same display name keeps its id and position, as
    /// `csv_datasets[name] = …` does; any other registered CSV is released, so the
    /// session holds at most one. `make` receives the id so the entry can reference itself.
    /// Also returns the entries it replaced, to be dropped outside the lock.
    pub fn register_csv(
        &mut self,
        draft_id: u32,
        seq: u64,
        display_name: &str,
        make: impl FnOnce(DatasetRef) -> CsvEntry,
    ) -> Result<(&CsvEntry, Vec<CsvEntry>), RegisterError> {
        if self.draft(draft_id).is_none() {
            return Err(RegisterError::StaleDraft);
        }
        if seq <= self.csv_seq {
            return Err(RegisterError::Superseded);
        }
        self.csv_seq = seq;
        let existing = self
            .csv
            .iter()
            .find(|(_, entry)| entry.entry.label == display_name)
            .map(|(id, _)| *id);
        let id = existing.unwrap_or_else(|| {
            self.next_csv_id = self.next_csv_id.wrapping_add(1);
            self.next_csv_id
        });
        // Every previous entry goes: another file, or this one being replaced. With at
        // most one entry, the position of a same-name file is kept trivially.
        let released: Vec<CsvEntry> = std::mem::take(&mut self.csv).into_values().collect();
        let dataset = DatasetRef::Csv { id };
        self.csv.insert(id, make(dataset));
        let selected_csv_was_released = matches!(
            self.selected,
            DatasetRef::Csv { id: old } if old != id
        );
        if seq > self.selection_seq || selected_csv_was_released {
            self.selected = dataset;
            self.selection_seq = self.selection_seq.max(seq);
        }
        Ok((&self.csv[&id], released))
    }

    /// Selects `dataset` for request `seq`. Returns `false` for an unknown CSV id; an
    /// older `seq` than the current selection is valid but changes nothing.
    pub fn select(&mut self, dataset: DatasetRef, seq: u64) -> bool {
        if let DatasetRef::Csv { id } = dataset {
            if !self.csv.contains_key(&id) {
                return false;
            }
        }
        if seq > self.selection_seq {
            self.selected = dataset;
            self.selection_seq = seq;
        }
        true
    }
}

//! Per-session data state (fase1-shell-datos.md §5.2).
//!
//! As in Python, the selected dataset is a single value shared by the four lessons and
//! the CSV page. UI-only state (active tab, open expanders…) lives in the frontend.

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

#[derive(Default)]
pub struct Session {
    pub selected: DatasetRef,
    /// In load order; the key is the `CsvId` of `DatasetRef::Csv`.
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

    pub fn store_draft(&mut self, draft: CsvDraftHandle) -> u32 {
        self.next_draft_id += 1;
        let id = self.next_draft_id;
        self.draft = Some((id, draft));
        id
    }

    pub fn draft(&self, id: u32) -> Option<CsvDraftHandle> {
        match &self.draft {
            Some((draft_id, draft)) if *draft_id == id => Some(draft.clone()),
            _ => None,
        }
    }

    /// Registers a CSV and selects it. A file with the same display name replaces the
    /// previous entry (keeping its id and position), as `csv_datasets[name] = …` does.
    /// `make_entry` receives the id so the `DatasetEntry` can reference itself.
    pub fn register_csv(
        &mut self,
        display_name: &str,
        make: impl FnOnce(DatasetRef) -> CsvEntry,
    ) -> &CsvEntry {
        let existing = self
            .csv
            .iter()
            .find(|(_, entry)| entry.entry.label == display_name)
            .map(|(id, _)| *id);
        let id = existing.unwrap_or_else(|| {
            self.next_csv_id += 1;
            self.next_csv_id
        });
        let dataset = DatasetRef::Csv { id };
        let entry = make(dataset);
        self.csv.insert(id, entry);
        self.selected = dataset;
        &self.csv[&id]
    }

    /// Selects a dataset if it exists; returns `false` for an unknown CSV id.
    pub fn select(&mut self, dataset: DatasetRef) -> bool {
        if let DatasetRef::Csv { id } = dataset {
            if !self.csv.contains_key(&id) {
                return false;
            }
        }
        self.selected = dataset;
        true
    }
}

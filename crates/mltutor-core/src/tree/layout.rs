//! Buchheim's tree layout (Reingold-Tilford in linear time), ported line by line
//! from `sklearn/tree/_reingold_tilford.py`, which `plot_tree` uses to place the
//! boxes. Each node gets an `x` (in units of "one box") and its depth as `y`; the
//! diagram puts node `i` at `((x + 0.5)/max_x, (max_y − y − 0.5)/max_y)` of the
//! axes, with `max_x, max_y` = the largest `x` and `y` plus one.
//!
//! The nodes live in an arena (`Vec<DrawNode>`) instead of Python objects; the
//! algorithm and the order of the floating-point operations are the same.

use super::DecisionTree;

#[derive(Debug, Clone)]
struct DrawNode {
    x: f64,
    y: usize,
    children: Vec<usize>,
    parent: Option<usize>,
    thread: Option<usize>,
    modifier: f64,
    ancestor: usize,
    change: f64,
    shift: f64,
    /// Position among its siblings, from 1.
    number: usize,
    /// Node id in the decision tree.
    tree_node: usize,
}

struct Arena {
    n: Vec<DrawNode>,
}

impl Arena {
    fn left(&self, v: usize) -> Option<usize> {
        self.n[v]
            .thread
            .or_else(|| self.n[v].children.first().copied())
    }

    fn right(&self, v: usize) -> Option<usize> {
        self.n[v]
            .thread
            .or_else(|| self.n[v].children.last().copied())
    }

    fn lbrother(&self, v: usize) -> Option<usize> {
        let parent = self.n[v].parent?;
        let siblings = &self.n[parent].children;
        let pos = siblings.iter().position(|&c| c == v)?;
        pos.checked_sub(1).map(|i| siblings[i])
    }

    fn lmost_sibling(&self, v: usize) -> Option<usize> {
        let parent = self.n[v].parent?;
        let first = *self.n[parent].children.first()?;
        (first != v).then_some(first)
    }

    fn first_walk(&mut self, v: usize, distance: f64) {
        if self.n[v].children.is_empty() {
            self.n[v].x = match self.lmost_sibling(v).and(self.lbrother(v)) {
                Some(b) => self.n[b].x + distance,
                None => 0.0,
            };
        } else {
            let children = self.n[v].children.clone();
            let mut default_ancestor = children[0];
            for &w in &children {
                self.first_walk(w, 1.0);
                default_ancestor = self.apportion(w, default_ancestor, distance);
            }
            self.execute_shifts(v);
            let first = children[0];
            let last = children[children.len() - 1];
            let midpoint = (self.n[first].x + self.n[last].x) / 2.0;
            match self.lbrother(v) {
                Some(w) => {
                    self.n[v].x = self.n[w].x + distance;
                    self.n[v].modifier = self.n[v].x - midpoint;
                }
                None => self.n[v].x = midpoint,
            }
        }
    }

    fn apportion(&mut self, v: usize, mut default_ancestor: usize, distance: f64) -> usize {
        let Some(w) = self.lbrother(v) else {
            return default_ancestor;
        };
        let (mut vir, mut vor) = (v, v);
        let mut vil = w;
        let mut vol = self.lmost_sibling(v).unwrap_or(w);
        let mut sir = self.n[v].modifier;
        let mut sor = sir;
        let mut sil = self.n[vil].modifier;
        let mut sol = self.n[vol].modifier;
        while let (Some(r), Some(l)) = (self.right(vil), self.left(vir)) {
            vil = r;
            vir = l;
            vol = self.left(vol).unwrap_or(vol);
            vor = self.right(vor).unwrap_or(vor);
            self.n[vor].ancestor = v;
            let shift = (self.n[vil].x + sil) - (self.n[vir].x + sir) + distance;
            if shift > 0.0 {
                let a = self.ancestor(vil, v, default_ancestor);
                self.move_subtree(a, v, shift);
                sir += shift;
                sor += shift;
            }
            sil += self.n[vil].modifier;
            sir += self.n[vir].modifier;
            sol += self.n[vol].modifier;
            sor += self.n[vor].modifier;
        }
        if self.right(vil).is_some() && self.right(vor).is_none() {
            self.n[vor].thread = self.right(vil);
            self.n[vor].modifier += sil - sor;
        } else {
            if self.left(vir).is_some() && self.left(vol).is_none() {
                self.n[vol].thread = self.left(vir);
                self.n[vol].modifier += sir - sol;
            }
            default_ancestor = v;
        }
        default_ancestor
    }

    fn move_subtree(&mut self, wl: usize, wr: usize, shift: f64) {
        let subtrees = self.n[wr].number as f64 - self.n[wl].number as f64;
        self.n[wr].change -= shift / subtrees;
        self.n[wr].shift += shift;
        self.n[wl].change += shift / subtrees;
        self.n[wr].x += shift;
        self.n[wr].modifier += shift;
    }

    fn execute_shifts(&mut self, v: usize) {
        let (mut shift, mut change) = (0.0, 0.0);
        let children = self.n[v].children.clone();
        for &w in children.iter().rev() {
            self.n[w].x += shift;
            self.n[w].modifier += shift;
            change += self.n[w].change;
            shift += self.n[w].shift + change;
        }
    }

    fn ancestor(&self, vil: usize, v: usize, default_ancestor: usize) -> usize {
        let a = self.n[vil].ancestor;
        match self.n[v].parent {
            Some(p) if self.n[p].children.contains(&a) => a,
            _ => default_ancestor,
        }
    }

    fn second_walk(&mut self, v: usize, m: f64, depth: usize, min: Option<f64>) -> Option<f64> {
        self.n[v].x += m;
        self.n[v].y = depth;
        let mut min = match min {
            Some(old) if self.n[v].x >= old => Some(old),
            _ => Some(self.n[v].x),
        };
        let children = self.n[v].children.clone();
        let next = m + self.n[v].modifier;
        for w in children {
            min = self.second_walk(w, next, depth + 1, min);
        }
        min
    }

    fn third_walk(&mut self, v: usize, n: f64) {
        self.n[v].x += n;
        let children = self.n[v].children.clone();
        for w in children {
            self.third_walk(w, n);
        }
    }
}

/// Position of every node of a tree.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeLayout {
    /// `x` of each node (index = node id).
    pub x: Vec<f64>,
    /// Depth of each node.
    pub y: Vec<usize>,
    /// `max_extents()[0] + 1`.
    pub max_x: f64,
    /// `max_extents()[1] + 1`.
    pub max_y: f64,
}

impl TreeLayout {
    /// Axes coordinates of `plot_tree` (`xy` of the annotation), in `[0, 1]`.
    pub fn axes_xy(&self, node: usize) -> [f64; 2] {
        [
            (self.x[node] + 0.5) / self.max_x,
            (self.max_y - self.y[node] as f64 - 0.5) / self.max_y,
        ]
    }
}

/// `buchheim(tree)` of scikit-learn for a fitted decision tree.
pub fn buchheim(tree: &DecisionTree) -> TreeLayout {
    let nodes = tree.nodes();
    let mut arena = Arena { n: Vec::new() };
    if nodes.is_empty() {
        return TreeLayout {
            x: Vec::new(),
            y: Vec::new(),
            max_x: 1.0,
            max_y: 1.0,
        };
    }
    // Build the DrawTree (arena index for each tree node).
    let mut stack = vec![(0usize, None::<usize>, 0usize, 1usize)];
    while let Some((tn, parent, depth, number)) = stack.pop() {
        let id = arena.n.len();
        arena.n.push(DrawNode {
            x: -1.0,
            y: depth,
            children: Vec::new(),
            parent,
            thread: None,
            modifier: 0.0,
            ancestor: id,
            change: 0.0,
            shift: 0.0,
            number,
            tree_node: tn,
        });
        if let Some(p) = parent {
            arena.n[p].children.push(id);
        }
        if let Some(s) = nodes[tn].split {
            // Pushed right first so that the left child gets the arena slot (and
            // the children list position) first.
            stack.push((s.right, Some(id), depth + 1, 2));
            stack.push((s.left, Some(id), depth + 1, 1));
        }
    }
    // The stack visits the left child first, so children are [left, right].
    arena.first_walk(0, 1.0);
    if let Some(min) = arena.second_walk(0, 0.0, 0, None) {
        if min < 0.0 {
            arena.third_walk(0, -min);
        }
    }
    let mut x = vec![0.0; nodes.len()];
    let mut y = vec![0; nodes.len()];
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, 0usize);
    for d in &arena.n {
        x[d.tree_node] = d.x;
        y[d.tree_node] = d.y;
        max_x = max_x.max(d.x);
        max_y = max_y.max(d.y);
    }
    TreeLayout {
        x,
        y,
        max_x: max_x + 1.0,
        max_y: max_y as f64 + 1.0,
    }
}

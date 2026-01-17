// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 LTSU

//! Topology planning for DT-CWPT tree structure.
//!
//! Manages the binary tree structure using heap-based indexing:
//! - Root = 1
//! - Left child of i = 2*i
//! - Right child of i = 2*i + 1


/// Maximum tree size (supports up to depth 9)
pub const MAX_TREE_SIZE: usize = 1024;

/// Plans and manages the filter bank topology.
///
/// Converts user-specified destination paths (e.g., "LLH", "LHL") into
/// a processing order for analysis and synthesis.
#[derive(Clone)]
pub struct TopologyPlanner {
    /// Active node flags
    is_active: Vec<bool>,
    /// Analysis processing order (parent -> children)
    pub analysis_order: Vec<usize>,
    /// Synthesis processing order (children -> parent)
    pub synthesis_order: Vec<usize>,
    /// Destination node indices
    pub destinations: Vec<usize>,
    /// Nodes eligible for morphing (excludes pure L and pure H paths)
    pub morphing_list: Vec<usize>,
}

impl TopologyPlanner {
    /// Create a new topology planner from destination path strings.
    ///
    /// # Arguments
    /// * `destinations` - List of destination paths, e.g., ["LL", "LH", "HL", "HH"]
    pub fn new(destinations: &[String]) -> Self {
        let mut planner = Self {
            is_active: vec![false; MAX_TREE_SIZE],
            analysis_order: Vec::new(),
            synthesis_order: Vec::new(),
            destinations: Vec::new(),
            morphing_list: Vec::new(),
        };

        planner.build_topology(destinations);
        planner
    }

    /// Convert a path string to a heap index.
    ///
    /// "L" -> left child, "H" -> right child
    /// Examples: "" -> 1, "L" -> 2, "H" -> 3, "LL" -> 4, "LH" -> 5
    pub fn path_to_index(path: &str) -> usize {
        let mut idx = 1; // Root
        for ch in path.chars() {
            idx <<= 1; // *2
            if ch == 'H' {
                idx |= 1; // +1
            }
        }
        idx
    }

    /// Convert a heap index back to a path string.
    pub fn index_to_path(idx: usize) -> String {
        if idx == 0 {
            return String::new();
        }
        
        let mut path = String::new();
        let bits = (usize::BITS - idx.leading_zeros()) as usize;
        
        for bit_pos in (0..bits - 1).rev() {
            if (idx >> bit_pos) & 1 == 0 {
                path.push('L');
            } else {
                path.push('H');
            }
        }
        path
    }

    /// Build the topology from destination paths.
    fn build_topology(&mut self, destinations: &[String]) {
        // 1. Mark all required nodes as active
        for path in destinations {
            let idx = Self::path_to_index(path);
            self.destinations.push(idx);
            self.is_active[idx] = true;

            // Mark all ancestors up to root
            let mut curr = idx;
            while curr >= 1 {
                self.is_active[curr] = true;
                curr >>= 1;
            }
        }

        // 2. Build analysis order (parent -> children, ascending index)
        for i in 1..MAX_TREE_SIZE {
            if self.is_active[i] {
                let left_child = i << 1;
                if left_child < MAX_TREE_SIZE && self.is_active[left_child] {
                    self.analysis_order.push(i);
                }
            }
        }

        // 3. Build synthesis order (children -> parent, descending index)
        for i in (1..MAX_TREE_SIZE).rev() {
            if self.is_active[i] {
                let left_child = i << 1;
                if left_child < MAX_TREE_SIZE && self.is_active[left_child] {
                    self.synthesis_order.push(i);
                }
            }
        }

        // 4. Build morphing list (exclude pure L and pure H paths)
        for &dest in &self.destinations {
            // Pure L path: 1, 10, 100, 1000, ... (power of 2)
            // Pure H path: 1, 11, 111, 1111, ... (2^n - 1)
            let is_pure_l = dest.is_power_of_two();
            let is_pure_h = (dest & (dest + 1)) == 0;
            
            if !is_pure_l && !is_pure_h {
                self.morphing_list.push(dest);
            }
        }
    }

    /// Check if a node index is active.
    pub fn is_node_active(&self, idx: usize) -> bool {
        idx < MAX_TREE_SIZE && self.is_active[idx]
    }

    /// Get the depth of a node (root = 0, first level children = 1, etc.)
    pub fn node_depth(idx: usize) -> usize {
        if idx == 0 {
            return 0;
        }
        (usize::BITS - idx.leading_zeros() - 1) as usize
    }

    /// Calculate total delay for a path through synthesis nodes.
    pub fn calculate_path_delay<F>(&self, target_idx: usize, get_delay: F) -> usize
    where
        F: Fn(usize, bool) -> usize, // (node_idx, is_high) -> delay
    {
        let mut total_delay = 0;
        let mut coef = 1;
        let num_bits = (usize::BITS - target_idx.leading_zeros()) as usize;
        let mut current_node = 1;

        for bit_pos in (0..num_bits - 1).rev() {
            let direction = (target_idx >> bit_pos) & 1;
            let delay = get_delay(current_node, direction == 1);

            if direction == 0 {
                current_node <<= 1;
            } else {
                current_node = (current_node << 1) | 1;
            }

            total_delay += delay * coef;
            coef <<= 1;
        }

        total_delay
    }
}

impl Default for TopologyPlanner {
    fn default() -> Self {
        // Default: 6-level DWT decomposition (low-only recursive split)
        // Structure: H, LH, LLH, LLLH, LLLLH, LLLLLH, LLLLLL
        Self::new(&[
            "H".to_string(),      // Level 1 high
            "LH".to_string(),     // Level 2 high
            "LLH".to_string(),    // Level 3 high
            "LLLH".to_string(),   // Level 4 high
            "LLLLH".to_string(),  // Level 5 high
            "LLLLLH".to_string(), // Level 6 high
            "LLLLLL".to_string(), // Level 6 low (deepest)
        ])
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_to_index() {
        assert_eq!(TopologyPlanner::path_to_index(""), 1);
        assert_eq!(TopologyPlanner::path_to_index("L"), 2);
        assert_eq!(TopologyPlanner::path_to_index("H"), 3);
        assert_eq!(TopologyPlanner::path_to_index("LL"), 4);
        assert_eq!(TopologyPlanner::path_to_index("LH"), 5);
        assert_eq!(TopologyPlanner::path_to_index("HL"), 6);
        assert_eq!(TopologyPlanner::path_to_index("HH"), 7);
    }

    #[test]
    fn test_index_to_path() {
        assert_eq!(TopologyPlanner::index_to_path(1), "");
        assert_eq!(TopologyPlanner::index_to_path(2), "L");
        assert_eq!(TopologyPlanner::index_to_path(3), "H");
        assert_eq!(TopologyPlanner::index_to_path(4), "LL");
        assert_eq!(TopologyPlanner::index_to_path(5), "LH");
    }

    #[test]
    fn test_node_depth() {
        assert_eq!(TopologyPlanner::node_depth(1), 0);
        assert_eq!(TopologyPlanner::node_depth(2), 1);
        assert_eq!(TopologyPlanner::node_depth(3), 1);
        assert_eq!(TopologyPlanner::node_depth(4), 2);
        assert_eq!(TopologyPlanner::node_depth(7), 2);
        assert_eq!(TopologyPlanner::node_depth(8), 3);
    }

    #[test]
    fn test_simple_topology() {
        let planner = TopologyPlanner::new(&["L".to_string(), "H".to_string()]);
        
        // Root should be in analysis order
        assert!(planner.analysis_order.contains(&1));
        
        // Destinations should be L=2 and H=3
        assert!(planner.destinations.contains(&2));
        assert!(planner.destinations.contains(&3));
    }
}

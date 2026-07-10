// Phase 1: simple Vec-backed array buffer.
//
// Fixed capacity. write_at wraps to 0 when full — oldest data silently
// overwritten. No atomics, no mmap, no multiprocessing yet.
// Phase 2 will replace the Vec with mmap(MAP_SHARED) and write_at with
// AtomicUsize::fetch_add, touching nothing else.

use crate::types::PpoTransition;

pub struct SimpleBuffer {
    slots: Vec<PpoTransition>,
    capacity: usize,
    write_at: usize, // next slot to write; wraps at capacity
    count: usize,    // live items; saturates at capacity
}

impl SimpleBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: vec![PpoTransition::zero(); capacity],
            capacity,
            write_at: 0,
            count: 0,
        }
    }

    /// Push one transition. Overwrites oldest when full.
    pub fn write(&mut self, t: PpoTransition) {
        println!(
            "write: slot={} count={}/{} overwrite={}",
            self.write_at,
            self.count,
            self.capacity,
            self.count == self.capacity
        );
        self.slots[self.write_at] = t;
        self.write_at = (self.write_at + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Read up to n transitions, starting from the oldest live slot.
    pub fn read_batch(&self, n: usize) -> Vec<PpoTransition> {
        let take = n.min(self.count);
        // when full, write_at points at the oldest slot
        let start = if self.count == self.capacity {
            self.write_at
        } else {
            0
        };
        (0..take)
            .map(|i| self.slots[(start + i) % self.capacity])
            .collect()
    }

    pub fn len(&self) -> usize {
        self.count
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }
}

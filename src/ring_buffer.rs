use crate::types::StateTransition;

pub struct RingBuffer {
    slots: Vec<StateTransition>,
    capacity: usize,
    head: usize,
    tail: usize,
    count: usize,

    total_pushes: u64,    // successful writes
    total_pops: u64,      // successful reads
    rejected_pushes: u64, // push() calls that hit a full buffer
    empty_pops: u64,      // pop() calls that hit an empty buffer
    peak_count: usize,    // highest `count` ever observed
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: vec![StateTransition::zero(); capacity],
            capacity,
            head: 0,
            tail: 0,
            count: 0,
            total_pushes: 0,
            total_pops: 0,
            rejected_pushes: 0,
            empty_pops: 0,
            peak_count: 0,
        }
    }

    pub fn push(&mut self, item: StateTransition) -> Result<(), &'static str> {
        if self.is_full() {
            self.rejected_pushes += 1;
            return Err("buffer full");
        }

        self.slots[self.head] = item;
        self.head = (self.head + 1) % self.capacity;
        self.count += 1;
        self.total_pushes += 1;
        if self.count > self.peak_count {
            self.peak_count = self.count;
        }

        self.log_metrics();

        Ok(())
    }

    pub fn pop(&mut self) -> Option<StateTransition> {
        if self.is_empty() {
            self.empty_pops += 1;
            return None;
        }

        let item = self.slots[self.tail];
        self.tail = (self.tail + 1) % self.capacity;
        self.count -= 1;
        self.total_pops += 1;

        self.log_metrics();

        Some(item)
    }

    pub fn len(&self) -> usize {
        self.count
    }
    pub fn get_capacity(&self) -> usize {
        self.capacity
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }

    pub fn log_metrics(&self) {
        let fill_pct = 100.0 * self.count as f32 / self.capacity as f32;
        println!(
            "in rust [ring_buffer] fill={}/{} ({:.1}%)  peak={}  pushes={} (rejected={})  pops={} (empty={})",
            self.count,
            self.capacity,
            fill_pct,
            self.peak_count,
            self.total_pushes,
            self.rejected_pushes,
            self.total_pops,
            self.empty_pops,
        );
    }
}

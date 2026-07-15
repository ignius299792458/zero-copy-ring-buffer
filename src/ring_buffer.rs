use crate::types::StateTransition;

pub struct RingBuffer {
    slots: Vec<StateTransition>,
    capacity: usize,
    head: usize,
    tail: usize,
    count: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: vec![StateTransition::zero(); capacity],
            capacity,
            head: 0,
            tail: 0,
            count: 0,
        }
    }

    pub fn push(&mut self, item: StateTransition) -> Result<(), &'static str> {
        if self.is_full() {
            return Err("buffer full");
        }
        self.slots[self.head] = item;
        self.head = (self.head + 1) % self.capacity;
        self.count += 1;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<StateTransition> {
        if self.is_empty() {
            return None;
        }
        let item = self.slots[self.tail];
        self.tail = (self.tail + 1) % self.capacity;
        self.count -= 1;
        Some(item)
    }

    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }
}

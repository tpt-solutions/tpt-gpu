//! Command Batching - Amortize submission overhead for small operations.
use tptr_core::command::{Command, QueueHandle, QueuePriority, CommandQueue};
use tptr_core::error::TptrResult;

/// A batch of commands to be submitted atomically.
#[derive(Debug)]
pub struct CommandBatch {
    commands: Vec<(Command, QueuePriority)>,
    queue: QueueHandle,
}

impl CommandBatch {
    pub fn new(queue: QueueHandle) -> Self {
        Self { commands: Vec::with_capacity(64), queue }
    }
    pub fn push(&mut self, command: Command, priority: QueuePriority) -> &mut Self {
        self.commands.push((command, priority));
        self
    }
    pub fn barrier(&mut self) -> &mut Self {
        self.commands.push((Command::Barrier, QueuePriority::High));
        self
    }
    pub fn submit(&mut self, queue: &mut CommandQueue) -> TptrResult<usize> {
        let mut count = 0;
        for (cmd, pri) in self.commands.drain(..) {
            queue.submit(cmd, pri)?;
            count += 1;
        }
        Ok(count)
    }
    pub fn len(&self) -> usize { self.commands.len() }
    pub fn is_empty(&self) -> bool { self.commands.is_empty() }
    pub fn clear(&mut self) { self.commands.clear(); }
}


/// High-performance batch submitter with auto-submit threshold.
#[derive(Debug)]
pub struct BatchSubmitter {
    batch: CommandBatch,
    auto_submit_threshold: usize,
    total_submitted: u64,
}

impl BatchSubmitter {
    pub fn new(queue: QueueHandle) -> Self {
        Self { batch: CommandBatch::new(queue), auto_submit_threshold: 32, total_submitted: 0 }
    }
    pub fn with_threshold(queue: QueueHandle, threshold: usize) -> Self {
        Self { batch: CommandBatch::new(queue), auto_submit_threshold: threshold, total_submitted: 0 }
    }
    pub fn push(&mut self, command: Command, priority: QueuePriority, queue: &mut CommandQueue) -> TptrResult<u64> {
        self.batch.push(command, priority);
        if self.batch.len() >= self.auto_submit_threshold {
            self.flush(queue)?;
        }
        Ok(self.total_submitted)
    }
    pub fn flush(&mut self, queue: &mut CommandQueue) -> TptrResult<usize> {
        if self.batch.is_empty() { return Ok(0); }
        let count = self.batch.submit(queue)?;
        self.total_submitted += count as u64;
        Ok(count)
    }
    pub fn total_submitted(&self) -> u64 { self.total_submitted }
    pub fn set_threshold(&mut self, threshold: usize) { self.auto_submit_threshold = threshold; }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tptr_core::command::CommandQueue;
    #[test]
    fn test_batch_new() {
        let qh = QueueHandle(1);
        let batch = CommandBatch::new(qh);
        assert!(batch.is_empty());
    }
    #[test]
    fn test_batch_submit() {
        let qh = QueueHandle(1);
        let mut queue = CommandQueue::new(qh, 128);
        let mut batch = CommandBatch::new(qh);
        batch.push(Command::Barrier, QueuePriority::Normal);
        batch.push(Command::Barrier, QueuePriority::High);
        let count = batch.submit(&mut queue).unwrap();
        assert_eq!(count, 2);
    }
    #[test]
    fn test_batch_submitter_auto() {
        let qh = QueueHandle(1);
        let mut queue = CommandQueue::new(qh, 128);
        let mut submitter = BatchSubmitter::with_threshold(qh, 4);
        for _ in 0..3 { submitter.push(Command::Barrier, QueuePriority::Normal, &mut queue).unwrap(); }
        assert_eq!(queue.len(), 0);
        submitter.push(Command::Barrier, QueuePriority::Normal, &mut queue).unwrap();
        assert_eq!(queue.len(), 4);
    }
}

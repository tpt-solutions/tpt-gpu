# Tutorial 7: Kernel Scheduling

**Estimated Time:** 45 minutes  
**Prerequisites:** Tutorial 6

---

## Introduction

This tutorial covers kernel launch configuration, command queues, priority scheduling, and synchronization events.

### Scheduling Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  High Queue │  │ Normal Queue│  │  Low Queue  │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         └────────────────┼────────────────┘                    │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Command Scheduler                           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Command Queues

```rust
pub enum QueuePriority {
    High,    // Time-critical operations
    Normal,  // Default priority
    Low,     // Background operations
}

pub struct CommandQueue {
    priority: QueuePriority,
    commands: Vec<Command>,
    device: Arc<Device>,
}
```

### Creating Queues

```rust
let high_queue = device.create_queue(QueuePriority::High)?;
let normal_queue = device.create_queue(QueuePriority::Normal)?;
let low_queue = device.create_queue(QueuePriority::Low)?;
```

---

## Command Types

```rust
pub enum Command {
    Allocate { size: usize, flags: MemoryFlags },
    Free { ptr: *mut u8 },
    MemcpyH2D { dst: *mut u8, src: *const u8, size: usize },
    MemcpyD2H { dst: *mut u8, src: *const u8, size: usize },
    LaunchKernel { kernel: KernelHandle, config: KernelConfig },
    Barrier,
    Event { id: u64 },
}
```

---

## Priority Scheduling

```rust
pub struct CommandScheduler {
    high_queue: VecDeque<Command>,
    normal_queue: VecDeque<Command>,
    low_queue: VecDeque<Command>,
    aging_counters: HashMap<QueuePriority, u64>,
}

impl CommandScheduler {
    pub fn next_command(&mut self) -> Option<Command> {
        // Priority with aging to prevent starvation
        if !self.high_queue.is_empty() {
            return self.high_queue.pop_front();
        }
        
        // Age normal queue
        self.aging_counters.entry(QueuePriority::Normal)
            .and_modify(|c| *c += 1)
            .or_insert(1);
        
        // Boost to high priority after 100 cycles
        if self.aging_counters[&QueuePriority::Normal] > 100 {
            self.aging_counters.insert(QueuePriority::Normal, 0);
            return self.normal_queue.pop_front();
        }
        
        self.normal_queue.pop_front()
            .or_else(|| self.low_queue.pop_front())
    }
}
```

---

## Synchronization Events

```rust
// Create event
let event = device.create_event()?;

// Wait for event with timeout
event.wait(Duration::from_secs(5))?;

// Record event after command execution
queue.record_event(&event)?;

// Wait for multiple events
Event::wait_all(&[&event1, &event2])?;
```

---

## Cross-Queue Synchronization

```rust
let compute_event = device.create_queue(QueuePriority::Normal)?;
let transfer_event = device.create_queue(QueuePriority::Normal)?;

// Start compute queue
compute_queue.launch_kernel(&kernel, &config);
compute_queue.record_event(&compute_event)?;

// Transfer queue waits for compute
transfer_queue.wait_for_event(&compute_event)?;
transfer_queue.launch_kernel(&transfer_kernel, &config);

// Wait for both to complete
compute_event.wait(timeout)?;
transfer_event.wait(timeout)?;
```

---

## Example: Multi-Queue Execution

```rust
fn parallel_workloads(device: &Device) -> Result<()> {
    let compute_queue = device.create_queue(QueuePriority::High)?;
    let transfer_queue = device.create_queue(QueuePriority::Normal)?;
    
    let a = device.allocate(4 * 1024 * 1024)?;
    let b = device.allocate(4 * 1024 * 1024)?;
    let c = device.allocate(4 * 1024 * 1024)?;
    
    let kernel = device.create_kernel("matmul")?;
    let config = KernelConfig::new()
        .grid(64, 1, 1)
        .block(256, 1, 1);
    compute_queue.launch_kernel(&kernel, &config, &[&a, &b, &c])?;
    
    let host_buf = device.allocate_host(4 * 1024 * 1024)?;
    transfer_queue.memcpy_d2h(&host_buf, &c, 4 * 1024 * 1024)?;
    
    compute_queue.synchronize()?;
    transfer_queue.synchronize()?;
    
    Ok(())
}
```

---

## Exercises

1. **Priority Experiment**: Measure performance impact of different queue priorities
2. **Event Synchronization**: Implement producer-consumer pattern with events
3. **Multi-Queue**: Design a pipeline using multiple priority queues

---

## Summary

- ✅ Priority queues: High, Normal, Low
- ✅ Aging-based priority boosting to prevent starvation
- ✅ Event-based synchronization between queues
- ✅ Cross-queue synchronization patterns

**Next:** [Tutorial 8: GPU Primitives](08_gpu_primitives.md)

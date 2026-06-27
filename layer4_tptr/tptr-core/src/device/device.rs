//! Device Abstraction Implementation - Unified interface over GPU backends.
use crate::error::TptrResult;
use crate::memory::{MemoryAllocation, MemoryRegion, MemType, MemAccess, GpuAllocator, BuddyAllocator, AllocatorStats};
use crate::command::{CommandScheduler, Command, QueuePriority, QueueHandle};
use crate::kernel::{Kernel, KernelConfig, KernelHandle, Dim3};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend { TPTNative, CUDA, ROCm, Metal, Simulated }

impl Backend {
    pub fn name(&self) -> &'static str {
        match self { Self::TPTNative => "TPT Native", Self::CUDA => "CUDA", Self::ROCm => "ROCm", Self::Metal => "Metal", Self::Simulated => "Simulated" }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String, pub total_memory: u64,
    pub compute_capability: (u32, u32), pub max_threads_per_block: u32,
    pub max_block_dim: Dim3, pub max_grid_dim: Dim3,
    pub shared_memory_per_block: u32, pub warp_size: u32,
    pub num_compute_units: u32, pub memory_clock_khz: u64, pub backend: Backend,
}

impl DeviceProperties {
    pub fn simulated(name: &str, total_memory: u64) -> Self {
        Self { name: name.to_string(), total_memory, compute_capability: (1, 0),
            max_threads_per_block: 1024, max_block_dim: Dim3::new(1024, 1024, 64),
            max_grid_dim: Dim3::new(1 << 30, 1 << 16, 1 << 16),
            shared_memory_per_block: 48 * 1024, warp_size: 32, num_compute_units: 16,
            memory_clock_khz: 0, backend: Backend::Simulated }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceHandle(pub u64);

#[derive(Debug, Clone)]
pub struct DeviceInfo { pub handle: DeviceHandle, pub properties: DeviceProperties }

pub struct Device {
    handle: DeviceHandle, info: DeviceInfo,
    allocator: Box<dyn GpuAllocator>,
    scheduler: CommandScheduler,
    next_kernel_id: AtomicU64,
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device").field("handle", &self.handle).field("info", &self.info).finish()
    }
}

impl Device {
    pub fn new_simulated(id: u64, props: DeviceProperties) -> Self {
        let handle = DeviceHandle(id);
        let allocator: Box<dyn GpuAllocator> = Box::new(BuddyAllocator::new(0x1000_0000_0000, props.total_memory, 4096));
        let info = DeviceInfo { handle, properties: props.clone() };
        Self { handle, info, allocator, scheduler: CommandScheduler::new(), next_kernel_id: AtomicU64::new(1) }
    }
    pub fn handle(&self) -> DeviceHandle { self.handle }
    pub fn info(&self) -> &DeviceInfo { &self.info }
    pub fn properties(&self) -> &DeviceProperties { &self.info.properties }
    pub fn allocate(&mut self, size: u64, region: MemoryRegion, mem_type: MemType, access: MemAccess) -> TptrResult<MemoryAllocation> {
        self.allocator.allocate(size, region, mem_type, access)
    }
    pub fn free(&mut self, allocation: &MemoryAllocation) -> TptrResult<()> { self.allocator.free(allocation) }
    pub fn allocator_stats(&self) -> AllocatorStats { self.allocator.stats() }
    pub fn memcpy_htod(&self, _dst: &MemoryAllocation, _src: &[u8], _size: u64, _dst_offset: u64) -> TptrResult<()> { Ok(()) }
    pub fn memcpy_dtoh(&self, _dst: &mut [u8], _src: &MemoryAllocation, _size: u64, _src_offset: u64) -> TptrResult<()> { Ok(()) }
    pub fn create_queue(&mut self, _priority: QueuePriority, capacity: usize) -> QueueHandle { self.scheduler.create_queue(capacity) }
    pub fn submit(&mut self, queue: QueueHandle, command: Command, priority: QueuePriority) -> TptrResult<u64> { self.scheduler.submit(queue, command, priority) }
    pub fn scheduler_mut(&mut self) -> &mut CommandScheduler { &mut self.scheduler }
    pub fn create_kernel(&self, name: &str) -> Kernel { Kernel::new(name) }
    pub fn launch_kernel(&self, _kernel: &Kernel, _config: &KernelConfig, _args: &[Vec<u8>]) -> KernelHandle {
        let id = self.next_kernel_id.fetch_add(1, Ordering::SeqCst);
        let handle = KernelHandle::new(id); handle.set_running(); handle.set_completed(); handle
    }
    pub fn synchronize(&self) { }
    pub fn pending_commands(&self) -> usize { self.scheduler.total_pending() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryRegion;
    fn test_device() -> Device { Device::new_simulated(0, DeviceProperties::simulated("TPT Sim v1", 16 << 30)) }
    #[test] fn test_creation() { let d = test_device(); assert_eq!(d.handle(), DeviceHandle(0)); assert_eq!(d.properties().name, "TPT Sim v1"); }
    #[test] fn test_alloc_free() { let mut d = test_device(); let m = d.allocate(4096, MemoryRegion::Global, MemType::Device, MemAccess::ReadWrite).unwrap(); assert!(m.device_ptr() > 0); d.free(&m).unwrap(); assert!(m.is_freed()); }
    #[test] fn test_kernel_launch() { let d = test_device(); let kernel = d.create_kernel("test"); let config = KernelConfig::new((1, 1, 1), (256, 1, 1)); let handle = d.launch_kernel(&kernel, &config, &[]); assert!(handle.is_complete()); }
    #[test] fn test_queues() { let mut d = test_device(); let qh = d.create_queue(QueuePriority::Normal, 64); let id = d.submit(qh, Command::Barrier, QueuePriority::Normal).unwrap(); assert_eq!(id, 1); assert_eq!(d.pending_commands(), 1); }
    #[test] fn test_properties() { let d = test_device(); let p = d.properties(); assert_eq!(p.backend, Backend::Simulated); assert_eq!(p.warp_size, 32); }
}

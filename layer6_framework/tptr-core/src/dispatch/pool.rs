//! Memory Pool - Efficient reuse of GPU allocations for framework integration.
use tptr_core::memory::{MemoryAllocation, MemoryRegion, MemType, MemAccess, Alignment};
use tptr_core::error::{TptrResult, TptrError, ErrorCode};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

/// Statistics for memory pool usage.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub total_allocations: u64,
    pub pool_hits: u64,
    pub pool_misses: u64,
    pub bytes_served: u64,
    pub current_pool_usage: u64,
    pub peak_pool_usage: u64,
}

impl PoolStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.pool_hits + self.pool_misses;
        if total == 0 { 0.0 } else { self.pool_hits as f64 / total as f64 }
    }
}

/// A memory pool that caches freed allocations for reuse.
#[derive(Debug)]
pub struct MemoryPool {
    free_list: VecDeque<MemoryAllocation>,
    block_size: u64,
    region: MemoryRegion,
    mem_type: MemType,
    access: MemAccess,
    stats: PoolStats,
    next_handle: AtomicU64,
    max_blocks: usize,
}

impl MemoryPool {
    pub fn new(block_size: u64, max_blocks: usize, region: MemoryRegion, mem_type: MemType, access: MemAccess) -> Self {
        Self { free_list: VecDeque::with_capacity(max_blocks), block_size, region, mem_type, access, stats: PoolStats::default(), next_handle: AtomicU64::new(1), max_blocks }
    }
    pub fn acquire(&mut self) -> TptrResult<MemoryAllocation> {
        if let Some(alloc) = self.free_list.pop_front() {
            self.stats.pool_hits += 1;
            self.stats.bytes_served += alloc.size();
            Ok(alloc)
        } else {
            self.stats.pool_misses += 1;
            let handle = self.next_handle.fetch_add(1, Ordering::SeqCst);
            let alloc = MemoryAllocation::new(handle, self.block_size, self.region, self.mem_type, self.access, 0x1000_0000_0000 + handle * self.block_size, Alignment::DEFAULT);
            self.stats.total_allocations += 1;
            self.stats.bytes_served += self.block_size;
            self.stats.current_pool_usage += self.block_size;
            self.stats.peak_pool_usage = self.stats.peak_pool_usage.max(self.stats.current_pool_usage);
/// Pool manager for multiple block sizes.
#[derive(Debug)]
pub struct PoolManager {
    pools: Vec<MemoryPool>,
}

impl PoolManager {
    pub fn new() -> Self { Self { pools: Vec::new() } }
    pub fn add_pool(&mut self, block_size: u64, max_blocks: usize, region: MemoryRegion, mem_type: MemType, access: MemAccess) {
        self.pools.push(MemoryPool::new(block_size, max_blocks, region, mem_type, access));
    }
    pub fn find_pool(&mut self, size: u64) -> Option<&mut MemoryPool> {
        self.pools.iter_mut().find(|p| p.block_size >= size)
    }
    pub fn acquire(&mut self, size: u64) -> TptrResult<MemoryAllocation> {
        if let Some(pool) = self.find_pool(size) { pool.acquire() }
        else { Err(TptrError::new(ErrorCode::OutOfMemory, format!("No pool for size {}", size))) }
    }
    pub fn release(&mut self, alloc: MemoryAllocation) {
        let size = alloc.size();
        if let Some(pool) = self.pools.iter_mut().find(|p| p.block_size == size) { pool.release(alloc); }
    }
    pub fn stats(&self) -> PoolStats {
        let mut agg = PoolStats::default();
        for pool in &self.pools { let s = pool.stats(); agg.total_allocations += s.total_allocations; agg.pool_hits += s.pool_hits; agg.pool_misses += s.pool_misses; agg.bytes_served += s.bytes_served; agg.current_pool_usage += s.current_pool_usage; agg.peak_pool_usage = agg.peak_pool_usage.max(s.peak_pool_usage); }
        agg
    }
}

impl Default for PoolManager {
    fn default() -> Self {
        let mut mgr = Self::new();
        mgr.add_pool(256, 1024, MemoryRegion::Global, MemType::Device, MemAccess::ReadWrite);
        mgr.add_pool(4096, 512, MemoryRegion::Global, MemType::Device, MemAccess::ReadWrite);
        mgr.add_pool(65536, 128, MemoryRegion::Global, MemType::Device, MemAccess::ReadWrite);
        mgr.add_pool(1 << 20, 32, MemoryRegion::Global, MemType::Device, MemAccess::ReadWrite);
        mgr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_pool_acquire() {
        let mut pool = MemoryPool::new(4096, 16, MemoryRegion::Global, MemType::Device, MemAccess::ReadWrite);
        let alloc = pool.acquire().unwrap();
        assert_eq!(alloc.size(), 4096);
    }
    #[test]
    fn test_pool_manager_default() {
        let mut mgr = PoolManager::default();
        let alloc = mgr.acquire(4096).unwrap();
        assert_eq!(alloc.size(), 4096);
        mgr.release(alloc);
    }
}

            Ok(alloc)
        }
    }
    pub fn release(&mut self, alloc: MemoryAllocation) {
        if self.free_list.len() < self.max_blocks && !alloc.is_freed() {
            self.free_list.push_back(alloc);
        }
        self.stats.current_pool_usage = self.stats.current_pool_usage.saturating_sub(self.block_size);
    }
    pub fn stats(&self) -> PoolStats { self.stats.clone() }
    pub fn available(&self) -> usize { self.free_list.len() }
    pub fn capacity(&self) -> usize { self.max_blocks }
    pub fn clear(&mut self) { self.free_list.clear(); self.stats.current_pool_usage = 0; }
}


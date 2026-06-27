//! # TPT Framework Dispatch / tptr-dispatch
//!
//! Performance-critical dispatch paths for framework integration.
//! Provides command batching, memory pooling, and operation dispatch.

pub mod dispatch;

// Re-export core types
pub use dispatch::{
    CommandBatch, BatchSubmitter,
    MemoryPool, PoolStats, PoolManager,
    DispatchTable, OpHandle, DispatchError, OpType, OpMetadata, Operation,
};

/// Initialize the dispatch system with default configuration.
pub fn init_dispatch() -> DispatchTable {
    DispatchTable::default()
}

/// Create a default pool manager with standard block sizes.
pub fn create_pool_manager() -> PoolManager {
    PoolManager::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_dispatch() {
        let table = init_dispatch();
        assert!(table.len() >= 6);
    }

    #[test]
    fn test_create_pool_manager() {
        let mgr = create_pool_manager();
        let stats = mgr.stats();
        assert_eq!(stats.total_allocations, 0);
    }
}


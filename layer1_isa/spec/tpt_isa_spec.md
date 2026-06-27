# TPT ISA Specification v1.0

**Tensor Processing Technology — Instruction Set Architecture**

**Version:** 1.0  
**Status:** Draft  
**License:** Apache License 2.0 (with Express Patent Grant)

---

## 1. Architecture Overview

The TPT ISA defines a 32-bit, load-store, register-based architecture designed for general-purpose GPU compute. It supports SIMT (Single Instruction, Multiple Threads) execution with explicit tensor acceleration units.

### 1.1 Key Design Goals

- **32-bit fixed-length instructions** — simple decode, no alignment issues
- **Large register file** (256 × 32-bit scalar + 64 × 512-bit vector registers) — enables warp-level computation without spilling
- **Unified memory addressing** — 48-bit virtual address space (256 TiB)
- **Explicit memory hierarchy** — global, shared, local, and constant address spaces
- **SIMT execution model** — warps of 32 lanes (SIMD width 32)
- **Tensor acceleration** — native matrix multiply-accumulate (MMA) operations

### 1.2 Execution Model

The TPT core executes in a SIMT fashion:
- **Threads** are grouped into **warps** of 32 lanes
- **Warps** are grouped into **thread blocks (CTAs)**
- All threads in a warp execute the same instruction (with predication for divergence)
- Hardware tracks warp state and schedules warps onto compute units

### 1.3 Memory Model

| Address Space | Access Scope | Width | Description |
|---|---|---|---|
| Global | All threads | 48-bit | Main device memory |
| Shared | Single CTA | 32-bit | Low-latency shared memory |
| Local | Single thread | 32-bit | Thread-private stack/memory |
| Constant | All threads (read-only) | 48-bit | Read-only constant cache |

---

## 2. Instruction Formats

All instructions are 32 bits. There are 6 formats:

### 2.1 R-Type (Register — Register)

```
 31    27  26   22  21   17  16   12  11    7  6    0
┌────────┬──────┬──────┬──────┬──────┬────────────┐
│  opcode │  rd   │  rs1  │  rs2  │  func │  0000000  │
└────────┴──────┴──────┴──────┴──────┴────────────┘
```

- `opcode` (5 bits): Major opcode
- `rd` (5 bits): Destination register (0–31 scalar, or vector via mode)
- `rs1` (5 bits): Source register 1
- `rs2` (5 bits): Source register 2
- `func` (5 bits): Function selector within opcode group
- Reserved: 7 bits (zero)

### 2.2 I-Type (Immediate)

```
 31    27  26   22  21   17  16          5  4   0
┌────────┬──────┬──────┬─────────────────┬──────┐
│  opcode │  rd   │  rs1  │    immediate      │ func │
└────────┴──────┴──────┴─────────────────┴──────┘
```

- `immediate` (12 bits): Signed immediate value (sign-extended to 32 bits)

### 2.3 M-Type (Memory)

```
 31    27  26   22  21   17  16          5  4   0
┌────────┬──────┬──────┬─────────────────┬──────┐
│  opcode │  rd   │  rs1  │      offset      │ func │
└────────┴──────┴──────┴─────────────────┴──────┘
```

- `offset` (12 bits): Signed byte offset (sign-extended for address calculation)
- `rs1`: Base address register
- `rd`: Data register (load) or zero (store)

### 2.4 B-Type (Branch)

```
 31    27  26   22  21   17  16   12  11         0
┌────────┬──────┬──────┬──────┬──────────────────┐
│  opcode │  rs1  │  rs2  │ func │    branch_offset │
└────────┴──────┴──────┴──────┴──────────────────┘
```

- `branch_offset` (12 bits): Signed PC-relative offset in instructions (×4 byte addressing)

### 2.5 J-Type (Jump)

```
 31    27  26                    5  4   0
┌────────┬────────────────────────┬──────┐
│  opcode │     jump_target        │ func │
└────────┴────────────────────────┴──────┘
```

- `jump_target` (22 bits): Absolute or PC-relative target

### 2.6 V-Type (Vector / Tensor)

```
 31    27  26   22  21   17  16   12  11  10  9   5  4   0
┌────────┬──────┬──────┬──────┬───┬───┬───────┬──────┐
│  opcode │  vd   │  vs1  │  vs2  │ sz│ dm│  func  │ subop│
└────────┴──────┴──────┴──────┴───┴───┴───────┴──────┘
```

- `vd` (5 bits): Vector destination register (0–63)
- `vs1` (5 bits): Vector source register 1
- `vs2` (5 bits): Vector source register 2
- `sz` (2 bits): Data size (00=8b, 01=16b, 10=32b, 11=64b)
- `dm` (2 bits): Destination modifier (packed, scatter, mask)
- `func` (5 bits): Vector function selector
- `subop` (5 bits): Sub-operation

# Tutorial 10: PyTorch Integration

**Estimated Time:** 50 minutes  
**Prerequisites:** Tutorial 9, PyTorch basics

---

## Introduction

TPT GPU integrates with PyTorch through custom autograd functions and device dispatch, enabling seamless GPU acceleration for ML workloads.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PyTorch Application                           │
├─────────────────────────────────────────────────────────────────┤
│  import torch                                                    │
│  import tptr.pytorch as tpt                                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              TptrTorchDevice                               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │
│  │  │  Autograd   │  │   Tensor    │  │   Stream    │      │  │
│  │  │  Functions  │  │   Wrapper   │  │   Manager   │      │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    TPT Runtime (Layer 4)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
cd layer6_framework
pip install -e ".[dev]"
```

Verify:
```python
import torch
import tptr.pytorch as tpt
print(tpt.is_available())
```

---

## Device Management

```python
import torch
import tptr.pytorch as tpt

# Check availability
if tpt.is_available():
    device = tpt.device(0)
    print(f"Device: {device.name}")
    print(f"Memory: {device.total_memory / 1024**3:.1f} GB")

# Use with PyTorch
with tpt.device_context(0):
    tensor = torch.randn(1024, 1024, device='tpt')
```

---

## Tensor Operations

```python
import torch
import tptr.pytorch as tpt

# Create tensors on TPT device
a = torch.randn(1024, 512, device='tpt')
b = torch.randn(512, 768, device='tpt')

# Standard PyTorch operations work
c = torch.matmul(a, b)
d = torch.relu(c)
e = torch.softmax(d, dim=-1)

# Convert to CPU
cpu_tensor = e.cpu()
```

---

## Custom Autograd Functions

```python
import torch
import tptr.pytorch as tpt

class TptAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return tpt.add(a, b)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output, grad_output

class TptMatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return tpt.matmul(a, b)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = tpt.matmul(grad_output, b.t())
        grad_b = tpt.matmul(a.t(), grad_output)
        return grad_a, grad_b

# Usage
a = torch.randn(1024, 512, device='tpt', requires_grad=True)
b = torch.randn(512, 768, device='tpt', requires_grad=True)
c = TptMatmulFunction.apply(a, b)
c.sum().backward()
```

---

## Stream Management

```python
import tptr.pytorch.stream as tpt_stream

# Create stream
stream = tpt_stream.Stream(device=0, priority='high')

# Use stream for operations
with tpt_stream.StreamContext(stream):
    a = torch.randn(1024, 1024, device='tpt')
    b = torch.randn(1024, 1024, device='tpt')
    c = torch.matmul(a, b)

# Synchronize
stream.synchronize()

# Events for cross-stream sync
event = tpt_stream.Event()
stream.record_event(event)

other_stream = tpt_stream.Stream(device=0)
other_stream.wait_event(event)
```

---

## HuggingFace Integration

```python
from tptr.pytorch.hf_bridge import TptHFModel
from transformers import AutoModel

# Load model with TPT backend
model = AutoModel.from_pretrained("bert-base-uncased")
tpt_model = TptHFModel(model, device=0)

# Run inference
import torch
input_ids = torch.randint(0, 30000, (1, 128), device='tpt')
attention_mask = torch.ones(1, 128, device='tpt')

with torch.no_grad():
    output = tpt_model(input_ids, attention_mask)
    print(output.last_hidden_state.shape)
```

---

## Example: Training Loop

```python
import torch
import torch.nn as nn
import tptr.pytorch as tpt

class SimpleModel(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        return self.linear2(x)

def train():
    device = tpt.device(0)
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(10):
        # Generate dummy data
        x = torch.randn(32, 128, 512, device=device)
        y = torch.randn(32, 128, 512, device=device)
        
        # Forward
        pred = model(x)
        loss = nn.MSELoss()(pred, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
```

---

## Exercises

1. **Custom Op**: Implement a custom activation function with autograd support
2. **Mixed Precision**: Add AMP (Automatic Mixed Precision) support
3. **Multi-GPU**: Extend training to multiple TPT devices

---

## Summary

- ✅ Device management with PyTorch integration
- ✅ Tensor operations on TPT device
- ✅ Custom autograd functions for backward pass
- ✅ Stream and event management
- ✅ HuggingFace model integration

**Next:** [Tutorial 11: TPT Script Basics](11_tpt_script_basics.md)

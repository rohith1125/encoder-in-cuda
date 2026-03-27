"""
Minimal autograd engine for backward pass through the encoder.

Each operation records itself on a tape. Calling .backward() on the
final output replays the tape in reverse, computing gradients via
the chain rule.

In CUDA, the backward pass mirrors the forward:
  - Every forward kernel has a corresponding backward kernel
  - Gradients flow in reverse order through the same memory layout
  - Fused kernels in forward → fused gradient kernels in backward
"""

import numpy as np
from typing import Callable


class Tensor:
    """A NumPy array with autograd support.

    Tracks the computation graph so .backward() can compute gradients
    for all upstream parameters.
    """

    def __init__(
        self,
        data: np.ndarray,
        requires_grad: bool = False,
        _children: tuple = (),
        _op: str = "",
    ):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        self._backward: Callable = lambda: None
        self._children = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.shape}, grad={'yes' if self.requires_grad else 'no'}, op={self._op or 'leaf'})"

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication with gradient tracking.

        In CUDA backward: two GEMMs per matmul
          dA = dOut @ B^T
          dB = A^T @ dOut
        """
        out = Tensor(self.data @ other.data, _children=(self, other), _op="matmul")

        def _backward():
            if self.requires_grad and self.grad is not None:
                # dL/dA = dL/dOut @ B^T
                g = out.grad @ np.swapaxes(other.data, -2, -1)
                self.grad = self.grad + g
            if other.requires_grad and other.grad is not None:
                # dL/dB = A^T @ dL/dOut
                g = np.swapaxes(self.data, -2, -1) @ out.grad
                other.grad = other.grad + g

        out._backward = _backward
        out.requires_grad = True
        out.grad = np.zeros_like(out.data)
        return out

    def __add__(self, other: "Tensor") -> "Tensor":
        """Element-wise add with gradient passthrough.

        In CUDA: gradient is identity (just copy), possibly with broadcast reduction.
        """
        out = Tensor(self.data + other.data, _children=(self, other), _op="add")

        def _backward():
            if self.requires_grad and self.grad is not None:
                g = out.grad
                # Handle broadcasting: sum over broadcasted dims
                while g.ndim > self.data.ndim:
                    g = g.sum(axis=0)
                for i, (gs, ss) in enumerate(zip(g.shape, self.data.shape)):
                    if ss == 1 and gs != 1:
                        g = g.sum(axis=i, keepdims=True)
                self.grad = self.grad + g
            if isinstance(other, Tensor) and other.requires_grad and other.grad is not None:
                g = out.grad
                while g.ndim > other.data.ndim:
                    g = g.sum(axis=0)
                for i, (gs, ss) in enumerate(zip(g.shape, other.data.shape)):
                    if ss == 1 and gs != 1:
                        g = g.sum(axis=i, keepdims=True)
                other.grad = other.grad + g

        out._backward = _backward
        out.requires_grad = True
        out.grad = np.zeros_like(out.data)
        return out

    def __mul__(self, other: "Tensor") -> "Tensor":
        """Element-wise multiply with gradient.

        In CUDA: element-wise kernel, gradient is other * upstream.
        """
        out = Tensor(self.data * other.data, _children=(self, other), _op="mul")

        def _backward():
            if self.requires_grad and self.grad is not None:
                g = other.data * out.grad
                while g.ndim > self.data.ndim:
                    g = g.sum(axis=0)
                for i, (gs, ss) in enumerate(zip(g.shape, self.data.shape)):
                    if ss == 1 and gs != 1:
                        g = g.sum(axis=i, keepdims=True)
                self.grad = self.grad + g
            if isinstance(other, Tensor) and other.requires_grad and other.grad is not None:
                g = self.data * out.grad
                while g.ndim > other.data.ndim:
                    g = g.sum(axis=0)
                for i, (gs, ss) in enumerate(zip(g.shape, other.data.shape)):
                    if ss == 1 and gs != 1:
                        g = g.sum(axis=i, keepdims=True)
                other.grad = other.grad + g

        out._backward = _backward
        out.requires_grad = True
        out.grad = np.zeros_like(out.data)
        return out

    def sum(self) -> "Tensor":
        """Reduce sum — backward distributes gradient uniformly."""
        out = Tensor(np.array(self.data.sum()), _children=(self,), _op="sum")

        def _backward():
            if self.requires_grad and self.grad is not None:
                self.grad = self.grad + np.full_like(self.data, out.grad)

        out._backward = _backward
        out.requires_grad = True
        out.grad = np.zeros_like(out.data)
        return out

    def backward(self):
        """Reverse-mode autodiff — topological sort then backward pass.

        In CUDA: the backward graph is compiled ahead of time (e.g., in
        cuDNN/CUTLASS). Each kernel's backward is launched in reverse order.
        """
        topo = []
        visited = set()

        def _build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    _build_topo(child)
                topo.append(v)

        _build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

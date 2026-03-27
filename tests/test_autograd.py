"""Tests for the autograd engine."""

import numpy as np
import pytest

from encoder.autograd import Tensor


class TestTensorBasics:
    def test_creation(self):
        t = Tensor(np.array([1, 2, 3]), requires_grad=True)
        assert t.shape == (3,)
        assert t.grad is not None

    def test_no_grad_by_default(self):
        t = Tensor(np.array([1, 2, 3]))
        assert t.grad is None
        assert not t.requires_grad


class TestMatmul:
    def test_forward(self):
        a = Tensor(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b = Tensor(np.array([[5, 6], [7, 8]], dtype=np.float32))
        c = a @ b
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_equal(c.data, expected)

    def test_backward(self):
        a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        b = Tensor(np.random.randn(4, 5).astype(np.float32), requires_grad=True)
        c = a @ b
        loss = c.sum()
        loss.backward()

        # Numerical gradient check
        eps = 1e-4
        for i in range(3):
            for j in range(4):
                a_plus = a.data.copy()
                a_plus[i, j] += eps
                a_minus = a.data.copy()
                a_minus[i, j] -= eps
                numerical = ((a_plus @ b.data).sum() - (a_minus @ b.data).sum()) / (2 * eps)
                np.testing.assert_allclose(a.grad[i, j], numerical, rtol=1e-2, atol=1e-2)


class TestAdd:
    def test_forward(self):
        a = Tensor(np.array([1, 2, 3], dtype=np.float32))
        b = Tensor(np.array([4, 5, 6], dtype=np.float32))
        c = a + b
        np.testing.assert_array_equal(c.data, [5, 7, 9])

    def test_backward(self):
        a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        b = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        c = a + b
        loss = c.sum()
        loss.backward()
        # Gradient of sum w.r.t. addends is all ones
        np.testing.assert_allclose(a.grad, np.ones_like(a.data), atol=1e-6)
        np.testing.assert_allclose(b.grad, np.ones_like(b.data), atol=1e-6)

    def test_broadcast_backward(self):
        a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        b = Tensor(np.random.randn(4).astype(np.float32), requires_grad=True)
        c = a + b
        loss = c.sum()
        loss.backward()
        # b's gradient should be summed over the broadcast dimension
        np.testing.assert_allclose(b.grad, np.full(4, 3.0), atol=1e-6)


class TestMul:
    def test_forward(self):
        a = Tensor(np.array([2, 3], dtype=np.float32))
        b = Tensor(np.array([4, 5], dtype=np.float32))
        c = a * b
        np.testing.assert_array_equal(c.data, [8, 15])

    def test_backward(self):
        a = Tensor(np.array([2.0, 3.0], dtype=np.float32), requires_grad=True)
        b = Tensor(np.array([4.0, 5.0], dtype=np.float32), requires_grad=True)
        c = a * b
        loss = c.sum()
        loss.backward()
        np.testing.assert_allclose(a.grad, b.data, atol=1e-6)
        np.testing.assert_allclose(b.grad, a.data, atol=1e-6)


class TestChainedOps:
    def test_matmul_add_chain(self):
        """Test gradient flows through matmul -> add -> sum."""
        W = Tensor(np.random.randn(4, 3).astype(np.float32), requires_grad=True)
        x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
        b = Tensor(np.random.randn(3).astype(np.float32), requires_grad=True)

        y = (x @ W) + b
        loss = y.sum()
        loss.backward()

        assert W.grad is not None
        assert x.grad is not None
        assert b.grad is not None
        assert not np.any(np.isnan(W.grad))
        assert not np.any(np.isnan(x.grad))

    def test_numerical_gradient_check(self):
        """Full numerical gradient verification for a small network."""
        np.random.seed(42)
        W1 = Tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=True)
        W2 = Tensor(np.random.randn(8, 2).astype(np.float32), requires_grad=True)
        x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)

        # Forward: x @ W1 @ W2 -> sum
        h = x @ W1
        y = h @ W2
        loss = y.sum()
        loss.backward()

        # Check W1 gradient numerically
        eps = 1e-4
        for i in range(min(4, W1.shape[0])):
            for j in range(min(4, W1.shape[1])):
                w_plus = W1.data.copy()
                w_plus[i, j] += eps
                w_minus = W1.data.copy()
                w_minus[i, j] -= eps
                f_plus = (x.data @ w_plus @ W2.data).sum()
                f_minus = (x.data @ w_minus @ W2.data).sum()
                numerical = (f_plus - f_minus) / (2 * eps)
                np.testing.assert_allclose(
                    W1.grad[i, j], numerical, rtol=1e-2, atol=0.02,
                    err_msg=f"Gradient mismatch at W1[{i},{j}]"
                )

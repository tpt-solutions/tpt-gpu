"""Tests for the backend dispatch router."""

import pytest

from tptf.backend import TptBackend, get_backend, set_backend, dispatch


class TestBackendSwitch:
    def test_default_is_pytorch(self):
        set_backend(TptBackend.PYTORCH)
        assert get_backend() == TptBackend.PYTORCH

    def test_switch_to_jax(self):
        set_backend(TptBackend.JAX)
        assert get_backend() == TptBackend.JAX
        set_backend(TptBackend.PYTORCH)  # restore

    def test_switch_via_string(self):
        set_backend("jax")
        assert get_backend() == TptBackend.JAX
        set_backend("pytorch")
        assert get_backend() == TptBackend.PYTORCH

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError):
            set_backend("nonexistent")


class TestRuntimeBridge:
    def test_sim_mode_gemm(self):
        import numpy as np
        from tptf.runtime_bridge import TptRuntime
        rt = TptRuntime(sim=True)
        a = np.eye(3, dtype=np.float32)
        b = np.full((3, 3), 2.0, dtype=np.float32)
        out = rt.launch_gemm(a, b)
        np.testing.assert_allclose(out, np.full((3, 3), 2.0), atol=1e-5)

    def test_sim_mode_attention(self):
        import numpy as np
        from tptf.runtime_bridge import TptRuntime
        rt = TptRuntime(sim=True)
        q = np.ones((1, 1, 2, 4), dtype=np.float32)
        k = np.ones((1, 1, 2, 4), dtype=np.float32)
        v = np.ones((1, 1, 2, 4), dtype=np.float32)
        out = rt.launch_attention(q, k, v)
        assert out.shape == (1, 1, 2, 4)

    def test_sim_mode_alloc(self):
        import numpy as np
        from tptf.runtime_bridge import TptRuntime
        rt = TptRuntime(sim=True)
        buf = rt.alloc(64)
        assert buf is not None

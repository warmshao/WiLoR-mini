import pytest
import torch

from wilor_mini.models.vit import rot6d_to_rotmat


def _identity_rot6d(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    )


def _assert_identity_rotation(out: torch.Tensor, device: torch.device, dtype: torch.dtype) -> None:
    expected = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0)
    assert out.shape == (1, 3, 3)
    assert out.device.type == device.type
    assert out.dtype == dtype
    assert torch.isfinite(out.float()).all()
    assert torch.allclose(out.float(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rot6d_to_rotmat_preserves_cpu_dtype_device_and_shape(dtype: torch.dtype) -> None:
    device = torch.device("cpu")
    out = rot6d_to_rotmat(_identity_rot6d(device, dtype))
    _assert_identity_rotation(out, device, dtype)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rot6d_to_rotmat_preserves_mps_dtype_device_and_shape(dtype: torch.dtype) -> None:
    device = torch.device("mps")
    out = rot6d_to_rotmat(_identity_rot6d(device, dtype))
    torch.mps.synchronize()
    _assert_identity_rotation(out, device, dtype)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_rot6d_to_rotmat_supports_mps_bfloat16_without_cross_kernel() -> None:
    device = torch.device("mps")
    out = rot6d_to_rotmat(_identity_rot6d(device, torch.bfloat16))
    torch.mps.synchronize()
    _assert_identity_rotation(out, device, torch.bfloat16)


def test_rot6d_to_rotmat_manual_cross_remains_differentiable_on_cpu() -> None:
    x = _identity_rot6d(torch.device("cpu"), torch.float32).requires_grad_(True)
    out = rot6d_to_rotmat(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

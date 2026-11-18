import numpy as np
import pytest

from mosaic.aligned_array import create_aligned_array, create_aligned_array_like, align_array


@pytest.mark.parametrize("alignment", [32, 64, 1024])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_alignment_from_unaligned_buffer(monkeypatch, alignment, dtype):
    original_empty = np.empty

    def unaligned_empty(size, dtype):
        allocation = original_empty(size + alignment, dtype=dtype)
        # Force a remainder for which the old positive-offset formula fails.
        start = (8 - allocation.ctypes.data) % alignment
        return allocation[start : start + size]

    monkeypatch.setattr(np, "empty", unaligned_empty)
    source = np.arange(6, dtype=dtype).reshape(2, 3)
    for result in (
        create_aligned_array(source.shape, dtype, alignment),
        create_aligned_array_like(source, alignment),
        align_array(source, alignment),
    ):
        assert result.ctypes.data % alignment == 0
        assert result.shape == source.shape
        assert result.dtype == source.dtype
    np.testing.assert_array_equal(align_array(source, alignment), source)


def test_already_aligned_array_is_reused():
    source = create_aligned_array((4,), np.float32, 64)
    assert align_array(source, 64) is source

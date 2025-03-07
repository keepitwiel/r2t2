import pytest
import numpy as np
import taichi as ti

from r2t2.taichi_renderer import TaichiRenderer


ti.init(arch=ti.cpu)

def test_full_initialization_power_of_two():
    # Test with a height map that has power of 2 dimensions
    height_map = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=np.float32)
    
    renderer = TaichiRenderer(height_map)
    maxmipmap, n_levels = renderer.update_maxmipmap()
    
    # Check number of levels
    assert n_levels == 2
    
    # Convert maxmipmap to numpy for testing
    mipmap_np = maxmipmap.to_numpy()
    
    # Test level 0 (2x2 blocks)
    np.testing.assert_equal(mipmap_np[0, 0], np.max(height_map[0:2, 0:2]))
    np.testing.assert_equal(mipmap_np[1, 0], np.max(height_map[2:4, 0:2]))
    np.testing.assert_equal(mipmap_np[0, 1], np.max(height_map[0:2, 2:4]))
    np.testing.assert_equal(mipmap_np[1, 1], np.max(height_map[2:4, 2:4]))
    
    # Test level 1 (4x4 block)
    np.testing.assert_equal(mipmap_np[0, 2], np.max(height_map))


def test_non_power_of_two_dimensions():
    # Test with a height map that doesn't have power of 2 dimensions
    height_map = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float32)
    
    renderer = TaichiRenderer(height_map)
    maxmipmap, n_levels = renderer.update_maxmipmap()
    
    # Should round up to next power of 2 (4x4)
    assert n_levels == 2
    
    # Convert maxmipmap to numpy for testing
    mipmap_np = maxmipmap.to_numpy()
    
    # Test level 0 (2x2 blocks)
    # For the first three blocks, we test against the actual data
    np.testing.assert_equal(mipmap_np[0, 0], np.max(height_map[0:2, 0:2]))  # Top-left block
    np.testing.assert_equal(mipmap_np[1, 0], np.max(height_map[2:3, 0:2]))  # Bottom-left block
    np.testing.assert_equal(mipmap_np[0, 1], np.max(height_map[0:2, 2:3]))  # Top-right block
    
    # For the bottom-right block, we expect the max of the available values
    # This includes the value at position [2,2] which is 9
    np.testing.assert_equal(mipmap_np[1, 1], height_map[2, 2])  # Bottom-right block
    
    # Test level 1 (4x4 block)
    # The max should be 9 as it's the highest value in the original height map
    np.testing.assert_equal(mipmap_np[0, 2], np.max(height_map))

def test_partial_update():
    height_map = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=np.float32)
    
    renderer = TaichiRenderer(height_map)
    # Initialize the maxmipmap first
    renderer.update_maxmipmap()
    
    # Update a 2x2 region in the middle
    original_mipmap = renderer.maxmipmap.to_numpy().copy()
    
    # Modify height map in the region
    height_map[1:3, 1:3] = np.array([[20, 21], [22, 23]])
    renderer.height_map.from_numpy(height_map)
    
    # Update the maxmipmap for the modified region
    renderer.update_maxmipmap(x=1, y=1, w=2, h=2)
    updated_mipmap = renderer.maxmipmap.to_numpy()
    
    # Test that only the affected regions were updated
    # Level 0: The 2x2 region should be updated
    np.testing.assert_equal(updated_mipmap[0, 0], original_mipmap[0, 0])  # Unchanged
    np.testing.assert_equal(updated_mipmap[1, 1], np.max(height_map[2:4, 2:4]))  # Changed
    
    # Level 1: The entire 4x4 region should be updated
    np.testing.assert_equal(updated_mipmap[0, 2], np.max(height_map))

def test_invalid_update_region():
    height_map = np.zeros((4, 4), dtype=np.float32)
    renderer = TaichiRenderer(height_map)
    # Initialize the maxmipmap first
    renderer.update_maxmipmap()
    
    # Test negative coordinates
    with pytest.raises(ValueError):
        renderer.update_maxmipmap(x=-1, y=0, w=2, h=2)
    
    # Test zero width/height
    with pytest.raises(ValueError):
        renderer.update_maxmipmap(x=0, y=0, w=0, h=2)
    
    # Test region exceeding map boundaries
    with pytest.raises(ValueError):
        renderer.update_maxmipmap(x=3, y=3, w=2, h=2)

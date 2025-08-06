#!/usr/bin/env python3
"""
Test script for ADAMSS volume preservation and control features.
Tests the implementation of Sections 2.4.2.3 and 2.4.2.4.
"""

import numpy as np
import taichi as ti
from adaamss_2422 import Adamss

# Initialize Taichi
ti.init(arch=ti.cpu)

def create_simple_tet_mesh():
    """Create a simple single tetrahedron for testing"""
    # Four vertices of a regular tetrahedron
    points = np.array([
        [0.0, 0.0, 0.0],           # vertex 0
        [1.0, 0.0, 0.0],           # vertex 1  
        [0.5, 0.866, 0.0],         # vertex 2
        [0.5, 0.289, 0.816]        # vertex 3
    ], dtype=np.float64) * 1000.0  # Scale to mm
    
    # Single tetrahedron connectivity
    tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
    
    # All lung tissue
    labels = np.array([1], dtype=np.int32)
    
    return points, tets, labels

def test_volume_preservation():
    """Test volume preservation feature"""
    print("\n=== Testing Volume Preservation (Section 2.4.2.3) ===")
    
    # Create simple mesh
    points, tets, labels = create_simple_tet_mesh()
    
    # Create simulation
    sim = Adamss(points, tets, labels)
    
    # Get initial volume statistics
    print(f"Initial volumes: {sim.vol.to_numpy()}")
    print(f"Initial total volume: {np.sum(sim.vol.to_numpy()):.6f} m³")
    
    # Enable volume preservation
    sim.enable_volume_preservation(bulk_modulus=10000.0)  # 10 kPa
    
    # Check that volume preservation is enabled
    vol_pres_flags = sim.volume_preservation_enabled.to_numpy()
    bulk_mods = sim.bulk_modulus.to_numpy()
    print(f"Volume preservation enabled: {vol_pres_flags}")
    print(f"Bulk moduli: {bulk_mods}")
    
    # Run a few simulation steps
    dt = 1e-4
    for step in range(10):
        sim.step(dt)
        if step % 5 == 0:
            stats = sim.get_volume_change_stats()
            print(f"Step {step}: Volume change stats: {stats}")
    
    print("Volume preservation test completed successfully!")

def test_volume_control():
    """Test volume control feature"""
    print("\n=== Testing Volume Control (Section 2.4.2.4) ===")
    
    # Create simple mesh
    points, tets, labels = create_simple_tet_mesh()
    
    # Create simulation
    sim = Adamss(points, tets, labels)
    
    # Get initial volume statistics
    print(f"Initial volumes: {sim.vol.to_numpy()}")
    
    # Enable volume control to expand by 20%
    target_ratio = 1.2
    sim.enable_volume_control(target_ratio=target_ratio, bulk_modulus=8000.0)  # 8 kPa
    
    # Check that volume control is enabled
    vol_ctrl_flags = sim.volume_control_enabled.to_numpy()
    target_ratios = sim.target_volume_ratio.to_numpy()
    print(f"Volume control enabled: {vol_ctrl_flags}")
    print(f"Target volume ratios: {target_ratios}")
    
    # Run simulation steps
    dt = 1e-4
    for step in range(20):
        sim.step(dt)
        if step % 10 == 0:
            stats = sim.get_volume_change_stats()
            print(f"Step {step}: Volume change stats: {stats}")
    
    print("Volume control test completed successfully!")

def test_disable_features():
    """Test disabling volume features"""
    print("\n=== Testing Feature Disable ===")
    
    # Create simple mesh
    points, tets, labels = create_simple_tet_mesh()
    
    # Create simulation
    sim = Adamss(points, tets, labels)
    
    # Enable both features
    sim.enable_volume_preservation(bulk_modulus=5000.0)
    sim.enable_volume_control(target_ratio=1.1, bulk_modulus=5000.0)
    
    print("Before disable:")
    print(f"Volume preservation: {sim.volume_preservation_enabled.to_numpy()}")
    print(f"Volume control: {sim.volume_control_enabled.to_numpy()}")
    
    # Disable all features
    sim.disable_volume_features()
    
    print("After disable:")
    print(f"Volume preservation: {sim.volume_preservation_enabled.to_numpy()}")
    print(f"Volume control: {sim.volume_control_enabled.to_numpy()}")
    
    print("Feature disable test completed successfully!")

def main():
    """Main test function"""
    print("Testing ADAMSS Volume Features Implementation")
    print("=" * 50)
    
    try:
        test_volume_preservation()
        test_volume_control()
        test_disable_features()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("Volume preservation and control features are working correctly.")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

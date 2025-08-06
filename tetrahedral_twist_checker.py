"""
Tetrahedral Twist Angle Checker

This script checks whether tetrahedral axes hardly ever twist away from their rest angle.
The check computes |δθ| = |θ_current - θ_rest| << 1 for each tetrahedral axis.

This implementation follows the lung deformation analysis approach:
- Loads preprocessed data with initial and current configurations
- Computes twist angles for all 3 anisotropy axes per tetrahedron
- Provides statistics and validation of twist constraints
"""

import numpy as np
import taichi as ti
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ti.init(arch=ti.cuda)

@ti.data_oriented
class TetrahedralTwistChecker:
    def __init__(self, preprocessed_data_path):
        """Initialize twist checker with preprocessed data"""
        # Load preprocessed data
        data = np.load(preprocessed_data_path, allow_pickle=True)
        
        self.N = data['mesh_points'].shape[0]  # #nodes
        self.M = data['tetrahedra'].shape[0]   # #tets
        
        print(f"Loaded data: {self.N} nodes, {self.M} tetrahedra")
        
        # Taichi fields for tetrahedral data
        self.tets = ti.Vector.field(4, ti.i32, shape=self.M)
        self.current_positions = ti.Vector.field(3, ti.f64, shape=self.N)
        self.initial_positions = ti.Vector.field(3, ti.f64, shape=self.N)
        
        # Anisotropy axes (3 per tetrahedron)
        self.current_axes = ti.Vector.field(3, ti.f64, shape=(self.M, 3))
        self.initial_axes = ti.Vector.field(3, ti.f64, shape=(self.M, 3))
        
        # Twist angle results
        self.twist_angles = ti.field(ti.f64, shape=(self.M, 3))      # δθ per axis
        self.rest_angles = ti.field(ti.f64, shape=(self.M, 3))       # θ_rest per axis
        self.current_angles = ti.field(ti.f64, shape=(self.M, 3))    # θ_current per axis
        self.twist_magnitude = ti.field(ti.f64, shape=(self.M, 3))   # |δθ| per axis
        
        # Statistics and flags
        self.is_valid_tet = ti.field(ti.i32, shape=self.M)           # valid tetrahedron flag
        self.max_twist_per_tet = ti.field(ti.f64, shape=self.M)      # max twist per tet
        
        # Load data into Taichi fields
        self._load_data(data)
        
        # Compute current anisotropy axes if needed
        self._compute_current_axes()
        
        # Initialize checker fields
        self._initialize_checker_fields()
        
    def _load_data(self, data):
        """Load data from preprocessed file into Taichi fields"""
        self.tets.from_numpy(data['tetrahedra'].astype(np.int32))
        
        # Current positions = initial + displacement
        initial_pos = data['initial_positions'].astype(np.float64)
        displacement = data['displacement_field'].astype(np.float64)
        current_pos = initial_pos + displacement
        
        self.current_positions.from_numpy(current_pos)
        self.initial_positions.from_numpy(initial_pos)
        
        # Load anisotropy axes
        self.current_axes.from_numpy(data['anisotropy_axes'].astype(np.float64))
        
        # If we have initial axis vectors, use them, otherwise compute from initial positions
        if 'initial_axis_vectors' in data:
            self.initial_axes.from_numpy(data['initial_axis_vectors'].astype(np.float64))
        else:
            # Use current axes as initial (assumes no deformation yet)
            self.initial_axes.from_numpy(data['anisotropy_axes'].astype(np.float64))
    
    @ti.kernel
    def _compute_current_axes(self):
        """Compute current anisotropy axes from deformed mesh configuration"""
        eps_zero = 1e-10
        
        for k in range(self.M):
            # Get deformed vertex positions
            v0 = self.current_positions[self.tets[k][0]]
            v1 = self.current_positions[self.tets[k][1]]
            v2 = self.current_positions[self.tets[k][2]]
            v3 = self.current_positions[self.tets[k][3]]
            
            # Compute current edge matrix (deformed configuration)
            edge1 = v1 - v0
            edge2 = v2 - v0
            edge3 = v3 - v0
            
            # Form edge matrix d_x = [edge1, edge2, edge3]
            d_x = ti.Matrix([[edge1[0], edge2[0], edge3[0]],
                            [edge1[1], edge2[1], edge3[1]],
                            [edge1[2], edge2[2], edge3[2]]])
            
            # Get reference edge matrix
            iv0 = self.initial_positions[self.tets[k][0]]
            iv1 = self.initial_positions[self.tets[k][1]]
            iv2 = self.initial_positions[self.tets[k][2]]
            iv3 = self.initial_positions[self.tets[k][3]]
            
            i_edge1 = iv1 - iv0
            i_edge2 = iv2 - iv0
            i_edge3 = iv3 - iv0
            
            D_m = ti.Matrix([[i_edge1[0], i_edge2[0], i_edge3[0]],
                            [i_edge1[1], i_edge2[1], i_edge3[1]],
                            [i_edge1[2], i_edge2[2], i_edge3[2]]])
            
            # Compute deformation gradient F = d_x * D_m^(-1)
            det_D_m = D_m.determinant()
            
            if ti.abs(det_D_m) > eps_zero:
                D_m_inv = D_m.inverse()
                F = d_x @ D_m_inv
                
                # SVD of deformation gradient: F = U * Sigma * V^T
                U, sigma, V = ti.svd(F)
                
                # Current anisotropy axes from V matrix (principal directions)
                # V columns are the principal stretch directions in reference config
                # U columns are the principal directions in current config
                self.current_axes[k, 0] = ti.Vector([U[0, 0], U[1, 0], U[2, 0]])
                self.current_axes[k, 1] = ti.Vector([U[0, 1], U[1, 1], U[2, 1]])
                self.current_axes[k, 2] = ti.Vector([U[0, 2], U[1, 2], U[2, 2]])
            else:
                # Degenerate case - keep current axes unchanged
                pass
    
    @ti.kernel
    def _initialize_checker_fields(self):
        """Initialize twist checker fields"""
        for k in range(self.M):
            self.is_valid_tet[k] = 1  # Assume valid unless proven otherwise
            self.max_twist_per_tet[k] = 0.0
            
            for axis_idx in ti.static(range(3)):
                self.twist_angles[k, axis_idx] = 0.0
                self.rest_angles[k, axis_idx] = 0.0
                self.current_angles[k, axis_idx] = 0.0
                self.twist_magnitude[k, axis_idx] = 0.0
    
    @ti.kernel
    def _compute_twist_angles(self):
        """Compute twist angles |δθ| = |θ_current - θ_rest| for each axis"""
        eps_axis = 1e-10
        
        for k in range(self.M):
            # Check if tetrahedron is valid (non-degenerate)
            v0 = self.current_positions[self.tets[k][0]]
            v1 = self.current_positions[self.tets[k][1]]
            v2 = self.current_positions[self.tets[k][2]]
            v3 = self.current_positions[self.tets[k][3]]
            
            # Compute volume to check for degeneracy
            vol = ti.abs((v1 - v0).dot((v2 - v0).cross(v3 - v0))) / 6.0
            
            if vol > 1e-12:
                max_twist = 0.0
                
                for axis_idx in ti.static(range(3)):
                    # Get initial and current axis vectors
                    initial_axis = self.initial_axes[k, axis_idx]
                    current_axis = self.current_axes[k, axis_idx]
                    
                    # Normalize axes
                    initial_norm = initial_axis.norm()
                    current_norm = current_axis.norm()
                    
                    if initial_norm > eps_axis and current_norm > eps_axis:
                        init_normalized = initial_axis / initial_norm
                        curr_normalized = current_axis / current_norm
                        
                        # Compute angle between axes using dot product
                        # cos(θ) = a·b / (|a||b|)
                        cos_angle = init_normalized.dot(curr_normalized)
                        
                        # Clamp to avoid numerical issues with acos
                        cos_angle = ti.max(-1.0, ti.min(1.0, cos_angle))
                        
                        # Compute twist angle
                        twist_angle = ti.acos(ti.abs(cos_angle))  # Use abs to get minimum angle
                        
                        # Store results
                        self.rest_angles[k, axis_idx] = 0.0  # Reference angle is 0
                        self.current_angles[k, axis_idx] = twist_angle
                        self.twist_angles[k, axis_idx] = twist_angle  # δθ = θ_current - θ_rest
                        self.twist_magnitude[k, axis_idx] = ti.abs(twist_angle)
                        
                        # Track maximum twist per tetrahedron
                        if ti.abs(twist_angle) > max_twist:
                            max_twist = ti.abs(twist_angle)
                    else:
                        # Invalid axis - mark as such
                        self.twist_magnitude[k, axis_idx] = -1.0  # Invalid marker
                
                self.max_twist_per_tet[k] = max_twist
            else:
                # Degenerate tetrahedron
                self.is_valid_tet[k] = 0
                for axis_idx in ti.static(range(3)):
                    self.twist_magnitude[k, axis_idx] = -1.0  # Invalid marker
    
    def check_twist_constraint(self, tolerance=0.1):
        """
        Check if tetrahedral axes satisfy twist constraint |δθ| << 1
        
        Args:
            tolerance: Maximum allowed twist angle in radians (default 0.1 rad ≈ 5.7°)
        
        Returns:
            dict: Statistics about twist constraint satisfaction
        """
        print(f"\n=== Tetrahedral Twist Angle Analysis ===")
        
        # Compute twist angles
        self._compute_twist_angles()
        
        # Extract results to numpy for analysis
        twist_magnitudes = self.twist_magnitude.to_numpy()
        valid_tets = self.is_valid_tet.to_numpy()
        max_twists = self.max_twist_per_tet.to_numpy()
        
        # Filter out invalid measurements
        valid_mask = (twist_magnitudes >= 0) & (valid_tets[:, None] == 1)
        valid_twists = twist_magnitudes[valid_mask]
        
        if len(valid_twists) == 0:
            print("No valid twist measurements found!")
            return {}
        
        # Statistics
        total_axes = np.sum(valid_mask)
        total_tets = np.sum(valid_tets)
        
        # Count violations
        violations = valid_twists > tolerance
        n_violations = np.sum(violations)
        violation_rate = n_violations / len(valid_twists) if len(valid_twists) > 0 else 0
        
        # Per-tetrahedron statistics
        valid_max_twists = max_twists[valid_tets == 1]
        tet_violations = valid_max_twists > tolerance
        n_tet_violations = np.sum(tet_violations)
        tet_violation_rate = n_tet_violations / len(valid_max_twists) if len(valid_max_twists) > 0 else 0
        
        # Statistical measures
        stats = {
            'total_tetrahedra': self.M,
            'valid_tetrahedra': total_tets,
            'total_axes': total_axes,
            'tolerance_rad': tolerance,
            'tolerance_deg': np.degrees(tolerance),
            
            # Twist angle statistics
            'mean_twist_rad': np.mean(valid_twists),
            'mean_twist_deg': np.degrees(np.mean(valid_twists)),
            'max_twist_rad': np.max(valid_twists),
            'max_twist_deg': np.degrees(np.max(valid_twists)),
            'std_twist_rad': np.std(valid_twists),
            'std_twist_deg': np.degrees(np.std(valid_twists)),
            
            # Constraint violations
            'axis_violations': n_violations,
            'axis_violation_rate': violation_rate,
            'tet_violations': n_tet_violations,
            'tet_violation_rate': tet_violation_rate,
            
            # Percentiles
            'p50_twist_deg': np.degrees(np.percentile(valid_twists, 50)),
            'p90_twist_deg': np.degrees(np.percentile(valid_twists, 90)),
            'p95_twist_deg': np.degrees(np.percentile(valid_twists, 95)),
            'p99_twist_deg': np.degrees(np.percentile(valid_twists, 99)),
        }
        
        # Print summary
        print(f"Total tetrahedra: {stats['total_tetrahedra']}")
        print(f"Valid tetrahedra: {stats['valid_tetrahedra']} ({100*stats['valid_tetrahedra']/stats['total_tetrahedra']:.1f}%)")
        print(f"Total axis measurements: {stats['total_axes']}")
        print(f"\nTolerance: {stats['tolerance_deg']:.1f}° ({stats['tolerance_rad']:.3f} rad)")
        
        print(f"\nTwist Angle Statistics:")
        print(f"  Mean twist: {stats['mean_twist_deg']:.2f}° ({stats['mean_twist_rad']:.4f} rad)")
        print(f"  Max twist: {stats['max_twist_deg']:.2f}° ({stats['max_twist_rad']:.4f} rad)")
        print(f"  Std deviation: {stats['std_twist_deg']:.2f}° ({stats['std_twist_rad']:.4f} rad)")
        
        print(f"\nPercentiles:")
        print(f"  50th percentile: {stats['p50_twist_deg']:.2f}°")
        print(f"  90th percentile: {stats['p90_twist_deg']:.2f}°")
        print(f"  95th percentile: {stats['p95_twist_deg']:.2f}°")
        print(f"  99th percentile: {stats['p99_twist_deg']:.2f}°")
        
        print(f"\nConstraint Violations:")
        print(f"  Axis violations: {stats['axis_violations']}/{stats['total_axes']} ({100*stats['axis_violation_rate']:.1f}%)")
        print(f"  Tetrahedron violations: {stats['tet_violations']}/{stats['valid_tetrahedra']} ({100*stats['tet_violation_rate']:.1f}%)")
        
        # Assessment
        if stats['axis_violation_rate'] < 0.05:  # Less than 5% violations
            assessment = "EXCELLENT - Twist constraint well satisfied"
        elif stats['axis_violation_rate'] < 0.10:  # Less than 10% violations
            assessment = "GOOD - Twist constraint mostly satisfied"
        elif stats['axis_violation_rate'] < 0.25:  # Less than 25% violations
            assessment = "ACCEPTABLE - Some twist constraint violations"
        else:
            assessment = "POOR - Many twist constraint violations"
        
        print(f"\nAssessment: {assessment}")
        
        return stats
    
    def visualize_tetrahedron(self, tet_idx, output_path=None):
        """Visualize original and deformed tetrahedron with anisotropy axes"""
        
        # Get tetrahedron vertex indices
        tet_vertices = self.tets.to_numpy()[tet_idx]
        
        # Get original and current positions
        original_pos = self.initial_positions.to_numpy()[tet_vertices]
        current_pos = self.current_positions.to_numpy()[tet_vertices]
        
        # Get original and current axes
        original_axes = self.initial_axes.to_numpy()[tet_idx]
        current_axes = self.current_axes.to_numpy()[tet_idx]
        
        # Compute centroids
        original_centroid = np.mean(original_pos, axis=0)
        current_centroid = np.mean(current_pos, axis=0)
        
        # Create 3D plot with single subplot for comparison
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot tetrahedron edges and setup
        edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        axis_colors = ['red', 'green', 'orange']
        axis_labels = ['Axis 1', 'Axis 2', 'Axis 3']
        axis_scale = np.linalg.norm(original_pos.max(axis=0) - original_pos.min(axis=0)) * 0.3  # Adaptive scale
        
        # Plot original tetrahedron (blue, solid)
        for edge in edges:
            pts = original_pos[list(edge)]
            ax.plot3D(*pts.T, 'b-', linewidth=3, alpha=0.8, label='Original' if edge == (0,1) else "")
        
        # Plot original vertices
        ax.scatter(*original_pos.T, c='blue', s=120, alpha=0.9, marker='o', edgecolors='black')
        for i, pos in enumerate(original_pos):
            ax.text(pos[0], pos[1], pos[2], f'  O{i}', fontsize=9, color='blue', weight='bold')
        
        # Plot deformed tetrahedron (red, dashed)
        for edge in edges:
            pts = current_pos[list(edge)]
            ax.plot3D(*pts.T, 'r--', linewidth=3, alpha=0.8, label='Deformed' if edge == (0,1) else "")
        
        # Plot deformed vertices
        ax.scatter(*current_pos.T, c='red', s=120, alpha=0.9, marker='s', edgecolors='black')
        for i, pos in enumerate(current_pos):
            ax.text(pos[0], pos[1], pos[2], f'  D{i}', fontsize=9, color='red', weight='bold')
        
        # Plot original anisotropy axes (solid arrows)
        for i, (axis, color, label) in enumerate(zip(original_axes, axis_colors, axis_labels)):
            ax.quiver(original_centroid[0], original_centroid[1], original_centroid[2],
                     axis[0] * axis_scale, axis[1] * axis_scale, axis[2] * axis_scale,
                     color=color, arrow_length_ratio=0.15, linewidth=3, alpha=0.8,
                     label=f'{label} (Orig)')
        
        # Plot deformed anisotropy axes (dashed-style, offset slightly)
        offset = np.array([axis_scale * 0.1, 0, 0])  # Small offset to avoid overlap
        deformed_centroid_offset = current_centroid + offset
        for i, (axis, color, label) in enumerate(zip(current_axes, axis_colors, axis_labels)):
            ax.quiver(deformed_centroid_offset[0], deformed_centroid_offset[1], deformed_centroid_offset[2],
                     axis[0] * axis_scale, axis[1] * axis_scale, axis[2] * axis_scale,
                     color=color, arrow_length_ratio=0.15, linewidth=2, alpha=0.6, linestyle='--',
                     label=f'{label} (Def)')
        
        # Draw lines connecting corresponding vertices
        for i in range(4):
            ax.plot3D([original_pos[i,0], current_pos[i,0]], 
                     [original_pos[i,1], current_pos[i,1]], 
                     [original_pos[i,2], current_pos[i,2]], 
                     'gray', alpha=0.5, linewidth=1)
        
        ax.set_title(f'Tetrahedron {tet_idx} Comparison\nOriginal (Blue/Solid) vs Deformed (Red/Dashed)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set equal aspect ratio and good viewing angle
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        # Print twist angle information for this tetrahedron
        twist_magnitudes = self.twist_magnitude.to_numpy()[tet_idx]
        print(f"\nTetrahedron {tet_idx} Twist Angles:")
        for i, twist in enumerate(twist_magnitudes):
            if twist >= 0:
                print(f"  Axis {i+1}: {np.degrees(twist):.2f}° ({twist:.4f} rad)")
            else:
                print(f"  Axis {i+1}: Invalid")
        
        # Print vertex positions
        print(f"\nOriginal vertices (mm):")
        for i, pos in enumerate(original_pos):
            print(f"  Vertex {i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        print(f"\nDeformed vertices (mm):")
        for i, pos in enumerate(current_pos):
            print(f"  Vertex {i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        print(f"\nOriginal centroid: [{original_centroid[0]:.3f}, {original_centroid[1]:.3f}, {original_centroid[2]:.3f}]")
        print(f"Deformed centroid: [{current_centroid[0]:.3f}, {current_centroid[1]:.3f}, {current_centroid[2]:.3f}]")
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        else:
            plt.show()
    
    


def main():
    """Main function"""
    # Configuration
    data_path = "data_processed_deformation/Case10Pack/Case10Pack_T70_to_T80_deformation.npz"
    tolerance = 0.1  # radians
    output_prefix = "tetrahedral_twist_results"
    
    # Check if data file exists
    if not Path(data_path).exists():
        print(f"Error: Data file not found: {data_path}")
        return
    
    print(f"=== Tetrahedral Twist Angle Checker ===")
    print(f"Data file: {data_path}")
    print(f"Tolerance: {tolerance:.3f} rad ({np.degrees(tolerance):.1f}°)")
    
    # Create checker
    checker = TetrahedralTwistChecker(data_path)
    
    # Perform analysis
    stats = checker.check_twist_constraint(tolerance=tolerance)
    
    # Find a tetrahedron with high twist for visualization
    twist_magnitudes = checker.twist_magnitude.to_numpy()
    valid_mask = twist_magnitudes >= 0
    max_twists_per_tet = np.max(twist_magnitudes * valid_mask, axis=1)
    tet_to_visualize = np.argmax(max_twists_per_tet)  # Choose tetrahedron with highest twist
    print(f"\n=== Visualizing Tetrahedron {tet_to_visualize} ===")
    checker.visualize_tetrahedron(tet_to_visualize, f"{output_prefix}_tet_{tet_to_visualize}_visualization.png")

    print(f"\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
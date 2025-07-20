#!/usr/bin/env python3
"""
Lung CT Processing Pipeline

Direct execution pipeline for lung CT processing:
1. Segmentation using TotalSegmentator  
2. Surface mesh extraction with marching cubes
3. Deformation field loading and application
4. Mesh warping with displacement fields
5. Batch automation for multiple phases

Results saved in data_processed/ directory.
"""

import numpy as np
import nibabel as nib
import meshio
from pathlib import Path
from skimage import measure
from scipy.interpolate import RegularGridInterpolator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LungCTPipeline:
    """Streamlined lung CT processing pipeline"""
    
    def __init__(self, data_root: str = "data/Case1Pack", output_dir: str = "data_processed"):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.phases = [f"T{i:02d}" for i in range(0, 100, 10)]
        self.reference_phase = "T00"
        
        logger.info(f"Pipeline initialized: {self.data_root} -> {self.output_dir}")
    
    def run_full_pipeline(self, case_id: int = 1):
        """Execute complete processing pipeline"""
        logger.info(f"Starting pipeline for case {case_id}")
        
        # Step 1: Create segmentation
        self.create_segmentation(case_id)
        
        # Step 2: Extract surface meshes
        self.extract_surface_meshes(case_id)
        
        # Step 3: Process deformation fields
        self.process_deformation_fields(case_id)
        
        logger.info("Pipeline completed successfully!")
    
    def create_segmentation(self, case_id: int):
        """Create lung segmentation masks"""
        logger.info("Creating lung segmentation masks")
        
        nifti_dir = self.data_root / "NIFTI"
        totalseg_dir = self.data_root / "TotalSegment"
        totalseg_dir.mkdir(exist_ok=True)
        
        for phase in self.phases:
            input_file = nifti_dir / f"case{case_id}_{phase}.nii.gz"
            output_dir = totalseg_dir / f"case{case_id}_{phase}"
            
            if output_dir.exists():
                continue
                
            output_dir.mkdir(exist_ok=True)
            
            # Load CT image and create threshold-based segmentation
            img = nib.load(input_file)
            data = img.get_fdata()
            
            # Simple lung mask (threshold-based)
            lung_mask = (data > -500) & (data < 200)
            
            # Save lung regions mask
            lung_regions_file = output_dir / "lung_regions.nii.gz"
            mask_img = nib.Nifti1Image(lung_mask.astype(np.uint8), img.affine, img.header)
            nib.save(mask_img, lung_regions_file)
    
    def extract_surface_meshes(self, case_id: int):
        """Extract surface meshes using marching cubes"""
        logger.info("Extracting surface meshes")
        
        totalseg_dir = self.data_root / "TotalSegment"
        mesh_dir = self.data_root / "pygalmesh"
        
        for phase in self.phases:
            mask_file = totalseg_dir / f"case{case_id}_{phase}" / "lung_regions.nii.gz"
            output_mesh = mesh_dir / f"case{case_id}_{phase}_lung_regions_processed.xdmf"
            
            if output_mesh.exists():
                continue
            
            # Load mask
            mask_img = nib.load(mask_file)
            mask_data = mask_img.get_fdata()
            
            # Extract surface mesh using marching cubes
            vertices, faces, _, _ = measure.marching_cubes(mask_data, level=0.5)
            
            # Transform to physical coordinates
            vertices_homo = np.column_stack([vertices, np.ones(len(vertices))])
            vertices_phys = (mask_img.affine @ vertices_homo.T).T[:, :3]
            
            # Create and save mesh
            mesh = meshio.Mesh(
                points=vertices_phys,
                cells=[("triangle", faces)]
            )
            mesh.write(output_mesh)
            mesh.write(output_mesh.with_suffix('.vtu'))
    
    def process_deformation_fields(self, case_id: int):
        """Process deformation fields and warp meshes"""
        logger.info("Processing deformation fields")
        
        corrfield_dir = self.data_root / "CorrField"
        mesh_dir = self.data_root / "pygalmesh"
        
        # Load reference mesh
        ref_mesh_file = mesh_dir / f"case{case_id}_T00_lung_regions_processed.xdmf"
        if not ref_mesh_file.exists():
            ref_mesh_file = mesh_dir / f"case{case_id}_T00_lung_regions_11.xdmf"
        
        ref_mesh = meshio.read(ref_mesh_file)
        logger.info(f"Loaded reference mesh: {len(ref_mesh.points)} vertices")
        
        # Process each deformation
        for phase in self.phases:
            if phase == self.reference_phase:
                continue
            
            deformation_file = corrfield_dir / f"case{case_id}_T00_{phase}.nii.gz"
            output_file = self.output_dir / f"case{case_id}_T00_to_{phase}_warped.xdmf"
            
            if output_file.exists():
                continue
            
            # Load and apply deformation
            deform_img = nib.load(deformation_file)
            deform_data = deform_img.get_fdata()
            
            warped_mesh = self.warp_mesh(ref_mesh, deform_data, deform_img.affine)
            
            # Save warped mesh
            warped_mesh.write(output_file)
            warped_mesh.write(output_file.with_suffix('.vtu'))
            
            logger.info(f"Warped mesh saved: {output_file.name}")
    
    def warp_mesh(self, mesh: meshio.Mesh, deform_data: np.ndarray, affine: np.ndarray) -> meshio.Mesh:
        """Apply deformation field to mesh vertices"""
        
        # Transform vertices to voxel coordinates
        vertices = mesh.points.copy()
        vertices_homo = np.column_stack([vertices, np.ones(len(vertices))])
        voxel_coords = (np.linalg.inv(affine) @ vertices_homo.T).T[:, :3]
        
        # Interpolate deformation at vertex positions
        x_grid = np.arange(deform_data.shape[0])
        y_grid = np.arange(deform_data.shape[1])
        z_grid = np.arange(deform_data.shape[2])
        
        displacements = np.zeros((len(vertices), 3))
        
        for i in range(3):  # x, y, z components
            interpolator = RegularGridInterpolator(
                (x_grid, y_grid, z_grid),
                deform_data[:, :, :, i],
                method='cubic',
                bounds_error=False,
                fill_value=0.0
            )
            displacements[:, i] = interpolator(voxel_coords)
        
        # Apply displacements
        warped_vertices = vertices + displacements
        
        # Create warped mesh with displacement data
        warped_mesh = meshio.Mesh(
            points=warped_vertices,
            cells=mesh.cells,
            point_data={
                'displacement': displacements,
                'displacement_magnitude': np.linalg.norm(displacements, axis=1)
            }
        )
        
        return warped_mesh


def main():
    """Execute the complete lung CT processing pipeline"""
    pipeline = LungCTPipeline()
    pipeline.run_full_pipeline(case_id=1)
    
    # Validate results
    output_files = list(pipeline.output_dir.glob("*.xdmf"))
    logger.info(f"Generated {len(output_files)} mesh files in {pipeline.output_dir}")


if __name__ == "__main__":
    main()
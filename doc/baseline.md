# Lung CT Processing Pipeline

A pipeline for processing lung CT images, including segmentation, mesh extraction, deformable registration, and warping. The pipeline creates meshes from CT images, applies deformation fields to them, and saves the results for further analysis.

## Overview

This pipeline processes 4D lung CT data to:
1. Extract lung surface meshes from CT images
2. Apply deformation fields between different respiratory phases
3. Generate warped meshes showing lung deformation over the breathing cycle
4. Save results in multiple formats (XDMF, VTU, H5) for analysis

**Results are saved in `data_processed/` directory.**

## Data Structure

The pipeline expects data organized as follows:

```
data/Case1Pack/
├── NIFTI/                    # CT images
│   ├── case1_T00.nii.gz     # Reference phase
│   ├── case1_T10.nii.gz     # Target phases
│   └── ...
├── CorrField/                # Deformation fields
│   ├── case1_T00_T10.nii.gz # T00 to T10 deformation
│   ├── case1_T00_T20.nii.gz # T00 to T20 deformation
│   └── ...
├── TotalSegment/             # Segmentation output
│   └── case1_T00/
│       └── lung_regions.nii.gz
└── pygalmesh/               # Generated meshes
    ├── case1_T00_lung_regions_processed.xdmf
    └── ...
```

## Pipeline Steps

### 1. Segmentation
- Uses **TotalSegmentator** to generate binary lung masks (`.nii.gz` format)
- Fallback: Threshold-based segmentation if TotalSegmentator unavailable
- Creates `lung_regions.nii.gz` files for each phase

### 2. Surface Mesh Extraction
- **Optional step**: Extract surface-only meshes for adaamms(not finished now) 
- Since volumetric meshes already exist in `pygalmesh/`, load it directly
- (Temporary try)If surface-only mesh required: extract triangular faces from existing `.xdmf` volume mesh. Alternative: Apply marching cubes to binary masks for pure surface extraction

### 3. Deformation Field Processing
- Loads pre-computed deformation fields from `CorrField/`
- Uses scipy interpolation for smooth displacement calculation
- Supports cubic/bspline interpolation with extrapolation handling
- Note: Can load CSV files first if available for faster processing

### 4. Mesh Warping
- Applies displacement fields to reference mesh vertices
- Interpolates deformation field values at vertex positions
- Preserves mesh topology while deforming geometry
- Adds displacement magnitude as point data

### 5. Batch Processing
- Processes all phase combinations automatically
- Generates T00 → other phase transformations (T10, T20, T30, T40, T50, T60, T70, T80, T90)
- Saves results in multiple formats for compatibility



## Output Files

### Generated Mesh Files
```
data_processed/
├── case1_T00_to_T10_warped.xdmf  # Main mesh file
├── case1_T00_to_T10_warped.h5    # Binary data
├── case1_T00_to_T10_warped.vtu   # VTK format
└── ...                           # All T00 → other phase combinations
```

### File Formats
- **XDMF**: XML-based format with HDF5 data storage
- **VTU**: VTK unstructured grid format
- **H5**: HDF5 binary data files

### Mesh Properties
- **Vertices**: ~130,000 points per mesh
- **Triangles**: ~260,000 triangular faces
- **Point Data**: Displacement vectors and magnitudes
- **Units**: Millimeters (mm)

## Configuration

### Pipeline Configuration (`pipeline_config.yaml`)
```yaml
# Data paths
data:
  root_dir: "data/Case1Pack"
  output_dir: "data_processed"

# Processing parameters
mesh:
  marching_cubes:
    level: 0.5
    step_size: 1
    use_smoothing: true

deformation:
  interpolation_method: "cubic"
  extrapolation_value: 0.0
```

### Key Parameters
- **Marching cubes level**: Iso-surface threshold (default: 0.5)
- **Step size**: Controls mesh resolution (1 = full resolution)
- **Interpolation method**: "cubic", "linear", or "nearest"
- **Smoothing**: Optional Laplacian smoothing

## Validation

### Automatic Quality Checks
- **Mesh integrity**: Validates triangle connectivity
- **Displacement bounds**: Checks for reasonable deformation
- **File format**: Ensures XDMF/VTK compatibility
- **Physical coordinates**: Validates spatial consistency

### Validation
```bash
# Check mesh statistics
python -c "
import meshio
mesh = meshio.read('data_processed/case1_T00_to_T10_warped.xdmf')
print(f'Vertices: {len(mesh.points)}')
print(f'Triangles: {len(mesh.cells[0].data)}')
"
```

## Implementation Details

### Core Components

1. **`lung_ct_pipeline.py`**: Complete pipeline implementation
   - `LungCTPipeline`: Streamlined processing class
   - Direct execution with `python lung_ct_pipeline.py`
   - Handles all steps: segmentation, meshing, deformation

2. **`mesh_utils.py`**: Mesh processing utilities
   - Surface extraction with marching cubes
   - Mesh quality assessment and validation
   - Displacement field application

### Algorithm Details

#### Surface Mesh Extraction
```python
# Marching cubes with physical coordinate transformation
vertices, faces, _, _ = measure.marching_cubes(mask_data, level=0.5)
vertices_phys = (affine @ vertices_homo.T).T[:, :3]
```

#### Deformation Field Interpolation
```python
# 3D interpolation for displacement vectors
from scipy.interpolate import RegularGridInterpolator
interpolator = RegularGridInterpolator(
    (x_grid, y_grid, z_grid),
    deform_data[:, :, :, component],
    method='cubic'
)
```

#### Mesh Warping
```python
# Apply displacements to vertices
warped_vertices = original_vertices + interpolated_displacements
```

### Performance Characteristics
- **Processing time**: ~2-3 minutes per phase transformation
- **Memory usage**: ~2-4 GB peak for full resolution meshes
- **Disk space**: ~50 MB per warped mesh file set

## Results Summary

### Successful Generation
**9 T00 → other phase transformations** generated successfully:
- T00 → T10, T20, T30, T40, T50, T60, T70, T80, T90
- Each with ~130k vertices and ~260k triangles
- All saved in `data_processed/` directory

### Quality Metrics
- **Mesh validation**: All surface meshes pass quality checks
- **Displacement range**: Physiologically reasonable (< 30mm)
- **File integrity**: All XDMF/VTU files load correctly

## Project Files

### Main Scripts
- `lung_ct_pipeline.py` - Complete pipeline implementation (directly executable)
- `mesh_utils.py` - Mesh processing utilities
- `pipeline_config.yaml` - Configuration file

### Data Directories
- `data/Case1Pack/` - Input CT data and deformation fields
- `data_processed/` - Output warped meshes and results



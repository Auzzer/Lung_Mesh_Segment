
import meshio

import nibabel as nib
import numpy as np
from pathlib import Path
from skimage import measure
from scipy.interpolate import RegularGridInterpolator
import logging  
mesh_path = "/home/haozhe/Lung_Mesh_Segment/mesh_files/case1_T00_lung_regions_11.xdmf"
mesh = meshio.read(mesh_path)
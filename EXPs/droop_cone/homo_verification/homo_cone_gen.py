"""
Mesh generation part:
The following code was used to generate the tetrahedral cone mesh saved as
`cone.xdmf`. It's kept here as reference and intentionally disabled so the
simulation can run directly from the pre-generated mesh file.
"""

import numpy as np
import fenics as fe
import gmsh
from mpi4py import MPI
import meshio
"""
!!!!!! Very IMPORTANT !!!!!!!!!!!!!
dolfinx.io installation requires fenics-dolfinx package. If we 
want to use dolfinx, fenics-dolfinx will replace fenics package.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
L, Rb, Ra, hmax = 0.6, 0.20, 0.1, 0.03

gmsh.initialize()
gmsh.model.add("cone")
vol = gmsh.model.occ.addCone(0, 0, 0, 0, 0, L, Rb, Ra)
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(3, [vol], 1) # optional now. It's required for gmshio.model_to_mesh in from dolfinx.io import gmshio, XDMFFile
gmsh.option.setNumber("Mesh.MeshSizeMax", hmax)
gmsh.model.mesh.generate(3)
gmsh.write("cone.msh")
gmsh.finalize()

m = meshio.read("cone.msh")
tets = m.get_cells_type("tetra")
mesh = meshio.Mesh(points=m.points, cells=[("tetra", tets)])
meshio.write("cone.xdmf", mesh, data_format="XML") # ASCII XDMF

print("Wrote cone.xdmf")




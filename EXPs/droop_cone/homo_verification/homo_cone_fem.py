import numpy as np
import fenics as fe
import gmsh
from mpi4py import MPI
import meshio


"""
Solve linear elasticity (droop) on a cone with top face clamped.

Method: standard small-strain linear elasticity with body force (gravity),
as in typical FEM PDE formulations in https://github.com/mattragoza/lung-project/blob/master/project/pde.py.

"""


def solve_cone_droop(
    mesh_path: str = "cone.xdmf",
    E: float = 5e4,
    rho: float = 1000.0,
    g_vec=(0.0, 0.0, -9.81),
    top_tol: float = 1e-4,
    results_prefix: str = "cone_droop"
):
    """Solve small-strain linear elasticity with body force on the cone mesh.

    Clamps the top face (z ~ z_max) with u = 0, applies body force f = rho * g_vec,
    and writes displacement plus deformed geometry outputs.
    """

    # Read mesh from XDMF (ensure using communicator-aware API)
    mesh = fe.Mesh()
    with fe.XDMFFile(fe.MPI.comm_world, mesh_path) as xdmf:
        xdmf.read(mesh)

    # Function space
    V = fe.VectorFunctionSpace(mesh, "P", 1)
    nu = 0.4  # Poisson's ratio (unitless)
    # Material parameters (Lamé constants)
    mu = fe.Constant(E / (2.0 * (1.0 + nu)))
    lmbda = fe.Constant(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    # Gravity as body force
    f = fe.Constant(tuple(rho * gi for gi in g_vec))

    # Identify top boundary (max z) and clamp it
    coords = mesh.coordinates()
    z_max = float(coords[:, 2].max())

    class TopBoundary(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[2], z_max, top_tol)

    top_bd = TopBoundary()
    zero = fe.Constant((0.0, 0.0, 0.0))
    bc_top = fe.DirichletBC(V, zero, top_bd)

    # Variational problem: linear elasticity with body force
    u = fe.Function(V)  # solution
    du = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    d = u.geometric_dimension()
    I = fe.Identity(d)

    def epsilon(w):
        return fe.sym(fe.grad(w))

    def sigma(w):
        return 2.0 * mu * epsilon(w) + lmbda * fe.tr(epsilon(w)) * I

    a = fe.inner(sigma(du), epsilon(v)) * fe.dx
    L = fe.dot(f, v) * fe.dx

    fe.solve(a == L, u, bcs=[bc_top])

    # Export final vertex positions (original + displacement at vertices)
    u_vertex = u.compute_vertex_values(mesh).reshape((d, -1)).T

    # Results stats
    disp_mag = np.linalg.norm(u_vertex, axis=1)
    print(f"Solved droop: |u|_max={disp_mag.max():.6e}, |u|_mean={disp_mag.mean():.6e}")
    print(f"Top z identified at z_max={z_max:.6f}, clamped nodes count ~{(np.abs(coords[:,2]-z_max) < top_tol).sum()}")
    # Write a single ASCII XDMF of the deformed mesh using meshio
    final_vertices = coords + u_vertex
    base = meshio.read(mesh_path)
    try:
        tets = base.get_cells_type("tetra")
    except Exception:
        tets = None
        for c in base.cells:
            if getattr(c, "type", None) == "tetra":
                tets = c.data
                break
    if tets is None or len(tets) == 0:
        raise RuntimeError("No tetrahedral cells found in base XDMF.")
    out_mesh = meshio.Mesh(points=final_vertices, cells=[("tetra", tets)])
    meshio.write(f"{results_prefix}.xdmf", out_mesh, data_format="XML")
    print(f"Result written: {results_prefix}.xdmf (ASCII deformed mesh)")

    


if __name__ == "__main__":
    # Default run with moderately soft material to show visible droop
    solve_cone_droop(
        mesh_path="cone.xdmf",
        E=1e4, # Poisson's ratio (unitless)
        rho=200.0, # Density (kg/m^3)
        g_vec=(0.0, 0.0, -9.81),  # vertical droop (-z)
        top_tol=1e-4,
        results_prefix="cone_droop",
    )

import taichi as ti, torch
ti.init(arch=ti.cuda)

N, M, NNZ = 1024, 1024, 200000
rows = torch.empty(NNZ, device="cuda", dtype=torch.int64)
cols = torch.empty(NNZ, device="cuda", dtype=torch.int64)
vals = torch.empty(NNZ, device="cuda", dtype=torch.float32)

@ti.kernel
def fill_triplets(r: ti.types.ndarray(), c: ti.types.ndarray(), v: ti.types.ndarray()):
    for k in range(NNZ):
        # toy pattern; replace with your logic
        r[k] = k % N
        c[k] = (k * 7) % M
        v[k] = 1.0

fill_triplets(rows, cols, vals)
indices = torch.stack([rows, cols], dim=0)        # shape (2, NNZ)
A_torch = torch.sparse_coo_tensor(indices, vals, size=(N, M), device="cuda").coalesce()

trip = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=NNZ)  # each elem: [i, j, val]
@ti.kernel
def make_triplets(T: ti.types.ndarray()):
    for k in range(NNZ):
        T[k] = ti.Vector([k % N, (k * 7) % M, 1.0], dt=ti.f32)

make_triplets(trip)
A_ti = ti.linalg.SparseMatrix(n=N, m=M, dtype=ti.f32)
A_ti.build_from_ndarray(trip)

b = ti.ndarray(ti.f32, shape=N)
# If b originates in Torch, copy GPU→GPU via a Taichi kernel that reads the Torch tensor
solver = ti.linalg.SparseSolver(solver_type="LLT")
solver.analyze_pattern(A_ti); solver.factorize(A_ti)
x = solver.solve(b)

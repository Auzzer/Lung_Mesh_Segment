# Simple Mass-Spring Model using Liu et al. method with Taichi sparse matrices
# Uses f32 precision for CUDA compatibility
import os, json
import numpy as np
import taichi as ti

ti.init(arch=ti.cuda, default_fp=ti.f32)

EPS = 1e-12

def load_metadata(mdir):
    js = os.path.join(mdir, 'metadata.json')
    if os.path.exists(js):
        with open(js, 'r') as f: return json.load(f)
    # fallback to legacy npy (pickle)
    npy = os.path.join(mdir, 'metadata.npy')
    return np.load(npy, allow_pickle=True).item()

@ti.data_oriented
class SimpleMSM:
    """
    Liu et al. Local-Global solver using direction-locked parameterization
    
    Local step:  d_e <- r_e * (x_i - x_j)/||x_i - x_j||
    Global step: K_FF s_F = rhs_F, where K_s = P^T L P, then x = x' + a_i s_i
    
    Uses Taichi sparse matrices and SparseSolver for efficient GPU computation.
    """

    def __init__(self, matrices_dir: str, gravity_g: float = 0.0):
        meta = load_metadata(matrices_dir)
        self.N = int(meta['N'])
        self.E = int(meta['E'])

        # ----- load arrays (host) -----
        V  = np.load(os.path.join(matrices_dir, 'vertices.npy'))                # (N,3)
        Eij = np.load(os.path.join(matrices_dir, 'edges.npy')).astype(np.int32) # (E,2)
        r  = np.load(os.path.join(matrices_dir, 'rest_lengths.npy'))            # (E,)
        a  = np.load(os.path.join(matrices_dir, 'registration_directions.npy')) # (N,3), unit
        U  = np.load(os.path.join(matrices_dir, 'displacement_vectors.npy'))    # (N,3)
        ke = np.load(os.path.join(matrices_dir, 'edge_stiffness.npy'))          # (E,)

        assert V.shape==(self.N,3) and Eij.shape==(self.E,2)
        assert r.shape==(self.E,) and a.shape==(self.N,3) and ke.shape==(self.E,)

        # ----- fields (device) -----
        self.x_ref  = ti.Vector.field(3, ti.f32, shape=self.N)  # reference (also x')
        self.x      = ti.Vector.field(3, ti.f32, shape=self.N)  # current
        self.a      = ti.Vector.field(3, ti.f32, shape=self.N)  # registration dirs (unit)
        self.s      = ti.field(ti.f32, shape=self.N)            # scalar DOFs
        self.s_pres = ti.field(ti.f32, shape=self.N)            # prescribed s (for fixed)
        self.is_fix = ti.field(ti.i32, shape=self.N)            # 1 if fixed

        self.edges  = ti.Vector.field(2, ti.i32, shape=self.E)  # (i,j)
        self.r_e    = ti.field(ti.f32, shape=self.E)
        self.k_e    = ti.field(ti.f32, shape=self.E)
        self.d_e    = ti.Vector.field(3, ti.f32, shape=self.E)  # auxiliary per-edge

        self.f      = ti.Vector.field(3, ti.f32, shape=self.N)  # external load (constant)
        self.Lx_b   = ti.Vector.field(3, ti.f32, shape=self.N)  # L x' (vector form), constant
        self.bias_s = ti.field(ti.f32, shape=self.N)            # - a·(f + Lx'), constant

        # RHS workspace (node accumulators & scalar forms)
        self.accum_vec = ti.Vector.field(3, ti.f32, shape=self.N) # sum_i k_e * d_e contributions
        self.rhs_full  = ti.field(ti.f32, shape=self.N)           # a·(accum_vec) + bias - corr

        # Free/fixed mapping (global->compact)
        self.free_idx  = ti.field(ti.i32, shape=self.N)           # -1 if fixed, else [0..nF)
        self.fixed_idx = ti.field(ti.i32, shape=self.N)           # -1 if free, else [0..nD)

        # Correction from fixed DOFs: corr_free[i] = (K_FD s_D)_i in full indexing (only for free nodes)
        self.corr_full = ti.field(ti.f32, shape=self.N)

        # ----- copy host data -----
        self._copy_host(V, Eij, r, a, U, ke)

        # set gravity or other loads (constant)
        self._set_gravity(gravity_g)

        # boundary conditions from loaded BC data
        self._load_and_apply_bc(matrices_dir, U)

        # precompute constant pieces & assemble/factorize K_FF
        self._precompute_Lxprime_and_bias()   # bias_s = - a·(f + Lx')
        self._build_maps_free_fixed()         # free_idx / fixed_idx (+ arrays on host)
        self._build_and_factorize_KFF()       # scalar K_FF, LLT once
        self._precompute_KFD_sD_correction()  # corr_full (in full indexing)

        # init x and d
        self._update_positions()
        self._init_d_from_rest()

    # -------------------- data and BC --------------------
    def _copy_host(self, V, Eij, r, a, U, ke):
        for i in range(self.N):
            self.x_ref[i] = V[i]
            self.x[i]     = V[i]
            self.a[i]     = a[i]
            self.s[i]     = 0.0
            self.s_pres[i]= 0.0
            self.is_fix[i]= 0
            self.f[i]     = ti.Vector([0.0,0.0,0.0])
        for e in range(self.E):
            self.edges[e] = ti.Vector(Eij[e].tolist())
            self.r_e[e]   = r[e]
            self.k_e[e]   = ke[e]

    def _set_gravity(self, g: float):
        if g == 0.0: return
        m_per_node = 1.0 # may change.
        Fz = - m_per_node * g
        for i in range(self.N):
            self.f[i] = ti.Vector([0.0, 0.0, Fz])

    def _apply_bc_by_percentile(self, U, perc=95.0):
        mags = np.linalg.norm(U, axis=1)
        thr  = np.percentile(mags, perc)
        cnt  = 0
        for i in range(self.N):
            if mags[i] >= max(thr, 1e-15):
                self.is_fix[i]  = 1
                s_val = float(np.dot(U[i], self.a.to_numpy()[i]))  # project along a_i
                self.s_pres[i] = s_val
                cnt += 1

    # -------------------- precompute constant RHS parts --------------------
    @ti.kernel
    def _precompute_Lxprime_and_bias(self):
        # init Lx_b
        for i in range(self.N):
            self.Lx_b[i] = ti.Vector([0.0,0.0,0.0])
        # L x' in vector form via edge assigns: sum_e k_e * (x'_i - x'_j)
        for e in range(self.E):
            i = self.edges[e][0]; j = self.edges[e][1]
            ke = self.k_e[e]
            v  = ke * (self.x_ref[i] - self.x_ref[j])
            ti.atomic_add(self.Lx_b[i][0], v[0]); ti.atomic_add(self.Lx_b[i][1], v[1]); ti.atomic_add(self.Lx_b[i][2], v[2])
            ti.atomic_add(self.Lx_b[j][0], -v[0]); ti.atomic_add(self.Lx_b[j][1], -v[1]); ti.atomic_add(self.Lx_b[j][2], -v[2])
        # bias_s = - a·(f + Lx')
        for i in range(self.N):
            self.bias_s[i] = - ( self.a[i].dot(self.f[i] + self.Lx_b[i]) )

    # -------------------- maps and K_FF --------------------
    def _build_maps_free_fixed(self):
        free = []
        fixed= []
        for i in range(self.N):
            if int(self.is_fix[i])==1: fixed.append(i)
            else: free.append(i)
        self.free_list  = np.array(free,  dtype=np.int32)
        self.fixed_list = np.array(fixed, dtype=np.int32)
        self.nF, self.nD = len(free), len(fixed)
        if self.nF == 0:
            raise RuntimeError("No free DOFs.")

        # fill maps (device)
        for i in range(self.N): self.free_idx[i] = -1
        for i in range(self.N): self.fixed_idx[i] = -1
        for c,gi in enumerate(self.free_list):  self.free_idx[gi]  = c
        for c,gi in enumerate(self.fixed_list): self.fixed_idx[gi] = c

    def _build_and_factorize_KFF(self):
        # K_s per edge: diag +k_e to both ends; offdiag -k_e * (a_i·a_j)
        KFF_builder = ti.linalg.SparseMatrixBuilder(self.nF, self.nF, max_num_triplets= (2+2)*self.E)
        self._fill_KFF_builder(KFF_builder)
        self.K_FF = KFF_builder.build()
        self.solver = ti.linalg.SparseSolver(solver_type="LDLT")
        self.solver.analyze_pattern(self.K_FF)
        self.solver.factorize(self.K_FF)

    @ti.kernel
    def _fill_KFF_builder(self, builder: ti.types.sparse_matrix_builder()):
        for e in range(self.E):
            i = self.edges[e][0]; j = self.edges[e][1]
            fi = self.free_idx[i]; fj = self.free_idx[j]
            ke = self.k_e[e]
            # diag contributions for any free endpoint
            if fi != -1: builder[fi, fi] += ke
            if fj != -1: builder[fj, fj] += ke
            # off-diagonal if both free
            if fi != -1 and fj != -1 and fi != fj:  # Don't add to diagonal twice
                cij = self.a[i].dot(self.a[j])
                val = -ke * cij
                builder[fi, fj] += val
                builder[fj, fi] += val
        
        # Add small regularization for numerical stability  
        regularization = 1e-6
        for fi in range(self.nF):
            builder[fi, fi] += regularization

    # K_FD s_D correction (precompute once into corr_full in full indexing)
    @ti.kernel
    def _precompute_KFD_sD_correction(self):
        for i in range(self.N): self.corr_full[i] = 0.0
        for e in range(self.E):
            i = self.edges[e][0]; j = self.edges[e][1]
            ke = self.k_e[e]; cij = self.a[i].dot(self.a[j])
            # free i, fixed j
            if self.free_idx[i] != -1 and self.is_fix[j] == 1:
                ti.atomic_add(self.corr_full[i], -ke * cij * self.s_pres[j])
            # free j, fixed i
            if self.free_idx[j] != -1 and self.is_fix[i] == 1:
                ti.atomic_add(self.corr_full[j], -ke * cij * self.s_pres[i])

    # -------------------- local/global loop pieces --------------------
    @ti.kernel
    def _update_positions(self):
        for i in range(self.N):
            self.x[i] = self.x_ref[i] + self.a[i] * self.s[i]

    @ti.kernel
    def _init_d_from_rest(self):
        for e in range(self.E):
            i = self.edges[e][0]; j = self.edges[e][1]
            v = self.x_ref[i] - self.x_ref[j]
            nrm = ti.sqrt(v.dot(v)) + EPS
            self.d_e[e] = self.r_e[e] * (v / nrm)

    @ti.kernel
    def local_step(self):
        for e in range(self.E):
            i = self.edges[e][0]; j = self.edges[e][1]
            v = self.x[i] - self.x[j]
            nrm = ti.sqrt(v.dot(v))
            if nrm > EPS:
                self.d_e[e] = self.r_e[e] * (v / nrm)

    @ti.kernel
    def _assemble_rhs_full(self):
        # accum_vec = sum over edges of (+ke*d_e at i, -ke*d_e at j)
        for i in range(self.N):
            self.accum_vec[i] = ti.Vector([0.0,0.0,0.0])
        for e in range(self.E):
            i = self.edges[e][0]; j = self.edges[e][1]
            v = self.k_e[e] * self.d_e[e]
            ti.atomic_add(self.accum_vec[i][0], v[0])
            ti.atomic_add(self.accum_vec[i][1], v[1]) 
            ti.atomic_add(self.accum_vec[i][2], v[2])
            ti.atomic_add(self.accum_vec[j][0], -v[0])
            ti.atomic_add(self.accum_vec[j][1], -v[1])
            ti.atomic_add(self.accum_vec[j][2], -v[2])
        # rhs_full = a·accum_vec + bias_s  (corr applied later when compressing)
        for i in range(self.N):
            self.rhs_full[i] = self.a[i].dot(self.accum_vec[i]) + self.bias_s[i]

    @ti.kernel
    def _compress_rhs_free(self, rhs_F: ti.types.ndarray()):
        for i in range(self.N):
            fi = self.free_idx[i]
            if fi != -1:
                rhs_F[fi] = self.rhs_full[i] - self.corr_full[i]

    def global_step(self):
        # assemble RHS in full indexing
        self._assemble_rhs_full()
        # compress to free vector (Ndarray for solver)
        rhs_F = ti.ndarray(dtype=ti.f32, shape=self.nF)
        self._compress_rhs_free(rhs_F)
        # solve with cached LLT
        xF = self.solver.solve(rhs_F)
        # assign back to s on device (kernel)
        self._assign_solution(xF)

    @ti.kernel
    def _assign_solution(self, xF: ti.types.ndarray()):
        for i in range(self.N):
            if self.is_fix[i] == 1:
                self.s[i] = self.s_pres[i]
            else:
                fi = self.free_idx[i]
                self.s[i] = xF[fi]

    # -------------------- solver --------------------
    def solve(self, max_iters=50, tol=1e-6):
        prev = self.x.to_numpy().copy()
        for it in range(max_iters):
            self.local_step()
            self.global_step()
            self._update_positions()
            cur = self.x.to_numpy()
            d = np.linalg.norm(cur - prev)
            print(f"iter {it:02d}: ||Δx|| = {d:.3e}")
            if d < tol:
                print(f"Converged in {it} iterations.")
                return True
            prev = cur
        print("Did not converge.")
        return False

    def get_positions(self):     
        return self.x.to_numpy()
    
    def get_displacements(self): 
        return (self.x.to_numpy() - self.x_ref.to_numpy())


def main():
    """Main function using pre-computed matrices"""
    path = os.path.dirname(os.path.abspath(__file__))
    matrices_dir = os.path.join(path, "precomputed_matrices")
    
    print("=== Loading Pre-computed Matrices ===")
    msm = SimpleMSM(matrices_dir, gravity_g=9.8)
    print(f"SimpleMSM: {msm.N} vertices, {msm.E} edges")
    print(f"Free DOFs: {msm.nF}, Fixed DOFs: {msm.nD}")
    
    print("=== Solving Static Equilibrium ===")
    success = msm.solve(max_iters=100, tol=1e-3)
    
    if success:
        positions = msm.get_positions()
        displacements = msm.get_displacements()
        
        disp_magnitudes = np.linalg.norm(displacements, axis=1)
        print(f"\nResults:")
        print(f"  Max displacement: {disp_magnitudes.max():.6f}")
        print(f"  Mean displacement: {disp_magnitudes.mean():.6f}")
        print(f"  RMS displacement: {np.sqrt(np.mean(disp_magnitudes**2)):.6f}")
    
    success = msm.solve(max_iters=100, tol=1e-3)
    
    return msm

if __name__ == "__main__":
    main()

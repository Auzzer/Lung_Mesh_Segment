"""
Logic:
construct the force of each tetra => use force and boundary condition to update the position
1. force assembly:
Method 1: Axial and Torsion Springs follow 2.4.2.1
   1) _build_topology: construct the topology of the mesh 
   2) axis_torsion_forces: bring the axil and torsion force
   3) volume_forces: assemble the volume control force
Method 2: Continuum Mechanics follow 2.4.2.2
   1) _build_topology_Continuum_Mechanics: construct the topology of the mesh 
   2) axis_torsion_forces_Continuum_Mechanics: Deformation Forces using Continuum Mechanics
   3) volume_forces: assemble the volume control force
2. use the force and boundary condition to update the position
"""
import taichi as ti
import meshio
import numpy as np

ti.init(arch=ti.cuda, debug=True)                      
# ---------- material & global parameters ----------
LAM_MU         = {i: (50+4*i, 30+3*i) for i in range(12)}      #lame constant to stiffness Pa
AXIAL_E        = 2e3        # Pa  (along zeta_l)
TORS_G         = 1e3        # Pa  (torsion zeta_l×zeta_m)
VOL_WEIGHT     = 5e-3       # multiplies bulk modulus in W_v
DENSITY        = 1e3        # kg·m³
DAMP_COEF      = 300.0      # s⁻¹ damping coefficient
DT             = 5e-6       # time step for testing
MIN_LEN        = 1e-8
MAX_Q_PER_TET  = 6          # 2 pts × 3 axes
FACE_VERT      = ((0,2,1),(0,1,3),(0,3,2),(1,2,3))

# ------------------------------------------------------------
@ti.data_oriented
class Adamss:
    def __init__(self, pts_np, tets_np, labels_np):
        self.N, self.M   = pts_np.shape[0], tets_np.shape[0]
        self.Q           = self.M * MAX_Q_PER_TET

        # vertex state -------------------------------------------------
        self.pos  = ti.Vector.field(3, ti.f64, shape=self.N)
        self.vel  = ti.Vector.field(3, ti.f64, shape=self.N)
        self.mass = ti.field(ti.f64, shape=self.N)
        self.pos.from_numpy(pts_np.astype(np.float64))

        # connectivity & labels ---------------------------------------
        self.tets   = ti.field(ti.i64, shape=(self.M,4))
        self.labels = ti.field(ti.i64, shape=self.M)
        self.tets.from_numpy(tets_np.astype(np.int64))
        self.labels.from_numpy(labels_np.astype(np.int64))

        # placeholder anisotropy axes zeta_l (edges from v0) --------------
        axes = np.stack([ pts_np[tets_np[:,1]]-pts_np[tets_np[:,0]],
                          pts_np[tets_np[:,2]]-pts_np[tets_np[:,0]],
                          pts_np[tets_np[:,3]]-pts_np[tets_np[:,0]] ], axis=1)
        axes /= (np.linalg.norm(axes,axis=2,keepdims=True)+1e-12)
        self.ax_dir = ti.Vector.field(3, ti.f64, shape=(self.M,3))
        self.ax_dir.from_numpy(axes.astype(np.float64))

        # intersection / shape data ------------------------------------
        self.q_face = ti.field(ti.i32, shape=self.Q)
        self.q_tet  = ti.field(ti.i32, shape=self.Q)
        self.q_pos  = ti.Vector.field(3, ti.f64, shape=self.Q)
        self.bary   = ti.Vector.field(2, ti.f64, shape=self.Q)

        # rest topology & stiffness ------------------------------------
        self.vol0        = ti.field(ti.f64, shape=self.M)
        self.Dm_inv      = ti.Matrix.field(3,3, ti.f64, shape=self.M)
        self.shape_grad  = ti.Vector.field(3, ti.f64, shape=(self.M,4))

        self.ax_L0   = ti.field(ti.f64, shape=(self.M,3))
        self.ax_k0   = ti.field(ti.f64, shape=(self.M,3))
        self.tor_ang0= ti.field(ti.f64, shape=(self.M,3))
        self.tor_k0  = ti.field(ti.f64, shape=(self.M,3))

        # volume-ratio control ----------------------------------------
        self.target_r = ti.field(ti.f64, shape=self.M)   # r = V_target / V_ref
        self.p_coef   = ti.field(ti.f64, shape=self.M)   # p in W_v = p(J-r)^2

        # run-time buffers --------------------------------------------
        self.force      = ti.Vector.field(3, ti.f64, shape=self.N)
        self.ext_force  = ti.Vector.field(3, ti.f64, shape=self.N)
        self.fixed_mask = ti.field(ti.i64, shape=self.N)

        # Lame => stiffness table -------------------------------------------
        max_lbl = max(LAM_MU)
        self.lam_tbl = ti.field(ti.f64, shape=max_lbl+1)
        self.mu_tbl  = ti.field(ti.f64, shape=max_lbl+1)
        for k,(la,mu) in LAM_MU.items():
            self.lam_tbl[k]=la; self.mu_tbl[k]=mu

        self._build_topology()  # compute all initial rest-state data

    # --------- helper: vertices of a local face ----------------------
    @ti.func
    def face_vertices(self, tet, f_id):
        ia=0; ib=0; ic=0
        if   f_id==0: 
            ia = ti.cast(self.tets[tet,0], ti.i32)
            ib = ti.cast(self.tets[tet,2], ti.i32)
            ic = ti.cast(self.tets[tet,1], ti.i32)
        elif f_id==1: 
            ia = ti.cast(self.tets[tet,0], ti.i32)
            ib = ti.cast(self.tets[tet,1], ti.i32)
            ic = ti.cast(self.tets[tet,3], ti.i32)
        elif f_id==2: 
            ia = ti.cast(self.tets[tet,0], ti.i32)
            ib = ti.cast(self.tets[tet,3], ti.i32)
            ic = ti.cast(self.tets[tet,2], ti.i32)
        else:         
            ia = ti.cast(self.tets[tet,1], ti.i32)
            ib = ti.cast(self.tets[tet,2], ti.i32)
            ic = ti.cast(self.tets[tet,3], ti.i32)
        return ia,ib,ic
    # --------- distribute intersection force -----------------
    @ti.func
    def add_q_force(self, q, f):
        """
        distribute the force to the vertices of the face
        formula 2.1
        """
        tet,fc = self.q_tet[q], self.q_face[q] # tet: tetrahedron idx, fc: face idx
        ia,ib,ic = self.face_vertices(tet,fc) # ia,ib,ic are the vertices of the face
        xi,eta = self.bary[q] # xi,eta: barycentric coordinates of q in that face
        w1,w2,w3 = 1-xi-eta, xi, eta # w1,w2,w3: weights of ia, ib, ic respectively

        # add the force to the vertex ia ib ic
        ti.atomic_add(self.force[ia], w1*f) 
        ti.atomic_add(self.force[ib], w2*f) 
        ti.atomic_add(self.force[ic], w3*f)


    # ---------------------Method 1:Axial and Torsion Springs(follow 2.4.2.1)---------------------------------------
    @ti.kernel
    def _build_topology(self):
        pass
    
    @ti.kernel
    def axis_torsion_forces(self):
        pass
    # ---------------------Method 2: Continuum Mechanics(follow 2.4.2.2)---------------------------------------
    @ti.kernel
    def _build_topology_Continuum_Mechanics(self):
        dn_eps = 1e-8            # threshold of the angle between the ray and the normal of the face
        for t in range(self.M):
            """
            After initial the data structure in taichi, 
            we need to fill in the them based on each tetrahedron
            We will compute the following data in sequence:
            1. rest vertices & deformation matrix
            2. torsion and axial axis
            3. torsion and axial rest length
            4. volume-ratio parameters
            """
            # -- rest vertices & Deformation matrix---------------------------------
            
            ids = [self.tets[t, k] for k in ti.static(range(4))]
            p0, p1, p2, p3 = [self.pos[ids[k]] for k in ti.static(range(4))]
            
            # Debug: print initial positions
            print("Initial positions for tetrahedron", t)
            print("p0:", p0)
            print("p1:", p1)
            print("p2:", p2)
            print("p3:", p3)

            Dm = ti.Matrix.cols([p1 - p0, p2 - p0, p3 - p0])# deformation matrix
            print("Initial Dm matrix:")
            print(Dm)

            vol = ti.abs(Dm.determinant()) / 6.0
            print("Initial volume:", vol)

            self.vol0[t]   = vol
            self.Dm_inv[t] = Dm.inverse()
            print("Dm_inv matrix:")
            print(self.Dm_inv[t])

            # lumped mass
            for k in ti.static(range(4)):
                ti.atomic_add(self.mass[ids[k]], DENSITY * vol / 4.0)

            # shape-grad \nabla N_i (ref)
            invT = self.Dm_inv[t].transpose()
            self.shape_grad[t, 1] = ti.Vector([invT[0, 0], invT[1, 0], invT[2, 0]])
            self.shape_grad[t, 2] = ti.Vector([invT[0, 1], invT[1, 1], invT[2, 1]])
            self.shape_grad[t, 3] = ti.Vector([invT[0, 2], invT[1, 2], invT[2, 2]])
            self.shape_grad[t, 0] = -(self.shape_grad[t, 1] +
                                    self.shape_grad[t, 2] +
                                    self.shape_grad[t, 3])

            print("Shape gradients:")
            print("grad0:", self.shape_grad[t, 0])
            print("grad1:", self.shape_grad[t, 1])
            print("grad2:", self.shape_grad[t, 2])
            print("grad3:", self.shape_grad[t, 3])

            bc = (p0 + p1 + p2 + p3) * 0.25        # bary-centre

            # ---------- 6 intersection points zeta+/zeta- ----------
            for ax in ti.static(range(3)):
                base_dir = self.ax_dir[t, ax].normalized()

                for side in ti.static(range(2)):
                    ray = base_dir if side == 0 else -base_dir
                    best_d = 1e30
                    best_f = -1

                    # iterate four faces
                    for f in ti.static(range(4)):
                        ia, ib, ic = self.face_vertices(t, f)
                        cross = (self.pos[ib] - self.pos[ia]).cross(
                                self.pos[ic] - self.pos[ia])

                        if cross.norm() >= 1e-12:            
                            n_f = cross.normalized()
                            dn  = ray.dot(n_f)
                            if dn < -dn_eps:
                                d = (self.pos[ia] - bc).dot(n_f) / dn
                                if d > 0.0 and d < best_d:
                                    best_d, best_f = d, f

                        n_f = cross.normalized()
                        dn  = ray.dot(n_f)
                        if dn < -dn_eps:              # shoot outward
                            d = (self.pos[ia] - bc).dot(n_f) / dn
                            if d > 0.0 and d < best_d:
                                best_d, best_f = d, f

                    # fallback: bary-centre
                    if best_f == -1:
                        best_f, best_d = 0, 0.0

                    qid = t * MAX_Q_PER_TET + ax * 2 + side
                    self.q_tet[qid], self.q_face[qid] = t, best_f
                    self.q_pos[qid] = bc + best_d * ray

                    # barycentric (area ratio)
                    ia, ib, ic = self.face_vertices(t, best_f)
                    a, b, c = self.pos[ia], self.pos[ib], self.pos[ic]
                    v0, v1 = c - a, b - a
                    v2     = self.q_pos[qid] - a
                    den    = v0.cross(v1).norm() + 1e-12
                    xi     = v2.cross(v0).norm() / den
                    eta    = v1.cross(v2).norm() / den
                    self.bary[qid] = ti.Vector([xi, eta])

            # -- axial rest length / k ----------------------------------
            for ax in ti.static(range(3)):
                qp, qm = t * MAX_Q_PER_TET + ax * 2, t * MAX_Q_PER_TET + ax * 2 + 1
                L0 = (self.q_pos[qm] - self.q_pos[qp]).norm()
                self.ax_L0[t, ax] = L0
                self.ax_k0[t, ax] = AXIAL_E

            # -- torsion rest angle / k ---------------------------------
            for a, b, idx in ti.static(((0, 1, 0), (0, 2, 1), (1, 2, 2))):
                qa, qb = t * MAX_Q_PER_TET + a * 2, t * MAX_Q_PER_TET + b * 2
                za = (self.q_pos[qa + 1] - self.q_pos[qa]).normalized()
                zb = (self.q_pos[qb + 1] - self.q_pos[qb]).normalized()
                self.tor_ang0[t, idx] = ti.acos(
                    ti.min(0.9999, ti.max(-0.9999, za.dot(zb))))
                self.tor_k0[t, idx] = TORS_G

            # -- volume-ratio parameters --------------------------------
            self.target_r[t] = 1.0
            lam, mu = self.lam_tbl[self.labels[t]], self.mu_tbl[self.labels[t]]
            bulk    = lam + 2 * mu / 3
            self.p_coef[t] = VOL_WEIGHT * bulk

    # --------- axial + torsion spring forces with Continuum Mechanics (section 2.4.2.2) --------------------
    @ti.kernel
    def axis_torsion_forces_Continuum_Mechanics(self):
        for i in self.force: self.force[i]=ti.Vector.zero(ti.f64,3)
        for t in range(self.M):
            # Debug: print tetrahedron index and label
            print("Processing tetrahedron", t, "with label", self.labels[t])
            
            # axial ---------------------------------------------------
            for ax in ti.static(range(3)):
                qp, qm = t*MAX_Q_PER_TET+ax*2, t*MAX_Q_PER_TET+ax*2+1
                vec = self.q_pos[qm]-self.q_pos[qp]
                L = vec.norm()
                # Skip if length is too small
                if L >= MIN_LEN:
                    dir = vec/L
                    eps = L - self.ax_L0[t,ax]
                    print("Axial force for axis", ax, "eps:", eps, "L:", L, "L0:", self.ax_L0[t,ax])

                    # dynamic k at qp
                    ia,ib,ic = self.face_vertices(t,self.q_face[qp])
                    cross=(self.pos[ib]-self.pos[ia]).cross(self.pos[ic]-self.pos[ia])
                    area=0.5*cross.norm()
                    if area >= MIN_LEN:
                        n=cross.normalized()
                        kqp = self.ax_k0[t,ax]*area*abs(n.dot(dir))
                        print("Axial force at qp:", -kqp*eps*dir)
                        self.add_q_force(qp, -kqp*eps*dir)

                    # dynamic k at qm
                    ia,ib,ic = self.face_vertices(t,self.q_face[qm])
                    cross=(self.pos[ib]-self.pos[ia]).cross(self.pos[ic]-self.pos[ia])
                    area=0.5*cross.norm()
                    if area >= MIN_LEN:
                        n=cross.normalized()
                        kqm = self.ax_k0[t,ax]*area*abs(n.dot(dir))
                        print("Axial force at qm:", kqm*eps*dir)
                        self.add_q_force(qm,  kqm*eps*dir)

            # torsion -------------------------------------------------
            for a,b,idx in ti.static(((0,1,0),(0,2,1),(1,2,2))):
                qa,qb = t*MAX_Q_PER_TET+a*2, t*MAX_Q_PER_TET+b*2
                za=(self.q_pos[qa+1]-self.q_pos[qa])
                zb=(self.q_pos[qb+1]-self.q_pos[qb])
                za_norm = za.norm()
                zb_norm = zb.norm()
                
                if za_norm >= MIN_LEN and zb_norm >= MIN_LEN:
                    za = za/za_norm
                    zb = zb/zb_norm
                    dth=ti.acos(ti.min(0.9999,ti.max(-0.9999,za.dot(zb)))) - self.tor_ang0[t,idx]
                    print("Torsion angle difference:", dth, "for axes", a, b)

                    # dynamic k at ends of axis a
                    ia,ib,ic=self.face_vertices(t,self.q_face[qa])
                    cross=(self.pos[ib]-self.pos[ia]).cross(self.pos[ic]-self.pos[ia])
                    area=0.5*cross.norm()
                    if area >= MIN_LEN:
                        n=cross.normalized()
                        ka=self.tor_k0[t,idx]*area*abs(n.dot(za))
                        print("Torsion force at qa:", -ka*dth*zb)
                        self.add_q_force(qa,  -ka*dth*zb)
                        self.add_q_force(qa+1, ka*dth*zb)

                    # dynamic k at ends of axis b
                    ia,ib,ic=self.face_vertices(t,self.q_face[qb])
                    cross=(self.pos[ib]-self.pos[ia]).cross(self.pos[ic]-self.pos[ia])
                    area=0.5*cross.norm()
                    if area >= MIN_LEN:
                        n=cross.normalized()
                        kb=self.tor_k0[t,idx]*area*abs(n.dot(zb))
                        print("Torsion force at qb:", -kb*dth*za)
                        self.add_q_force(qb,  -kb*dth*za)
                        self.add_q_force(qb+1, kb*dth*za)

    # --------- continuum volume control force (2.4.2.4) -------------------
    @ti.kernel
    def volume_forces(self):
        for t in range(self.M):
            # Debug: print tetrahedron index and label
            print("Processing volume forces for tetrahedron", t, "with label", self.labels[t])
            
            # Fix type casting warnings by explicitly casting indices to int32
            id0 = ti.cast(self.tets[t,0], ti.i32)
            id1 = ti.cast(self.tets[t,1], ti.i32)
            id2 = ti.cast(self.tets[t,2], ti.i32)
            id3 = ti.cast(self.tets[t,3], ti.i32)
            x0,x1,x2,x3 = self.pos[id0],self.pos[id1],self.pos[id2],self.pos[id3]

            # Debug: print vertex positions
            print("Vertex positions for tetrahedron", t)
            print("x0:", x0)
            print("x1:", x1)
            print("x2:", x2)
            print("x3:", x3)

            # Debug: print Dm_inv
            print("Dm_inv for tetrahedron", t)
            print(self.Dm_inv[t])

            Ds = ti.Matrix.cols([x1-x0, x2-x0, x3-x0])
            print("Ds matrix:")
            print(Ds)

            F  = Ds @ self.Dm_inv[t]
            print("Deformation gradient F:")
            print(F)

            J  = F.determinant()
            print("Volume ratio J:", J, "for tetrahedron", t)
            
            # Add numerical stability check
            if ti.abs(J) >= 1e-8:
                r  = self.target_r[t]
                p  = self.p_coef[t]
                print("Volume force parameters - r:", r, "p:", p)

                P = 2.0 * p * (J - r) * J * F.inverse().transpose()   # 1st-Piola
                print("First Piola-Kirchhoff stress:", P)

                V0 = self.vol0[t]
                for k in ti.static(range(4)):
                    grad = self.shape_grad[t,k]
                    fi   = -V0 * (P @ grad)            # nodal force
                    print("Volume force at vertex", k, ":", fi)
                    # Fix type casting warning by explicitly casting index to int32
                    ti.atomic_add(self.force[ti.cast(self.tets[t,k], ti.i32)], fi)

    # --------- external forces (don't need at the moment)-----------
    @ti.kernel
    def add_ext(self):
        for i in self.force: self.force[i] += self.ext_force[i]
    
    # --------- wrap the force  -------------------------------------
    def assemble_forces(self):
        # Clear forces first
        self.clear_forces()
        
        # Compute forces in sequence
        self.axis_torsion_forces_Continuum_Mechanics()
        self.volume_forces()
        
        # Debug: print max force magnitude
        max_force = 0.0
        force_np = self.force.to_numpy()
        for i in range(len(force_np)):
            force_mag = np.linalg.norm(force_np[i])
            if force_mag > max_force:
                max_force = force_mag
        print("Max force magnitude:", max_force)

    @ti.kernel
    def clear_forces(self):
        for i in self.force:
            self.force[i] = ti.Vector.zero(ti.f64, 3)

    # --------- explicit integration (for debugging) --------------
    @ti.kernel
    def integrate(self, dt: ti.f64):
        for i in range(self.N):
            if self.fixed_mask[i]==0:
                mass = self.mass[i] + 1e-8
                a = self.force[i]/mass
                self.vel[i]+=dt*a
                self.vel[i]*=ti.exp(-dt*DAMP_COEF)
                self.pos[i]+=dt*self.vel[i]

    @ti.kernel
    def update_q_pos(self):
        for q in range(self.Q):
            tet,fc = self.q_tet[q], self.q_face[q]
            ia,ib,ic = self.face_vertices(tet,fc)
            xi,eta = self.bary[q]
            self.q_pos[q] = (1-xi-eta)*self.pos[ia] + xi*self.pos[ib] + eta*self.pos[ic]

    # --------- step the simulation for debugging -----------------------------------
    def step(self, dt=DT):
        self.assemble_forces()
        self.integrate(dt)
        self.update_q_pos()

    # --------- helpers for debugging -----------------------------------------------
    def volume_max(self): return(np.linalg.norm(self.vel.to_numpy(),axis=1).max())

# ------------------------------------------------------------
# demo for testing
# ------------------------------------------------------------
if __name__ == "__main__":
    mesh = meshio.read("/home/haozhe/lung-project/HeterogeneousSegmentGNN/LungSimulation/mesh_files/case1_T00_lung_regions_11.xdmf")
    pts  = mesh.points.astype(np.float64)           # mm → m
    tid  = next(i for i,c in enumerate(mesh.cells) if c.type=="tetra")
    tets = mesh.cells[tid].data.astype(np.int64)
    lbls = mesh.cell_data["c_labels"][tid].astype(np.int64)
    
    sim = Adamss(pts, tets, lbls)#Adamss(pts, tets, lbls)
    sim.target_r.fill(1.2)# example: inflate to 120 % of rest volume and compute static forces
    sim.assemble_forces()
    print("max |f| after inflation:", np.linalg.norm(sim.force.to_numpy(),axis=1).max())

    for _ in range(100):
        sim.step() #  explicit dynamics for testing



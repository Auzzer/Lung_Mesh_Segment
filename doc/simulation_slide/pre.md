## **P1 — A Simple Spring–Mass Analogy (2D cloth)**

* **Mass points & springs.** Cloth meshes (2D) are often modeled with masses at vertices and linear springs on edges.
  **Stretch** springs penalize edge length changes; **shear** springs penalize angle change between warp/weft; (optional) **bending** springs penalize dihedral changes along folds.

* **Energy**

  $$
  U_{\text{cloth}}=\tfrac12\!\sum_{(i,j)\in E}k_s\big(\|X_i-X_j\|-\|x_i-x_j\|\big)^2
  \;
  $$

  Differentiation gives **internal forces** that are linear in displacements around the rest state.

Our 3D **SMS (for tetrahedra)** follows the same idea: we define simple “measurements” of deformation along a few directions and build a **quadratic energy**, whose gradient gives internal forces and whose Hessian gives a **global stiffness**. The next pages lift this intuition to 3D tets with anisotropy.

---

## **P2 — Notation and Problem Setup**

* **Tetrahedral mesh with $N$ vertices.**

  $$
  \mathbf{x}\in\mathbb{R}^{3N} \ \text{(reference)},\qquad
  \mathbf{X}\in\mathbb{R}^{3N} \ \text{(current)},\qquad
  \mathbf{u}=\mathbf{X}-\mathbf{x}.
  $$
* **Dirichlet data from image registration.**
  On the constrained set $\mathcal D$: $X_i=x_i+u_i$. Free DOFs $\mathcal F$ are unknown.
* **Per‑tet notation.**
  For $\mathcal V_k$, reference vertices $x_0,x_1,x_2,x_3$, barycenter $x_b=\tfrac14\sum_{i=0}^3 x_i$.
* **Goal.** Solve for $\mathbf{X}_\mathcal{F}$ so that internal elastic forces balance external loads on the free set.

---

## **P3 — Deformation Gradient $F_k$ from Edge Matrices & SVD Axes**

We adopt your **edge‑matrix** construction.

* Form 3×3 edge matrices in each tet (local nodes $0,1,2,3$):

  $$
  D_m=\big[X_1{-}X_0,\ X_2{-}X_0,\ X_3{-}X_0\big],\qquad
  d_x=\big[x_1{-}x_0,\ x_2{-}x_0,\ x_3{-}x_0\big].
  $$
* Assume the mapping **current → reference** is affine: $x=F X+\text{const}$. Eliminating the translation gives

  $$
  d_x = F\, D_m \ \ \Rightarrow\ \ \boxed{\,F_k = d_x\, D_m^{-1}\,}.
  $$

  (For a nondegenerate tet, $D_m$ is invertible.)
* **SVD and axes of anisotropy.**
  Take $F_k=U_k\Sigma_kV_k^\top$. Since $F_k$ maps **current→reference**, the **left** singular vectors $U_k$ live in the reference space. We **set the three orthonormal axes**

  $$
  \boxed{\,e_\ell=(U_k)_{:\ell},\ \ell=0,1,2\,}
  $$

  (enforce right‑handedness if needed). These axes are **fixed afterwards** and used in the energy on P5.



---



## P4 — Intersection Points & Coefficient Matrix

* **Barycenter and rays.**

  $$
  x_b=\tfrac14\sum_{i=1}^4 x_i .
  $$

  From $x_b$ cast rays along $\pm e_0,\pm e_1,\pm e_2$ (axes from SVD, see **P3**) to the tet faces. Each axis intersects twice → six points $q_j$ $(j=1,\dots,6)$.

* **Point-in-triangle test & local (area) coordinates.**
  If a ray hits face $\Delta_{i_1 i_2 i_3}$,

  $$
  S_{\Delta_{i_1 i_2 i_3}}
  = S_{\Delta_{q_j i_2 i_3}} + S_{\Delta_{i_1 q_j i_3}} + S_{\Delta_{i_1 i_2 q_j}},
  $$

  and the area (barycentric) coordinates are

  $$
  \xi=\frac{S_{\Delta_{q_j i_2 i_3}}}{S_{\Delta_{i_1 i_2 i_3}}},\quad
  \eta=\frac{S_{\Delta_{q_j i_1 i_3}}}{S_{\Delta_{i_1 i_2 i_3}}},\quad
  1-\xi-\eta=\frac{S_{\Delta_{i_1 i_2 q_j}}}{S_{\Delta_{i_1 i_2 i_3}}}.
  $$

* **Coefficient matrix $C^k\in\mathbb{R}^{4\times 6}$.**
  Let $N_i$ be the linear shape functions. On the hit face:

  $$
  N_{i_1}(q_j)=1-\xi-\eta,\;\; N_{i_2}(q_j)=\xi,\;\; N_{i_3}(q_j)=\eta,\;\; N_{i_4}(q_j)=0.
  $$

  Define

  $$
  C^k_{ij}=N_i(q_j)\quad(i=1..4,\ j=1..6).
  $$

  With current vertices $X_i^t$, intersections update **affinely**:

  $$
  X_j^t=\sum_{i=1}^4 C^k_{ij}\,X_i^t .
  $$

## P5 — Final Energy 

**with the previous SVD-based axes, we can define energy :**

$$
\boxed{
U_k(F_k)=V_k\!\left[
\frac{\alpha_k}{2}\sum_{\ell=0}^{2}\big(\,\|F_k e_\ell\|^2-1\,\big)^2
+\frac{\beta_k}{2}\sum_{\ell<m}\big(\,(F_k e_\ell)\!\cdot\!(F_k e_m)\,\big)^2
+\frac{\kappa_k}{2}\,(J_k-1)^2
\right]}
$$

with $J_k=\det F_k$.



* **Axial:** $\|F_ke_\ell\|^2 = e_\ell^\top C_k e_\ell$ with $C_k=F_k^\top F_k$ (right Cauchy–Green).
  At rest $C_k=I\Rightarrow \|F_ke_\ell\|^2-1=0$. For small strain $F_k=I+H$,

  $$
  \|F_ke_\ell\|^2-1 \approx 2\,e_\ell^\top\varepsilon\,e_\ell = 2\,\varepsilon_{\ell\ell},
  $$

  i.e., it is the **normal strain along $e_\ell$**. Squaring penalizes axial stretch/compression along that axis.

* **Shear:** $(F_k e_\ell)\!\cdot\!(F_k e_m)=e_\ell^\top C_k e_m$ is the **off‑diagonal** entry of the metric in the $\{e_\ell\}$ frame.
  At rest, axes are orthonormal $\Rightarrow 0$. For small strain,

  $$
  (F_k e_\ell)\!\cdot\!(F_k e_m) \approx 2\,\varepsilon_{\ell m},
  $$

  i.e., it captures **loss of orthogonality** (shear) between the two deformed axis images. Squaring penalizes that shear.

* **Volume reservation:** $J_k-1$ is the relative volume change; for small strain $J_k-1\approx \operatorname{tr}\varepsilon$.

**Small‑strain quadratic form (SMS):**

$$
w_{\text{quad}}(\varepsilon)
=2\alpha_k\!\sum_\ell \varepsilon_{\ell\ell}^2
+2\beta_k\!\sum_{\ell<m}\varepsilon_{\ell m}^2
+\tfrac{\kappa_k}{2}(\operatorname{tr}\varepsilon)^2,\qquad
U_k\approx V_k\,w_{\text{quad}}(\varepsilon_k).
$$

**Continuous FEM correspondence.**
Linear elasticity $W_{\text{FEM}}(\varepsilon)=\mu\,\varepsilon:\varepsilon+\tfrac{\lambda}{2}(\operatorname{tr}\varepsilon)^2$ is recovered by
$\alpha=\mu/2,\ \beta=\mu,\ \kappa=\lambda$.

---

## P6 — Forward Solver

1. **Assemble** the global stiffness $K(\alpha,\beta,\kappa)$ by summing per‑tet contributions from the quadratic form above (axis, shear, volume parts) and restricting to **free DOFs**.
2. **Build the RHS** $b$ from body forces (e.g., gravity via lumped masses) 
3. **Solve the SPD system**

   $$
   K\,u=b
   $$

Then recover $\mathbf{X}=\mathbf{x}+u$.

---



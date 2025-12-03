# 1) Energy and Notation
For a tet $k$ with reference volume $V_k$ and a fixed orthonormal triad $\{e_0,e_1,e_2\}$ in the **reference** configuration, the hyperelastic energy we wrote is

$$
U_k(F_k)=V_k\Bigg[
\frac{\alpha}{2}\sum_{\ell=0}^{2}\big(\|F_ke_\ell\|^2-1\big)^2
+\frac{\beta}{2}\sum_{\ell<m}\big((F_ke_\ell) \cdot(F_ke_m)\big)^2
+\frac{\kappa}{2}(J_k-1)^2
\Bigg],
$$

with $F_k=\partial x/\partial X$ and $J_k=\det F_k$.

Key tensors:

$$
C = F^\top F,\qquad \text{so}\quad
\|Fe_\ell\|^2 = e_\ell^\top C\,e_\ell,\quad
(Fe_\ell)\!\cdot\!(Fe_m) = e_\ell^\top C\,e_m .
$$

We will linearize around the identity using the displacement $u(X)$:

$$
F=I+H,\quad H:=\nabla u,\quad \varepsilon:=\tfrac12(H+H^\top).
$$

---

# 2) Taylor expansions up to quadratic order

## 2.1 Metric tensor $C$

$$
C = F^\top F=(I+H)^\top(I+H)=I+H+H^\top+H^\top H
= I + 2\varepsilon + H^\top H .
$$

* $2\varepsilon$ is **first order** in $\|H\|$.
* $H^\top H$ is **second order**.

Therefore

$$
C-I = 2\varepsilon + O(\|H\|^2).
$$

## 2.2 Directional norms/dots

Using $C$,

$$
\|Fe_\ell\|^2-1 = e_\ell^\top(C-I)e_\ell
= 2\,e_\ell^\top\varepsilon\,e_\ell + e_\ell^\top(H^\top H)e_\ell.
$$

Keep only the first-order part **inside** the parentheses because the whole term will be squared in the energy (details just below):

$$
\|Fe_\ell\|^2-1 \;\approx\; 2\,\varepsilon_{\ell\ell}
\quad\text{with}\quad \varepsilon_{\ell\ell}:=e_\ell^\top\varepsilon e_\ell .
$$

Likewise, for $\ell\neq m$,

$$
(Fe_\ell)\!\cdot\!(Fe_m)= e_\ell^\top C e_m
= 2\,\varepsilon_{\ell m} + e_\ell^\top(H^\top H)e_m
\;\approx\;2\,\varepsilon_{\ell m},
$$

where $\varepsilon_{\ell m}:=e_\ell^\top\varepsilon e_m$.

### Why can we drop $H^\top H$ **inside** the square

Write generically $a = a_1 + a_2$ with $a_1=O(\|H\|)$, $a_2=O(\|H\|^2)$. Then

$$
a^2 = a_1^2 + 2a_1a_2 + a_2^2
= O(\|H\|^2) + O(\|H\|^3) + O(\|H\|^4).
$$

Since we seek a **quadratic** (second-order) energy, we keep only $a_1^2$ and drop the $O(\|H\|^3)$ and $O(\|H\|^4)$ parts. This is exactly what we did.

## 2.3 Determinant $J=\det F=\det(I+H)$

Two equivalent second-order expansions:

* **Jacobi/log-det route**:

  $$
  \log\det(I+H)=\operatorname{tr}\log(I+H)
  = \operatorname{tr}\Big(H - \tfrac12 H^2 + O(H^3)\Big)
  = \operatorname{tr}H - \tfrac12\operatorname{tr}(H^2) + O(H^3),
  $$

  hence

  $$
  \det(I+H)=\exp\!\big(\operatorname{tr}H - \tfrac12\operatorname{tr}(H^2) + O(H^3)\big)
  = 1 + \operatorname{tr}H + \tfrac12\big((\operatorname{tr}H)^2 - \operatorname{tr}(H^2)\big) + O(H^3).
  $$

* **Two-point Taylor via Jacobi’s formula**:
  Let $J(t)=\det(I+tH)$. Then $J'(0)=\operatorname{tr}H$ and
  $J''(0)=(\operatorname{tr}H)^2 - \operatorname{tr}(H^2)$.
  Therefore

  $$
  J(1)=J(0)+J'(0)+\tfrac12 J''(0) + O(H^3)
  = 1 + \operatorname{tr}H + \tfrac12\big((\operatorname{tr}H)^2 - \operatorname{tr}(H^2)\big) + O(H^3).
  $$

Thus

$$
J-1 = \underbrace{\operatorname{tr}H}_{O(H)}\; + \;\underbrace{\tfrac12\big((\operatorname{tr}H)^2 - \operatorname{tr}(H^2)\big)}_{O(H^2)} \;+\; O(H^3).
$$

Squaring and keeping up to order $O(H^2)$:

$$
(J-1)^2 = (\operatorname{tr}H)^2 + O(H^3).
$$

Finally, since $\operatorname{tr}\varepsilon=\tfrac12\operatorname{tr}(H+H^\top)=\operatorname{tr}H$,

$$
(J-1)^2 \approx (\operatorname{tr}\varepsilon)^2.
$$

---



Substituting back gives the **quadratic energy density**

$$
w_{\text{quad}}(\varepsilon)
=2\alpha\sum_{\ell}\varepsilon_{\ell\ell}^2
+2\beta\sum_{\ell<m}\varepsilon_{\ell m}^2
+\frac{\kappa}{2}\big(\operatorname{tr}\varepsilon\big)^2.
$$

In linear tetrahedra, $\varepsilon$ is constant within the element, so

$$
U_k \approx V_k\,w_{\text{quad}}(\varepsilon_k).
$$

> Note: To align with isotropic linear elasticity $w=\mu\|\varepsilon\|_F^2+\frac{\lambda}{2}(\operatorname{tr}\varepsilon)^2$, simply take
> $\alpha=\mu/2,\ \beta=\mu,\ \kappa=\lambda$. However, the derivation below holds for arbitrary $\alpha,\beta,\kappa$.

---

# 3) Express $\varepsilon$ as **Linear** Functions of Nodal Displacements

Let the displacements of the four vertices of the element be $\{u_0,u_1,u_2,u_3\}$, stack them as

$$
u_e=\begin{bmatrix}
u_0^x,u_0^y,u_0^z,\;u_1^x,\dots,\;u_3^z
\end{bmatrix}^\top\in\mathbb R^{12}.
$$

So for each element, we have four nodes × three directions = 12 degrees of freedom. Any linear scalar measurement $\text{lsm}$ within the element can be written as

$$
\text{lsm} = r\,u_e,
$$

where $r\in\mathbb R^{1\times 12}$ a **linear measurement row vector**. We construct three types of measurements to approximate the three types of components of $\varepsilon$.

## 3.1 Axial Strain $\varepsilon_{\ell\ell}$

In the reference position, take two points $p_\ell^{(1)},p_\ell^{(2)}$ collinear with axis $e_\ell$. Express using the **barycentric coordinates** of the tetrahedron:

$$
p = \sum_{a=0}^3 C_k[a,\text{pt}]\,X_a,\qquad
X_a\ \text{are vertex coordinates},
$$ 
where $a$ is the tetra vertex index and pt is the intersection points index.

Linear displacement interpolation gives

$$
u(p)=\sum_{a=0}^3 C_k[a,\text{pt}]\,u_a.
$$

Let

$$
s_\ell[a]=C_k[a,p_\ell^{(2)}]-C_k[a,p_\ell^{(1)}],\quad
\Delta u_\ell=\sum_a s_\ell[a]\,u_a,\quad
\ell_\ell^0=\|p_\ell^{(2)}-p_\ell^{(1)}\|.
$$

The small strain component along $e_\ell$ is approximated as

$$
\varepsilon_{\ell\ell}\;\approx\;\frac{e_\ell^\top \Delta u_\ell}{\ell_\ell^0}.
$$

Thus, to write it in the form $\text{lsm}_{\ell\ell}=r_{\ell\ell}u_e$ as a row vector, we need each nodal block

$$
r_{\ell\ell}^{(a)}=\frac{s_\ell[a]}{\ell_\ell^0}\,e_\ell^\top
\quad\Longrightarrow\quad
r_{\ell\ell}=\big[r_{\ell\ell}^{(0)}\ \ r_{\ell\ell}^{(1)}\ \ r_{\ell\ell}^{(2)}\ \ r_{\ell\ell}^{(3)}\big]\in\mathbb R^{1\times 12}.
$$

## 3.2 Shear Strain $\varepsilon_{\ell m}$ ($\ell\neq m$)

$$
\varepsilon_{\ell m}
=\tfrac12\,(e_\ell^\top \nabla u\, e_m + e_m^\top \nabla u\, e_\ell)
\;\approx\;\tfrac12\!\left(
\frac{e_m^\top\Delta u_\ell}{\ell_\ell^0}
+\frac{e_\ell^\top\Delta u_m}{\ell_m^0}
\right).
$$

Therefore, define the measurement

$$
\text{lsm}_{\ell m}=\tfrac12\!\left(
\frac{e_m^\top\Delta u_\ell}{\ell_\ell^0}
+\frac{e_\ell^\top\Delta u_m}{\ell_m^0}
\right)=r_{\ell m}u_e,
$$

where each nodal block is

$$
r_{\ell m}^{(a)}=\tfrac12\!\left(
\frac{s_\ell[a]}{\ell_\ell^0}\,e_m^\top
+\frac{s_m[a]}{\ell_m^0}\,e_\ell^\top
\right),
$$

assembled into $r_{\ell m}\in\mathbb R^{1\times 12}$.

## 3.3 Volume/Trace $\operatorname{tr}\varepsilon$

The first-order volume change of a linear tetrahedron is

$$
\delta V=\sum_{a=0}^3 g_a\cdot u_a,
$$

where $g_a$ is the **volume gradient** in the reference configuration (proportional to the outward normal of the opposite face):

$$
g_0=\frac{(X_1-X_2)\times(X_3-X_2)}{6},\quad
g_1=\frac{(X_2-X_0)\times(X_3-X_0)}{6},\quad
\newline
g_2=\frac{(X_0-X_1)\times(X_3-X_1)}{6},\quad
g_3=\frac{(X_0-X_2)\times(X_1-X_2)}{6}.
$$

Therefore

$$
\text{lsm}_{\text{vol}}=\frac{\delta V}{V_k}
=\sum_a\frac{g_a}{V_k}\cdot u_a
=r_{\text{vol}}u_e
\quad\text{and}\quad
r_{\text{vol}}^{(a)}=\left(\frac{g_a}{V_k}\right)^\top .
$$

Since $\delta V/V_k\approx \operatorname{tr}\varepsilon$, this measurement corresponds to the trace component.


## Summary:
The three types of row vectors $r_{\ell\ell},r_{\ell m},r_{\text{vol}}$ are all of **length 12**, with 4 nodal blocks each being $1\times 3$ row vectors, exactly expressing the strain components within the element as **linear** combinations of nodal displacements.

---

# 4) From Energy to Stiffness

The element quadratic energy can be uniformly written as

$$
U_k(u_e)=\sum_{\ell}(2\alpha V_k)\,\text{lsm}_{\ell\ell}^2
+\sum_{\ell<m}(2\beta V_k)\,\text{lsm}_{\ell m}^2
+\Big(\tfrac{\kappa}{2}V_k\Big)\,\text{lsm}_{\text{vol}}^2
=\frac12\,u_e^\top K_k\,u_e.
$$

Since $\text{lsm}=r\,u_e$ is **linear**, we have

$$
\text{lsm}^2=(r\,u_e)^2=u_e^\top (r^\top r)\,u_e.
$$

For each term $c\,\text{lsm}^2$ (with coefficient $c$ as above), the corresponding element stiffness contribution is

$$
K_k\ \mathrel{+}= \ 2c \; r^\top r.
$$

Substituting the three coefficients $c$ gives

$$
\boxed{
\begin{aligned}
K_k \ \mathrel{+}&=\ \sum_{\ell=0}^{2}\big(4\alpha V_k\big)\, r_{\ell\ell}^\top r_{\ell\ell}\\
&\quad+\ \sum_{\ell<m}\big(4\beta V_k\big)\, r_{\ell m}^\top r_{\ell m}\\
&\quad+\ \big(\kappa V_k\big)\, r_{\text{vol}}^\top r_{\text{vol}}.
\end{aligned}}
$$

Assembling $K_k$ using standard assembly (according to nodal local-to-global mapping) into the global matrix yields a constant, symmetric positive definite $K$ (after removing rigid body modes). 

Note that the sparse matrix K is symetric and psd.

---

# 5) RHS ($r$ vector; written as $b$ in the code)

The equilibrium equation is $K_{FF} u_F = f_F$. For this quadratic model, the internal force $f_{\text{int}}=K\,u$, so the RHS is the **external force**:

* Body force (gravity): In kg–mm–s units, $g=[0,-9810,0]\; \mathrm{mm/s^2}$, nodal mass $m_i$ (already obtained from $V[\mathrm{m}^3]$ in preprocessing to get kg). For each free node $i$,

  $$
  f_F^{(i)} \mathrel{+}= m_i\, g .
  $$
* Surface traction/pressure: For boundary triangles, perform numerical integration based on reference area, distribute forces to three nodes, then map to free DOFs and accumulate.

Since Dirichlet DOFs are **compressed and removed** (we directly set boundary positions as $x=X_0+u_{\text{bc}}$ and remove them from the system), the current approach **does not** need to add correction terms like $-K_{FD}u_D$.

---

# 6) Implementation Details in Code

1. In the reference configuration, find axial intersection points $p_\ell^{(1)},p_\ell^{(2)}$, obtain barycentric coefficients $C_k[a,\text{pt}]$ and axial length $\ell_\ell^0$.
   $s_\ell[a]=C_k[a,p^{(2)}_\ell]-C_k[a,p^{(1)}_\ell]$.
2. Calculate three axial row vectors:
   $r_{\ell\ell}^{(a)}=\dfrac{s_\ell[a]}{\ell_\ell^0}e_\ell^\top$.
3. Calculate three shear row vectors:
   $r_{\ell m}^{(a)}=\tfrac12\big(\dfrac{s_\ell[a]}{\ell_\ell^0}e_m^\top+\dfrac{s_m[a]}{\ell_m^0}e_\ell^\top\big)$.
4. Calculate volume gradients $g_a$ from reference vertices $\{X_a\}$ (cross product formula above), obtain
   $r_{\text{vol}}^{(a)}=(g_a/V_k)^\top$.
5. For each element, perform outer products of the three types of row vectors and multiply by corresponding coefficients $4\alpha V_k,4\beta V_k,\kappa V_k$, assemble into the sparse builder for free DOFs (write only upper triangle, skip near-zero components).
6. RHS: Multiply free nodal masses by gravity and accumulate to $f_F$; if there are pressure/traction forces, accumulate similarly.
7. Solve $K_{FF}u_F=f_F$, fill $u_F$ back to the full field and update free nodal positions.




## 6.1 Torch CSR assembly (rows, cols, vals)

Implementation follows the verification solvers and assembles the global free-DOF
matrix on CUDA as triplets, then builds a combined sparse tensor and converts to
CSR. The assembler are accelerated with taichi's kernel.  
Only three 1-D tensors are needed:

- `rows[nnz]` (int64): row indices of nonzeros
- `cols[nnz]` (int64): column indices of nonzeros
- `vals[nnz]` (float32): values of nonzeros

Triplets are emitted by a GPU kernel as follows (logic):

- Map each element’s 12 local DOFs to global free-DOF indices using the compressed
  `dof_map` (constrained entries are `-1` and skipped).
- For every element, evaluate the three SMS contributions (axial, shear, volumetric):
  compute inner products of the corresponding 12-length row vectors and form the
  scalar contribution `val = V_k (4 alpha s_ax + 4 beta s_sh + kappa s_v)`.
- For all local pairs `(p, q)` with valid global indices `(gp, gq)`, append one
  triplet `(rows[idx]=gp, cols[idx]=gq, vals[idx]=val)` using an atomic counter.
- After emission, truncate the buffers to the written `nnz` and combine duplicate
  pairs `(row, col)` by summation. We need `nnz` because:
  1. Capacity vs. actual writes: The triplet buffers (rows, cols, vals) are allocated with a conservative capacity (cap = 144*M), but the kernel usually writes fewer entries. nnz records the actual number of emitted nonzeros.
  2. Correct slicing: Only the first nnz slots contain valid data. Truncating to [:nnz] avoids passing uninitialized buffer tails to PyTorch, which would otherwise introduce garbage or explicit zeros.
  3. COO construction requires exact length: When building a COO tensor, the indices/values arrays must have equal, correct length. Using nnz guarantees the tensor reflects exactly the emitted entries.
  4. Performance: Smaller arrays (sliced to nnz) reduce data transfer and make coalescing (merging duplicates) faster, since the combiner doesn’t scan unused capacity.
  5. Numerical correctness: Including buffer tails can create spurious entries (even explicit zeros), altering sparsity structure and slowing or corrupting the conversion to CSR and downstream solves.
  6. Natural with FE assembly: Many elements contribute to the same global (row, col), producing duplicates by design. The workflow is “emit triplets → slice to nnz → combine duplicates by sum → convert to CSR,” and nnz is the precise count that makes those steps correct and efficient.
- Build a COO sparse tensor from `(rows, cols, vals)`, combine it, and convert to
  CSR. Row pointers (`crow_indices`) are created internally by the framework when
  converting COO→CSR, so they do not need to be built manually.

## 6.2 Build RHS and solve

Gravity RHS on free DOFs is accumulated by mapping each free node to its free-DOF
rows and adding `mass[i] * g` (with `g = (0, 0, -9.81)`) component-wise. Solve the
CSR system with a helper that:

- Accepts `(A_csr, b_free, tol)` and returns `(solution, iterations, residual_norm,
  converged)`.
- Uses a sparse-direct path when available, otherwise falls back to a dense
  `torch.linalg.solve` on the materialized matrix.
- Checks convergence via `torch.allclose(residual, 0, rtol=tol, atol=tol)`.
---

# 7) Inverse Problem

Two equivalent routes exist for computing gradients of the misfit
$$
\mathcal L(u^*) = \frac{\|u^* - u_{\text{obs}}\|_2^2}{\|u_{\text{obs}}\|_2^2}
$$
with respect to the parameters
$\{\theta_k\}_{k=1}^P$
appearing in $K(\theta)$. The denominator uses the same DOF subset as $u^*$
(typically the free-DOF restriction), so it is a fixed normalization constant
for a given set of observations.

## 7.1 Forward sensitivity (direct differentiation)

Differentiate the equilibrium equation:

$$
K\,\frac{\partial u^*}{\partial \theta_k}
\;=\;\underbrace{\frac{\partial b}{\partial \theta_k}-\frac{\partial K}{\partial \theta_k}\,u^*}_{\displaystyle r_k} .
$$

Treat the right-hand side as a new load vector and solve one linear system per
parameter:

$$
K\,w_k=r_k,\qquad w_k=\frac{\partial u^*}{\partial \theta_k} .
$$

Then the gradient contribution is

$$
\frac{\partial \mathcal L}{\partial \theta_k}
= \frac{2}{\|u_{\text{obs}}\|_2^2}\,(u^*-u_{\text{obs}})^\top w_k .
$$

This approach requires one solve per parameter. The cost is
$
\mathcal O(P)
$
solves. When the parameter vector is large — for example, heterogeneous fields with
per-element parameters
$
(\alpha_k,\beta_k,\kappa_k)
$
— the total cost becomes prohibitive even if a factorization of
$
K
$
is reused for back-substitutions.

## 7.2 Adjoint method

Solve a single adjoint system

$$
K\,\lambda = \frac{2}{\|u_{\text{obs}}\|_2^2}\,(u^*-u_{\text{obs}}),
$$

and compute the gradient via

$$
\frac{\partial\mathcal L}{\partial \theta_k}
= -\lambda^\top\frac{\partial K}{\partial \theta_k}u^*
  + \lambda^\top\frac{\partial b}{\partial \theta_k} .
$$

In the present setting the optimization targets
$
(\alpha,\beta,\kappa)
$
and
$
b
$
does not depend on them, so the second term vanishes. Using the element-level
outer-product kernels constructed from the previous measurement rows,

$$
\begin{aligned} \frac{\partial \mathcal{L}}{\partial \alpha_k} & =-\frac{1}{4} V_k \sum_{\ell=0}^2\left(r_{\ell \ell}[k] \cdot u^\star\right)\left(r_{\ell \ell}[k] \cdot \lambda\right), \\ \frac{\partial \mathcal{L}}{\partial \beta_k} & =-\frac{1}{4} V_k \sum_{0 \leq \ell<m \leq 2}\left(r_{\ell m}[k] \cdot u^\star\right)\left(r_{\ell m}[k] \cdot \lambda\right), \\ \frac{\partial \mathcal{L}}{\partial \kappa_k} & =-V_k\left(r_{\mathrm{vol}}[k] \cdot u^\star\right)\left(r_{\mathrm{vol}}[k] \cdot \lambda\right) .\end{aligned}
$$



All operations reduce to local dot products; no additional linear solves are
required beyond the single adjoint solve.

## 7.3 Equivalence and cost

The two routes are mathematically equivalent. Eliminating
$w_k$
and
$\lambda$
yields

$$
= \frac{2}{\|u_{\text{obs}}\|_2^2}\,(u^*-u_{\text{obs}})^\top w_k
= \lambda^\top\!\left(\frac{\partial b}{\partial \theta_k}-\frac{\partial K}{\partial \theta_k}u^*\right),
$$

so both produce the same gradient. The practical difference is computational:

- Forward sensitivity: $\mathcal O(P)$ linear solves; best when the number of
  parameters is very small (e.g., a few global scalars $\alpha,\beta,\kappa$).
- Adjoint: $\mathcal O(1)$ linear solve plus elementwise inner products; best
for heterogeneous cases with many parameters.

## 7.4 Torch implementation 

The inverse routine proceeds as follows (in `hetero_cone_inverse.py`):

- Use the free-DOF stiffness once per iteration on CUDA as CSR for the
  current parameter fields set $\theta=\{\alpha, \beta, \kappa\}$.
- Restrict the observed displacement field to the free-DOF vector using the
  compressed `dof_map` to form `u_obs_free`.
- Precompute the normalization factor $c_{\text{rel}} = 2 / \|u_{\text{obs,free}}\|_2^2$
  (or reuse the cached denominator if $u_{\text{obs,free}}$ is fixed).
- Forward solve: compute `u*` as the solution of `K u = b` with the tolerance set
  by the caller and record the residual norm and a convergence flag.
- Adjoint solve: form the right-hand side $c_{\text{rel}}(u^* − u_{\text{obs,free}})$
  and solve $K \lambda = c_{\text{rel}}(u^* − u_{\text{obs,free}})$ with the same
  tolerance and reporting.
- Gradient evaluation per element `k` using the precomputed SMS rows:
  
1. $\partial L/\partial \alpha_k = −4 V_k \sum_{\ell} (r_{axis}[k,\ell]·u^*)(r_{axis}[k,\ell]·\lambda)$

2. $\partial L/ \partial \beta_k = −4 V_k Σ_s (r_{shear}[k,s]·u^*)(r_{shear}[k,s]·\lambda)$
  
3. $\partial L / \partial \kappa_k = −1 V_k (r_{vol}[k]·u^*)(r_{vol}[k]·\lambda)$

These are elementwise dot-products; no extra linear solves are required.

The matrix assemble/solve is reused across forward and adjoint solves in each
iteration; only the right-hand side changes.



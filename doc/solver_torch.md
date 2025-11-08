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


## 7.5 Inverse Problem with single param and log-TV regularization
We estimate the parameter field through **one scalar** (\alpha). The other SMS parameters are tied to (\alpha) by the constitutive relations

$$
\boxed{\ \beta=2\alpha,\qquad \kappa=\dfrac{4\nu}{1-2\nu},\alpha\ }\tag{7.0}
$$
so the global stiffness assembled in section 4 depends only on $(\alpha)$ via those substitutions. The forward displacement $(u^\star)$ is the solution of $K(\alpha),u^\star=b$.  
### 0 Use the previous deformation gradient F to represent the grad u in the regularization
Borrow the ideas from Computational Methods for Inverse Problems chapter 2:

A deformation is a map from material to spatial coordinates:

$$
x=\chi(X)=X+u(X) .
$$


Here $u(X)$ is the displacement field. The object that measures how $x$ changes with $X$ is the Jacobian (Fréchet derivative) of $\chi$ :

$$
F(X)=\frac{\partial \chi}{\partial X}(X)=\nabla_X x(X) .
$$


This matrix is called the deformation gradient.
Connect to the displacement gradient. Substituting $x=X+u(X)$ gives

$$
F(X)=\frac{\partial}{\partial X}(X+u(X))=I+\nabla_X u(X) .
$$


So the change in the deformation gradient relative to identity is exactly the gradient of displacement:

$$
F-I=\nabla_X u, \quad \delta F=\nabla_X(\delta u) .
$$


Any penalty or regularizer placed on "changes of $F^{\prime \prime}$ is therefore a penalty on $\nabla u$ (or on $\nabla \delta u$ for updates).

Now we will explain why we can use $D_s, D_m$ to represent the deformation gradient. 

Firstly relate $F$ to a first-order approximation. With Fréchet derivative, use Taylor expansion to represent the approximation of $\chi$ around $X$ is

$$
\chi(X+d X)=\chi(X)+D \chi(X) d X+r(d X) \text { with } \frac{\|r(d X)\|}{\|d X\|} \rightarrow 0 .
$$


Dropping the higher-order remainder yields the standard local relation

$$
\chi(X+d X)-\chi(X) \approx F(X) d X .
$$


Geometrically, $F$ maps an very small material line element $d X$ to its current image $d x$.

Then in our  discrete tetera mesh setting. On each tetrahedron, $\chi$ is affine s, so the previous approximation is exact inside the element. Introduce edge matrices built from one vertex  to the other three:

$$
D_m=\left[\begin{array}{lll}
  X_2-X_1 & X_3-X_1 & X_4-X_1
\end{array}\right],
$$


$$ 
D_s=\left[\begin{array}{lll}
x_2-x_1 & x_3-x_1 & x_4-x_1
\end{array}\right] .
$$


Given a tetrahedron with reference (material) vertices $X_1,X_2,X_3,X_4$, any point $X$ inside that tetrahedron can be written as an affine combination of the vertices:

$$X=\sum_{a=1}^{4}\lambda_a,X_a,\qquad \sum_{a=1}^{4}\lambda_a=1,\ \ \lambda_a\ge 0\ \text{(for points inside).}$$

The coefficients $\lambda_a$ are the local barycenter coordinates (). In the deformed (spatial) configuration, the corresponding point is $x=\sum_{a=1}^{4}\lambda_a,x_a,$
using the **same** $\lambda_a$ because linear shape functions $N_a\equiv\lambda_a$ map consistently from reference to current configuration.

Then we want to use a edge-basis $\zeta$ coordinates to prepresent $D_m, D_s$
For a compact matrix form, eliminate $\lambda_1$ via $\lambda_1=1-(\lambda_2+\lambda_3+\lambda_4)$ and define
$$
\zeta=\begin{bmatrix}\lambda_2\ \lambda_3\ \lambda_4\end{bmatrix}\in\mathbb{R}^3.
$$
Build the edge (column) matrices from vertex 1 to the other three vertices:
$$D_m=\big[X_2{-}X_1\ \ X_3{-}X_1\ \ X_4{-}X_1\big],\qquad
D_s=\big[x_2{-}x_1\ \ x_3{-}x_1\ \ x_4{-}x_1\big].$$
Then any point in the element can be written as

$$X = X_1 + D_m\zeta,\qquad x = x_1 + D_s\zeta.$$

Here $\zeta$ are the **local (edge-basis) coordinates**: they tell  how far to move from vertex 1 along the three reference edges to reach $X$; applying the same $\zeta$ to $D_s$ reaches $x$ in the deformed configuration.


So use barycentric coordinates $\zeta$, $X, x$ can be represented with:
$$
X=X_1+D_m \zeta, \quad x=x_1+D_s \zeta,
$$

differentiate and apply the chain rule:

$$
\frac{\partial x}{\partial X}=\frac{\partial x}{\partial \zeta} \frac{\partial \zeta}{\partial X}=D_s D_m^{-1} .
$$


Thus, for a nondegenerate linear tetrahedron, the deformation gradient is the unique linear map that sends the three material edges to their deformed images:

$$
F=D_s D_m^{-1}
$$


Simliarly for the axis-based shape function, with $x(X)=\sum_{a=1}^4 N_a(X) x_a$ and constant $\nabla_X N_a$,

$$
F=\sum_{a=1}^4 x_a \otimes \nabla_X N_a=I+\sum_{a=1}^4 u_a \otimes \nabla_X N_a=I+\nabla_X u,
$$

which matches previous $D_s \,D_m^{-1}$ expression.

Also, from this aspect, we can obtain the $\nabla_X N_a$ as the columns of $D_m^{-1}$ which will be use in the following inverse problem. Using local coordinates $X=X_1+D_m \zeta$ with $N_2=\zeta_1, N_3=\zeta_2, N_4=\zeta_3, N_1=1-\zeta_1-\zeta_2-\zeta_3$, we have $\frac{\partial \zeta}{\partial X}=D_m^{-1}$, so $\nabla N_2=\nabla \zeta_1=D_m^{-T} e_1$, etc, where $e_i$ are the standard basis vectors in $\mathbb{R}^3$.
Then
$$
\boxed{
\nabla N_2=D_m^{-T} e_1, \quad \nabla N_3=D_m^{-T} e_2, \quad \nabla N_4=D_m^{-T} e_3, \quad \nabla N_1=-\left(\nabla N_2+\nabla N_3+\nabla N_4\right)}
$$
We can define the volume with $D_m$ as weell: $V=\frac{1}{6}\det(D_m).$

### How to add to the previous adjoint problem
We have mesh: tetrahedra $T_k$, $k=1,\dots,M$, and the volume is $V_k$.
The foward solution has the displacement $u_{\text{sim}}\in\mathbb{R}^{3N_f}$(free DOFs).
On each tet $T_k$ with local nodes (a=1..4), linear shape‑function gradients $\nabla N_a^{(k)}\in\mathbb{R}^3$ are constant.
The Element deformation gradient of the simulation (minus identity) equals the displacement gradient
$$
  \nabla u_{\text{sim}, T_k}
  = \sum_{a=1}^{4} u_a\otimes\nabla N_a^{(k)}\in\mathbb{R}^{3\times 3}.
$$
Here $u_a\in\mathbb{R}^3$ are the nodal displacement vectors of the four nodes of $T_k$.

We regularize the deformation gradient named TV with Charbonnier smoothing:
$$

R(u_{\text{sim}})=\sum_{k=1}^M V_k\;\sqrt{\,|\nabla u_{\text{sim}}||_{F,k}^{2}+\varepsilon^{2}} , 
\newline
||\nabla u_{\text{sim}}||_{F,k}^{2}=||\nabla u_{\text{sim,}T_k}||_F^2 .
$$
We use $|,\cdot,|_F$ makes the TV penalty **isotropic** for the vector field (u): it does not prefer any coordinate axis or a particular component of $\nabla u$.
Discretely, $||\nabla u||_F^2$ is exactly the standard (H1) seminorm density (sum of squares of all spatial derivatives), so the Charbonnier version $\sqrt{|\nabla u|_F^2+\varepsilon^2}$ is a smooth, edge‑preserving analogue of that.

Then is the finding the regularization gradient towards u_sim with the mapping operation $G$

*Definition of $G$

Let $u_e=[u_1^\top,u_2^\top,u_3^\top,u_4^\top]^\top\in\mathbb{R}^{12}$ be the 4×3 DOFs of element T_k. Define the $R^{3×4}$ matrix of shape‑function gradients
$$
B_k = [\nabla N_1^{(k)}\ \ \nabla N_2^{(k)}\ \ \nabla N_3^{(k)}\ \ \nabla N_4^{(k)}].
$$
Then
$$
\nabla u_{\text{sim, }T_k} =
\sum_{i=1}^4 u_i\otimes\nabla N_i^{(k)}
= [-u_i-]B_k^\top=[u_1\ u_2\ u_3\ u_4]B_k^\top\in R^9 .
$$
Vectorize the 3*3  gradient (stack columns) to flatten to a 9‑vector:
$$
\underbrace{\operatorname{vec}\left(\nabla u_\text {sim， } T_k\right)}_{\in \mathbb{R}^9}=\underbrace{\left(B_k \otimes I_3\right)}_{:= G_k \in \mathbb{R}^{9 \times 12}} \quad\underbrace{\operatorname{vec}\left(\left[u_1 u_2 u_3 u_4\right]\right)}_{=u_e} .
$$
So the element gradient operator is
$$
G_k = (B_k\otimes I_3)
= \big[\ (\nabla N_1^{(k)}\otimes I_3)(\nabla N_2^{(k)}\otimes I_3)(\nabla N_3^{(k)}\otimes I_3)(\nabla N_4^{(k)}\otimes I_3)\ \big] \in\mathbb{R}^{9\times 12}.
$$

Note the $P_k\in\mathbb{R}^{12\times 3N_f}$ is the usual element‑to‑global gather which picks the 12 DOFs of element (k) from the global vector Stacking all elements (9 rows per element) gives the global sparse operator

$$\boxed{~
G=\left[\begin{array}{c}
G_1 P_1 \\
\vdots \\
G_M P_M
\end{array}\right] \in \mathbb{R}^{9 M \times 3 N_f}, \quad \operatorname{vec}\left(\nabla u_{\text {sim }}\right)=G u_{\text {sim }}{\in \mathbb{R}^{9 M}} .~}
$$



Write the per-element piece as

Define $S_k := \nabla u_{\text{sim}, T_k}$, 
$$
R_k(u_\text{sim})=\sqrt{\|S\|_F^2+\varepsilon} = \sqrt{tr(S_k^T S_k)+\varepsilon^2}
$$

Take the first variation. Let $R(S)=\sqrt{\operatorname{tr}\left(S^{\top} S\right)+\varepsilon^2}$. Then

$$
\mathrm{d} R(S)=\frac{1}{2 \sqrt{\operatorname{tr}\left(S^{\top} S\right)+\varepsilon^2}} \mathrm{~d} \operatorname{tr}\left(S^{\top} S\right)=\frac{1}{2 \sqrt{\operatorname{tr}\left(S^{\top} S\right)+\varepsilon^2}} 2 \operatorname{tr}\left(S^{\top} \mathrm{d} S\right)=\frac{\operatorname{tr}\left(S^{\top} \mathrm{d} S\right)}{\sqrt{\operatorname{tr}\left(S^{\top} S\right)+\varepsilon^2}} 
$$

So each for each $T_k$, 
$$
\delta R_k=V_k\frac{\operatorname{tr}\left(S_k^{\top} \delta S_k\right)}{\sqrt{\left\|S_k\right\|_F^2+\varepsilon^2}} .
$$



But $\delta S_k=\nabla\delta u_{sim, T_k}$ and with vectorization, $\operatorname{tr}\left(S_k^{\top} \delta S_k\right)=\operatorname{vec}\left(S_k\right)^{\top} \operatorname{vec}\left(\delta S_k\right)=\operatorname{vec}\left(S_k\right)^{\top} G_k \delta u_e$,  
where $u_e \in \mathbb{R}^{12}$ are the $4 \times 3$ DOFs of $T_k$

Therefore

$$
\delta R_k=\frac{V_k}{\sqrt{\left\|S_k\right\|_F^2+\varepsilon^2}}  \operatorname{vec}\left(S_k\right)^{\top} G_k \delta u_e .
$$


Sum over elements and scatter to global DOFs (use $u_e=P_k u_{\text {sim }}, \delta u_e=P_k \delta u$ ):

$$
\delta R=\sum_{k=1}^M \frac{V_k}{\sqrt{\left\|S_k\right\|_F^2+\varepsilon^2}} \operatorname{vec}\left(S_k\right)^{\top} G_k P_k \delta u=\left(\sum_{k=1}^M P_k^{\top} G_k^{\top} \frac{V_k}{\sqrt{\left\|S_k\right\|_F^2+\varepsilon^2}} \operatorname{vec}\left(S_k\right)\right)^{\top} \delta u .
$$


Hence the gradient w.r.t. $u_{\text {sim }}$ is

$$
\frac{\partial R}{\partial u_{\mathrm{sim}}}=\sum_{k=1}^M P_k^{\top} G_k^{\top} \frac{V_k}{\sqrt{\left\|S_k\right\|_F^2+\varepsilon^2}} \operatorname{vec}\left(S_k\right) .
$$


And $\operatorname{vec}\left(S_k\right)=G_k P_k u_{\text {sim }}$. Define the block-diagonal weight matrix
$W\left(u_{\text {sim }}\right)=\operatorname{blkdiag}\left(\frac{V_k}{\sqrt{\left\|S_k\right\|_F^2+\varepsilon^2}} I_9\right)_{k=1}^M.$

$$
r_{\mathrm{reg}} = \frac{\partial R}{\partial u_{\mathrm{sim}}}
= \sum_{k=1}^M P_k^{\top} G_k^{\top} \frac{V_k}{\sqrt{\left\|S_k\right\|_F^2+\varepsilon^2}} G_k P_k u_{\text {sim }} = G^{\top} W\left(u_{\mathrm{sim}}\right) G u_{\mathrm{sim}} .
$$
i.e.

$$
\boxed{
r_{\mathrm{reg}}:=\frac{\partial R}{\partial u_{\mathrm{sim}}}=G^{\top} W\left(u_{\mathrm{sim}}\right) G u_{\mathrm{sim}}} .
$$




Finally they can plug into the previous loss function: The total loss is
$$
\mathcal L = \frac{|u_{\text{sim}}-u_{\rm obs}|^2}{|u_{\rm obs}|^2+(\epsilon_{\rm rel}|u_{\rm obs}|)^2}*{\mathcal L*{\rm data}}
+\lambda_FR(u_{\text{sim}}),
$$
the adjoint right‑hand side is
$$
\frac{\partial \mathcal{L}}{\partial u_{\mathrm{sim}}}=\frac{2}{\left\|u_{\mathrm{obs}}\right\|^2+\left(\epsilon_{\mathrm{rel}}\left\|u_{\mathrm{obs}}\right\|\right)^2}\left(u_{\mathrm{sim}}-u_{\mathrm{obs}}\right)+r_{\mathrm{reg}}.
$$
We then solve $K(\alpha),\lambda = \partial \mathcal L/\partial u_{\text{sim}}$ (same (K) as in the forward step), and compute the **α‑gradient** with the existing SMS formulas plus the $alpha$‑only chain rule.


--- 

so called "Gaussian‑smoothed objective":

for $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$
$$
\mathbb{E} f(x+\varepsilon)=f(x)+\frac{1}{2} \operatorname{tr}(H(x) \Sigma)+O\left(\|\varepsilon\|^3\right),   .
$$

With $\Sigma=\sigma^2 I$,
$\mathbb{E} f(x+\varepsilon) \approx f(x)+\frac{\sigma^2}{2} \Delta f(x)$,

Also, the params are log transformed: $\theta = log(\alpha)$, $\theta$ add some variance may be negative.

Add the perturbations to the log-params  after an accepted L-BFGS step, then reset L-BFGS memory. Keep the closure fully deterministic. 

After optimizer.step(closure) (outside the closure), perturb theta:
theta += varepsilon. Reset lbfgs; the sigma value is set to small and decays to 0
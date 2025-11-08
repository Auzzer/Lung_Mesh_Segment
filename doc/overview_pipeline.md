## 1. Problem setup (global matrices)

Let the mesh have $N$ vertices, stacked as

$$
\mathbf x\in\mathbb R^{3N},\qquad
\mathbf X\in\mathbb R^{3N}\ \text{(reference)}.
$$

The image registration gives nodewise displacements $\mathbf u\in\mathbb R^{3N}$. Dirichlet boundary nodes are set to

$$
\mathbf x_D = \mathbf X_D + \mathbf u_D,
$$

and the solver finds the **free** DOFs $\mathbf x_F$ by static equilibrium.

We use the “intersection/spring” mass–spring model in matrix form. Precompute (from mesh + interpolation weights):


### Objects and stacking 

* Vertices (global DOFs):
  $\displaystyle \mathbf x\in\mathbb R^{3N}$ stacks $(x_{1x},x_{1y},x_{1z},\dots,x_{Nx},x_{Ny},x_{Nz})^\top$.

* For each tetra $t$ with global vertex indices $(i_0,i_1,i_2,i_3)$, stack its 4 vertex DOFs as
  $\displaystyle \mathbf x_t\in\mathbb R^{12}$.

* Intersection points inside tetra $t$: there are 6 of them (two per axis); stack as
  $\displaystyle \mathbf q_t\in\mathbb R^{18}$ (6 points × 3 coords).

* Three axial springs per tetra; their endpoint pairs among the 6 intersections are $(a_s,b_s)$, $s=1,2,3$.

* We stack all tets’ intersections as
  $\displaystyle \mathbf q = \big[\mathbf q_1^\top\;\cdots\;\mathbf q_M^\top\big]^\top \in \mathbb R^{18M}$,
  and all spring differences as
  $\displaystyle \Delta = \big[\Delta_1^\top\;\cdots\;\Delta_{3M}^\top\big]^\top \in \mathbb R^{3S},\ S=3M$.



Then use a distribution matrix $B$ mapping vertex to intersections

We already have per-tet interpolation coefficients $C_t\in\mathbb R^{6\times 4}$ such that, **for tetra $t$**,

$$
\mathbf q_t \;=\; (C_t\otimes I_3)\,\mathbf x_t .
$$

Let $P_t\in\{0,1\}^{12\times 3N}$ be the selection that extracts the 4 vertices of tetra $t$ (with xyz per vertex) from the global vector $\mathbf x$, i.e. $\mathbf x_t=P_t\,\mathbf x$. Then

$$
\mathbf q_t \;=\; (C_t\otimes I_3)\,P_t\,\mathbf x .
$$

Stacking all tets **by vertical concatenation** gives the **global** sparse operator

$$
\boxed{\;
B \;=\;
\begin{bmatrix}
(C_1\!\otimes I_3)P_1\\
\vdots\\
(C_M\!\otimes I_3)P_M
\end{bmatrix}
\;\in\;\mathbb R^{(18M)\times(3N)},\qquad
\mathbf q \;=\; B\,\mathbf x .
\;}
$$

In code we store $C_t$ as `self.C_t[t, vertex, point]` (shape $4\times 6$). In the math above we use $C_t$ as $6\times 4$. So

$$
C_t(j,a) \;=\; \texttt{self.C\_t}[t,\,a,\,j].
$$

When building $B$, for each tet $t$, for each intersection $j\in\{0,\dots,5\}$, and for each vertex $a\in\{0,\dots,3\}$, add the $3\times 3$ bloct $C_t(j,a)\,I_3$ at the row block of $(t,j)$ and the column block of global vertex $i_a$.

Also, we define a global incidence matrix $M$ mapping intersections to spring differences

For one spring $s$ inside tetra $t$, let $a_s,b_s\in\{0,\dots,5\}$ be its two endpoints in the 6 intersections. Define the **6-vector**

$$
d_s \;=\; e_{b_s}-e_{a_s}\in\mathbb R^{6},
$$

and its 3D lifting

$$
M_{t,s} \;=\; d_s^{\!\top}\otimes I_3 \;\in\; \mathbb R^{3\times 18},
\qquad
\Delta_{t,s} \;=\; M_{t,s}\,\mathbf q_t \;=\; \mathbf q_{b_s}-\mathbf q_{a_s}.
$$

Stack the three springs of tetra $t$ as

$$
M_t \;=\;
\begin{bmatrix}
d_1^{\!\top}\!\otimes I_3\\[2pt]
d_2^{\!\top}\!\otimes I_3\\[2pt]
d_3^{\!\top}\!\otimes I_3
\end{bmatrix}
\;\in\; \mathbb R^{9\times 18},
\qquad
\Delta_t \;=\; M_t\,\mathbf q_t .
$$

Finally, **block-diagonal** over tets:

$$
\boxed{\;
M \;=\; \operatorname{blkdiag}(M_1,\dots,M_M)\;\in\;\mathbb R^{(3S)\times(18M)},
\qquad
\Delta \;=\; M\,\mathbf q \;=\; M\,B\,\mathbf x .
\;}
$$

This is exactly the compact identity we used before:

$$
\boxed{\;A_0 \;=\; M\,B\ \in \mathbb R^{(3S)\times(3N)},\qquad \Delta \;=\; A_0\,\mathbf x.\;}
$$

---

## Dimension check

* $C_t$ is $6\times 4$, $P_t$ is $12\times 3N$ ⇒ $(C_t\!\otimes I_3)P_t$ is $18\times 3N$.
* Stack t$M$ as blkdiag of $M_t\in\mathbb R^{9\times 18}$ ⇒ $M\in\mathbb R^{(9M)\times(18M)}=(3S)\times(18M)$.
* Hence $B\in\mathbb R^{(18M)\times(3N)}$ and $A_0=MB\in\mathbb R^{(3S)\times(3N)}$.




* A **global distribution** $B\in\mathbb R^{(3S)\times(3N)}$ taking vertex DOFs to all spring endpoints differences via a global **incidence** $M$:

$$
\Delta \;=\; A_0\,\mathbf x,\qquad A_0:=M\,B\in\mathbb R^{(3S)\times(3N)}.
$$

Here $S$ is the number of axial springs (3 per tet), and $\Delta$ stacks all spring 3-vectors $\Delta_s=\mathbf q_{b_s}-\mathbf q_{a_s}$.

* For each spring $s$, define its **unit direction**

$$
\mathbf n_s \;=\; \frac{\Delta_s}{\|\Delta_s\|},\qquad
N_s=\mathbf n_s\mathbf n_s^\top\in\mathbb R^{3\times 3}.
$$

Stack the $3\times 3$ blocks into a block-diagonal projector

$$
\mathcal N \;=\; \operatorname{blkdiag}(N_1,\dots,N_S)\in\mathbb R^{(3S)\times(3S)}.
$$

* Stiffnesses $t_s>0$ (one per spring) define the block-diagonal

$$
\mathcal K \;=\; \operatorname{blkdiag}(k_1 I_3,\dots,k_S I_3)\in\mathbb R^{(3S)\times(3S)}.
$$

the free–free tangent and internal force

$$
K_{FF} \;=\; A_{0F}^\top(\mathcal K\,\mathcal N)\,A_{0F},\qquad
\mathbf f_{F} \;=\; A_{0F}^\top\,\mathbf y,
$$

where $A_{0F}$ is $A_0$ with columns restricted to free DOFs, and $\mathbf y\in\mathbb R^{3S}$ stacks per-spring endpoint forces

$$
\mathbf y \;=\; \operatorname{blkdiag}(k_1\mathbf n_1,\dots,k_S\mathbf n_S)\ \mathrm{ext},
\quad
\mathrm{ext}_s \;=\; \mathbf n_s^\top\Delta_s - L_{0,s}.
$$

(Here $L_{0,s}$ is the rest length along the axis.)

**Equilibrium** for the free DOFs is the nonlinear system

$$
\mathbf f_F(\mathbf x_F;\,\mathbf k)=\mathbf 0,
\qquad
\text{with}\quad
\mathbf f_F(\cdot)=A_{0F}^\top\,\mathbf y(\mathbf x).
$$

The simulation output is the full deformed configuration

$$
\mathbf x'_{\text{sim}} = \begin{bmatrix}\mathbf x_F^* \\ \mathbf x_D\end{bmatrix}
\quad\text{with }\mathbf x_F^* \text{ solving } \mathbf f_F(\mathbf x_F^*;\mathbf k)=\mathbf 0.
$$

---

## 2. Loss

Let ground-truth deformed positions be $\mathbf x'_{\text{gt}}$. Define

$$
\mathcal L(\mathbf x_F^*) \;=\; \tfrac12\big\|\, \!\big(\mathbf x'_{\text{sim}}-\mathbf x'_{\text{gt}}\big)\big\|^2.
$$

Only the free part depends on $\mathbf k$. Let $S_F$ be the column selector picking free DOFs from full length. Then

$$
\frac{\partial\mathcal L}{\partial \mathbf x_F}
\;=\;
S_F^\top \big(\mathbf x'_{\text{sim}}-\mathbf x'_{\text{gt}}\big)
\;=\;
:\ \mathbf g_L \in \mathbb R^{3N_F}.
$$

---

## 3. Implicit differentiation (adjoint)

We need $\dfrac{d\mathcal L}{d\mathbf k}$. The free solution satisfies $\mathbf f_F(\mathbf x_F^*;\mathbf k)=\mathbf 0$. Differentiate:

$$
\underbrace{\frac{\partial \mathbf f_F}{\partial \mathbf x_F}}_{K_{FF}}\,
\frac{d\mathbf x_F^*}{d\mathbf k}
\;+\;
\underbrace{\frac{\partial \mathbf f_F}{\partial \mathbf k}}_{f_k}
\;=\;\mathbf 0
\quad\Longrightarrow\quad
\frac{d\mathbf x_F^*}{d\mathbf k}
\;=\;
-\,K_{FF}^{-1}\,f_k.
$$

Chain rule gives

$$
\frac{d\mathcal L}{d\mathbf k}
\;=\;
\frac{\partial\mathcal L}{\partial \mathbf x_F}\,
\frac{d\mathbf x_F^*}{d\mathbf k}
\;=\;
-\,\mathbf g_L^\top K_{FF}^{-1}\,f_k.
$$

Introduce the **adjoint** $\boldsymbol\lambda\in\mathbb R^{3N_F}$ as the solution of

$$
K_{FF}^\top \boldsymbol\lambda \;=\; \mathbf g_L.
$$

Because $K_{FF}$ is symmetric, $K_{FF}^\top=K_{FF}$. Then the gradient collapses to a cheap inner product:

$$
\boxed{\;
\frac{d\mathcal L}{d\mathbf k}
\;=\;
-\,f_k^\top\,\boldsymbol\lambda.
\;}
$$

So we solve **one** linear system in the same matrix  already factorized for Newton, and then take a projected dot-product.

---

## 4. The partial $f_k$ at fixed $\mathbf x$

By definition $\mathbf f_F(\mathbf x)=A_{0F}^\top\,\mathbf y(\mathbf x,\mathbf k)$, with

$$
\mathbf y \;=\; \operatorname{blkdiag}(k_1\mathbf n_1,\dots,k_S\mathbf n_S)\ \mathrm{ext},
\qquad
\mathrm{ext}_s=\mathbf n_s^\top\Delta_s - L_{0,s}.
$$

When taking $\dfrac{\partial}{\partial \mathbf k}$ **at fixed $\mathbf x$**, the directions $\mathbf n_s$ and extensions $\mathrm{ext}_s$ are constants. Therefore

$$
\frac{\partial\mathbf y}{\partial k_s}
\;=\;
\begin{bmatrix}\mathbf 0\\ \vdots\\ \mathbf n_s\,\mathrm{ext}_s\\ \vdots\\ \mathbf 0\end{bmatrix}
\in\mathbb R^{3S},
\qquad
\Rightarrow\qquad
\boxed{\;
f_k \;=\; \frac{\partial \mathbf f_F}{\partial \mathbf k}
\;=\;
A_{0F}^\top\ \operatorname{blkdiag}\!\big(\mathbf n_1\mathrm{ext}_1,\dots,\mathbf n_S\mathrm{ext}_S\big).
\;}
$$

Plug into the adjoint gradient:

$$
\frac{d\mathcal L}{d\mathbf k}
\;=\;
-\Big[\operatorname{blkdiag}\!\big(\mathbf n_s\mathrm{ext}_s\big)\Big]^\top
\left(A_{0F}\,\boldsymbol\lambda\right).
$$

Written per spring $s$:

$$
\boxed{\;
\frac{d\mathcal L}{d k_s}
\;=\;
-\,\mathrm{ext}_s \;\mathbf n_s^\top\,\big(A_{0F}\,\boldsymbol\lambda\big)_s,
\quad
\text{with}\ \mathrm{ext}_s=\mathbf n_s^\top\Delta_s - L_{0,s},\ \Delta=A_0\,\mathbf x.
\;}
$$

This is extremely efficient: compute $z=A_{0F}\boldsymbol\lambda\in\mathbb R^{3S}$, reshape in 3-blocks $z_s\in\mathbb R^3$, then $d\mathcal L/dk_s= -\,\mathrm{ext}_s\,(\mathbf n_s\cdot z_s)$.

If you add a quadratic regularizer $\tfrac{\beta}{2}\|\mathbf k\|^2$, simply add $+\beta\,\mathbf k$ to the gradient.

---

## 5. From constant $\mathbf k$ to ANN parameters $\theta$

If you first **treat $\mathbf k$ as directly learnable constants**, then $\theta\equiv\mathbf k$ and the formula above already gives $\dfrac{d\mathcal L}{d\theta}$.

If instead $\mathbf k = \mathrm{ANN}(\Phi;\theta)$ (e.g. per-spring features $\Phi$ from $\mathbf X$, labels, region, etc.), apply the chain rule:

$$
\boxed{\;
\frac{d\mathcal L}{d\theta}
\;=\;
\Big(\frac{\partial \mathbf k}{\partial \theta}\Big)^\top
\frac{d\mathcal L}{d\mathbf k}.
\;}
$$

In practice you feed $\Phi$ through your network to produce $\mathbf k$, take $\dfrac{d\mathcal L}{d\mathbf k}$ from the adjoint above, and let your DL framework back-propagate $\left(\partial \mathbf k/\partial\theta\right)^\top$ automatically.

---

## 6. One Newton/adjoint step (algorithmic summary)

1. Build $A_0$ once; extract $A_{0F}$ by selecting free columns.

2. Given current $\mathbf x$ (with $\mathbf x_D$ set), compute

$$
\Delta = A_0\,\mathbf x,\quad
\mathbf n_s = \Delta_s/\|\Delta_s\|,\quad
\mathrm{ext}_s = \mathbf n_s^\top\Delta_s - L_{0,s}.
$$

3. Assemble $K_{FF} = A_{0F}^\top\, \operatorname{blkdiag}(k_s\,\mathbf n_s\mathbf n_s^\top)\,A_{0F}$ and

$$
\mathbf r_F = A_{0F}^\top\,\operatorname{blkdiag}(k_s\mathbf n_s)\,\mathrm{ext}.
$$

4. Newton step for equilibrium:

$$
K_{FF}\,\Delta \mathbf x_F = -\,\mathbf r_F,\qquad
\mathbf x_F \leftarrow \mathbf x_F + \Delta \mathbf x_F.
$$



1. With converged $\mathbf x_F^*$, compute the loss $\mathcal L$ and its free-DOF gradient $\mathbf g_L = \partial\mathcal L/\partial \mathbf x_F$.

2. **Adjoint solve** $K_{FF}\,\boldsymbol\lambda = \mathbf g_L$.

3. **Gradient in stiffness space**:

$$
\frac{d\mathcal L}{d k_s} = -\,\mathrm{ext}_s\;\mathbf n_s^\top\,(A_{0F}\boldsymbol\lambda)_s
\quad(\text{plus } \beta k_s \text{ if regularized}).
$$

8. If $\mathbf k=\mathrm{ANN}(\Phi;\theta)$, back-propagate:

$$
\frac{d\mathcal L}{d\theta}
= \Big(\frac{\partial \mathbf k}{\partial \theta}\Big)^\top \frac{d\mathcal L}{d\mathbf k}.
$$

This yields exact gradients under the small-strain tangent (no need to unroll solver steps) and costs **one** linear solve for $\boldsymbol\lambda$ per loss, reusing the factorization of $K_{FF}$.



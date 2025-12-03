# Inverse Problem with single param and log-TV regularization

We estimate the parameter field through **one scalar** (\alpha). The other SMS parameters are tied to (\alpha) by the constitutive relations

$$
\boxed{\ \beta=2\alpha,\qquad \kappa=\dfrac{4\nu}{1-2\nu},\alpha\ }\tag{7.0}
$$
so the global stiffness assembled in section 4 depends only on $(\alpha)$ via those substitutions. The forward displacement $(u^\star)$ is the solution of $K(\alpha),u^\star=b$.  
## 1. Use the previous deformation gradient F to represent the grad u in the regularization
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

### 2.  add to the previous adjoint problem
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
\mathcal L = \frac{\|u_{\text{sim}}-u_{\rm obs}\|_2^2}{\|u_{\rm obs}\|_2^2+(\epsilon_{\rm rel}\|u_{\rm obs}\|_2)^2}*{\mathcal L*{\rm data}}
+\lambda_FR(u_{\text{sim}}),
$$
the adjoint right‑hand side is
$$
\frac{\partial \mathcal{L}}{\partial u_{\mathrm{sim}}}=\frac{2}{\left\|u_{\mathrm{obs}}\right\|_2^2+\left(\epsilon_{\mathrm{rel}}\left\|u_{\mathrm{obs}}\right\|_2\right)^2}\left(u_{\mathrm{sim}}-u_{\mathrm{obs}}\right)+r_{\mathrm{reg}}.
$$
We then solve $K(\alpha),\lambda = \partial \mathcal L/\partial u_{\text{sim}}$ (same (K) as in the forward step), and compute the **α‑gradient** with the existing SMS formulas plus the $\alpha$‑only chain rule.

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



## 2. Useing deformation gradient for alpha regulatization
From the real-world data experiments, we find that using the deformation gradient for alpha regularization is not good. This is because the deformation gradient $\nabla \Delta u$ is very mall. This matches the previous section's model assumption: all the strain in our simulation is small. So we modify the previous log-TV regularization for alpha.



### 2.1 From $R(\mu)$ (in Warp) to $R(\alpha)$

In warp, the continuum regularization form on $\mu$:
$$
R(\mu)=\int_{\Omega}\sqrt{\frac{||\nabla\mu||^2}{(\mu+\varepsilon_{\rm div})^2}+\varepsilon_{\rm reg}^2}\,{\rm d}V. \tag{3.1}
$$

Substitute $\mu=2\alpha$,we have $\nabla\mu=2\nabla\alpha$. So
$$
\frac{||\nabla\mu||^2}{(\mu+\varepsilon_{\rm div})^2}
=\frac{4||\nabla\alpha||^2}{(2\alpha+\varepsilon_{\rm div})^2}
=\frac{||\nabla\alpha||^2}{\big(\alpha+\tfrac12\varepsilon_{\rm div}\big)^2}.
$$
Thus the continuum functional written in $\alpha$ is
$$
R(\alpha)=\int_{\Omega}\sqrt{\frac{||\nabla\alpha||^2}{\big(\alpha+\tfrac12\varepsilon_{\rm div}\big)^2}+\varepsilon_{\rm reg}^2}\,{\rm d}V. \tag{3.2}
$$

**FE discretization**

On each tetrahedron $T_k$ (volume $V_k$) the linear‐FE gradient is constant:
$$
\nabla_{T_k}\alpha=\sum_{i=1}^4 \alpha_i\,\nabla N_i^{(k)}.
$$
Evaluating $\alpha$ at the barycenter gives the element average
$\bar\alpha_k=\tfrac14\sum_{i=1}^4\alpha_i$.
Using one‑point (barycentric) quadrature, the volume integral becomes a sum of element contributions, each equal to *$V_k$ times* the integrand evaluated with $\nabla_{T_k}\alpha$ and $\bar\alpha_k$:
$$
\boxed{
R(\alpha)\approx\sum_{k=1}^{M}V_k
\sqrt{\frac{||\nabla_{T_k}\alpha||^2}{\big(\bar\alpha_k+\tfrac12\varepsilon_{\rm div}\big)^2}
+\varepsilon_{\rm reg}^2}.} \tag{3.3}
$$
Equivalently—in exactly the same form as Warp, we can keep the denominator as $2\bar\alpha_k+\varepsilon_{\rm div}$:
$$
\sum_k V_k\sqrt{\frac{4||\nabla_{T_k}\alpha||^2}{(2\bar\alpha_k+\varepsilon_{\rm div})^2}+\varepsilon_{\rm reg}^2}
=\sum_k V_k\sqrt{\frac{||\nabla_{T_k}\alpha||^2}{(\bar\alpha_k+\tfrac12\varepsilon_{\rm div})^2}+\varepsilon_{\rm reg}^2}. \tag{3.4}
$$



###  2.2 discrete gradient

Define the per‑element contribution
$$
R_k(\alpha):=V_k\sqrt{\frac{||\nabla_{T_k}\alpha||^2}{\big(\bar\alpha_k+\tfrac12\varepsilon_{\rm div}\big)^2}
+\varepsilon_{\rm reg}^2}. \tag{3.5}
$$
We differentiate $R_k$ with respect to the four nodal values $\alpha_1,\dots,\alpha_4$ on $T_k$.

1.  Differentiate the square‑root.
Let
$$
A_k:=\frac{||\nabla_{T_k}\alpha||^2}{\big(\bar\alpha_k+\tfrac12\varepsilon_{\rm div}\big)^2}.
$$
Then
$$
\delta R_k=V_k\cdot\frac{1}{2\sqrt{A_k+\varepsilon_{\rm reg}^2}}\;\delta A_k. \tag{3.6}
$$

2.  Differentiate $A_k$.

Use the product/quotient rule :
$$
\delta A_k
=\frac{2\,\nabla_{T_k}\alpha\cdot\nabla_{T_k}(\delta\alpha)}{\big(\bar\alpha_k+\tfrac12\varepsilon_{\rm div}\big)^2}
-\frac{2\,||\nabla_{T_k}\alpha||^2}{\big(\bar\alpha_k+\tfrac12\varepsilon_{\rm div}\big)^3}\,\delta\bar\alpha_k. \tag{3.7}
$$

3.  Express the variations in nodal increments $\delta\alpha_a$.

For linear tets,
$$
\nabla_{T_k}(\delta\alpha)=\sum_{a=1}^4 \delta\alpha_a\,\nabla N_a^{(k)},\qquad
\delta\bar\alpha_k=\tfrac14\sum_{a=1}^4 \delta\alpha_a. \tag{3.8}
$$
Insert (3.8) into (3.7) and then into (3.6), and collect the coefficients of each $\delta\alpha_a$:
$$
\delta R_k
=\sum_{a=1}^4
\left[
V_k\left(
\frac{\nabla_{T_k}\alpha\cdot\nabla N_a^{(k)}}{\big(\bar\alpha_k+\tfrac12\varepsilon_{\rm div}\big)^{2}
\sqrt{\dfrac{||\nabla_{T_k}\alpha||^2}{\big(\bar\alpha_k+\tfrac12\varepsilon_{\rm div}\big)^2}
+\varepsilon_{\rm reg}^2}}
-\frac{||\nabla_{T_k}\alpha||^2}{\big(\bar\alpha_k+\tfrac12\varepsilon_{\rm div}\big)^{3}
\sqrt{\dfrac{||\nabla_{T_k}\alpha||^2}{\big(\bar\alpha_k+\tfrac12\varepsilon_{\rm div}\big)^2}
+\varepsilon_{\rm reg}^2}}\cdot\frac14
\right)
\right]\delta\alpha_a. 
$$

From this we can give the element gradient vector (four entries, one per node):
$$
\boxed{
\frac{\partial R_k}{\partial \alpha_e^{(k)}}=
V_k\left(
\frac{\begin{bmatrix}
\nabla_{T_k}\alpha\cdot\nabla N_1^{(k)}\\
\nabla_{T_k}\alpha\cdot\nabla N_2^{(k)}\\
\nabla_{T_k}\alpha\cdot\nabla N_3^{(k)}\\
\nabla_{T_k}\alpha\cdot\nabla N_4^{(k)}
\end{bmatrix}}
{\big(\bar\alpha_k+\tfrac12\varepsilon_{\rm div}\big)^{2}
\sqrt{\dfrac{||\nabla_{T_k}\alpha||^2}{\big(\bar\alpha_k+\tfrac12\varepsilon_{\rm div}\big)^2}
+\varepsilon_{\rm reg}^2}}
-\frac{||\nabla_{T_k}\alpha||^2}{\big(\bar\alpha_k+\tfrac12\varepsilon_{\rm div}\big)^{3}
\sqrt{\dfrac{||\nabla_{T_k}\alpha||^2}{\big(\bar\alpha_k+\tfrac12\varepsilon_{\rm div}\big)^2}
+\varepsilon_{\rm reg}^2}}\;\frac{\mathbf 1}{4}
\right).
} 
$$
As usual, assemble $\partial R/\partial\alpha$ by scattering these four entries to the global vector and summing over all tets.

**Per‑tet unknowns.** If the optimization parameters are per‑tet $\alpha_k$ but we can evaluate $\nabla_{T_k}\alpha$ using vertex values (e.g., local averaging to nodal values before applying ), just apply the chain rule: propagate the nodal gradient back to incident tets using the same averaging weights used to create nodal $\alpha$ from $\alpha_k$.



### 2.3 Using it in the objective

With $\omega>0$ the regularization weight, the objective is
$$
\mathcal L(u^\star,\alpha)=\frac{\|u^\star-u_{\rm obs}\|_2^2}{\|u_{\rm obs}\|_2^2}+\omega\,R(\alpha),
$$
and the gradient w.r.t. $\alpha$ is the existing data‑term expression plus $\omega$ times the assembled vector. The adjoint system for $u^\star$ is unchanged because the regularizer depends only on $\alpha$. The adjoint equation is identical to the original one; only the $\alpha$ gradient gets an extra $\omega \partial R / \partial \alpha$ term.


Set up the Lagrangian with the PDE constraint $K(\alpha) u^{\star}=b$ and use $\omega>0$ for the regularization weight on $\alpha$ :

$$
\Phi\left(u^{\star}, \alpha, \lambda\right)=\underbrace{\frac{\left\|u^{\star}-u_{\text {obs }}\right\|_2^2}{\left\|u_{\text {obs }}\right\|_2^2}}_{\mathcal{L}_{\text {data }}\left(u^{\star}\right)}+\omega R(\alpha)-\lambda^{\top}\left(K(\alpha) u^{\star}-b\right) .
$$

- Variation w.r.t. $u^{\star}$ :

$$
\delta_{u^{\star}} \Phi=\frac{2}{\left\|u_{\mathrm{obs}}\right\|_2^2}\left(u^{\star}-u_{\mathrm{obs}}\right)^{\top} \delta u^{\star}-\lambda^{\top} K(\alpha) \delta u^{\star} .
$$


Since this must vanish for all $\delta u^{\star}$, the adjoint satisfies

$$
K(\alpha)^{\top} \lambda=\frac{2}{\left\|u_{\mathrm{obs}}\right\|_2^2}\left(u^{\star}-u_{\mathrm{obs}}\right) .
$$


For linear elasticity $K(\alpha)$ is symmetric, so the adjoint solve is exactly

$$
K(\alpha) \lambda=\frac{2}{\left\|u_{\mathrm{obs}}\right\|_2^2}\left(u^{\star}-u_{\mathrm{obs}}\right) .
$$


If the data term uses a slightly different normalization (e.g., $\left\|u_{\text {obs }}\right\|_2^2+\left(\epsilon_{\text {rel }}\left\|u_{\text {obs }}\right\|_2\right)^2$ ), just replace the denominator accordingly. The key point is that $R(\alpha)$ contributes no $\partial R / \partial u^{\star}$, so the adjoint RHS is unchanged.
- Variation w.r.t. $\lambda$ :

$$
\delta_\lambda \Phi=-\left(K(\alpha) u^{\star}-b\right)^{\top} \delta \lambda \quad \Rightarrow \quad K(\alpha) u^{\star}=b
$$

(the forward equation, unchanged).
- Variation w.r.t. $\alpha$ :

$$
\delta_\alpha \Phi=\omega \frac{\partial R}{\partial \alpha} \cdot \delta \alpha-\lambda^{\top}\left(\frac{\partial K}{\partial \alpha} u^{\star}\right) \cdot \delta \alpha .
$$


Thus the gradient of back-propagate to $\alpha$ is

$$
\frac{\partial \mathcal{L}}{\partial \alpha}=-\lambda^{\top} \frac{\partial K}{\partial \alpha} u^{\star}+\omega \frac{\partial R}{\partial \alpha} .
$$


So, only the $\alpha$-update picks up the new regularizer term; the adjoint linear system and its RHS stay the same.

## 3. Weighted combined regularizations

We now use **both** the deformation-gradient TV regularizer on the displacement and the log-TV regularizer on the parameter field $\alpha$ in a single PDE-constrained objective. Recall

* The displacement regularizer $R_u(u_{\text{sim}})$ from Section 1 with gradient
  $$
  \frac{\partial R_u}{\partial u_{\text{sim}}}
  = r_{\mathrm{reg}}(u_{\text{sim}})
  = G^\top W(u_{\text{sim}}) G u_{\text{sim}}.
  $$
* The log-TV regularizer $R_\alpha(\alpha)$ from Section 2, with per-element contributions $R_k(\alpha)$ and element gradient $\partial R_k / \partial \alpha_e^{(k)}$ assembled into the global vector $\partial R_\alpha/\partial\alpha$.

Here we denote the forward displacement as $u^\star$, which we can identify with $u_{\text{sim}}$.

### 3.1 Objective with two weights

Let the (normalized) data misfit be
$$
\mathcal L_{\text{data}}(u^\star)
= \frac{\|u^\star-u_{\rm obs}\|_2^2}
{\|u_{\rm obs}\|_2^2
+ \big(\epsilon_{\rm rel}\|u_{\rm obs}\|_2\big)^2},
$$
or any of the equivalent normalizations used earlier (the denominator is a fixed constant).

Introduce two positive weights:

* $\omega_1>0$ for the displacement regularizer $R_u$,
* $\omega_2>0$ for the parameter regularizer $R_\alpha$.

The combined objective is
$$
\boxed{
\mathcal L(u^\star,\alpha)
= \mathcal L_{\text{data}}(u^\star)
+ \omega_1 R_u(u^\star)
+ \omega_2 R_\alpha(\alpha).
  }
$$

The forward displacement still satisfies the PDE constraint
$$
K(\alpha) u^\star = b.
$$

### 3.2 Lagrangian and adjoint equation

Introduce the Lagrangian with adjoint variable $\lambda$:
$$
\Phi(u^\star,\alpha,\lambda)
= \mathcal L_{\text{data}}(u^\star)
+ \omega_1 R_u(u^\star)
+ \omega_2 R_\alpha(\alpha)
- \lambda^\top\big(K(\alpha)u^\star - b\big).
$$
The gradient of the data term is
$$
\frac{\partial \mathcal L_{\text{data}}}{\partial u^\star}
= c\,(u^\star - u_{\rm obs}),
\qquad
c := \frac{2}
{\|u_{\rm obs}\|_2^2
+ \big(\epsilon_{\rm rel}\|u_{\rm obs}\|_2\big)^2}.
$$

Using the displacement regularizer gradient
$$
\frac{\partial R_u}{\partial u^\star}
= r_{\mathrm{reg}}(u^\star)
= G^\top W(u^\star) G u^\star,
$$
the variation of $\Phi$ with respect to $u^\star$ is
$$
\delta_{u^\star}\Phi
= \big(
 c\,(u^\star - u_{\rm obs})
+ \omega_1 r_{\mathrm{reg}}(u^\star)
- K(\alpha)^\top \lambda
  \big)^\top \delta u^\star.
$$

Since this must vanish for all $\delta u^\star$, the adjoint satisfies
$$
K(\alpha)^\top \lambda
= c\,(u^\star - u_{\rm obs})
+ \omega_1\,r_{\mathrm{reg}}(u^\star).
$$

For linear elasticity the stiffness is symmetric, $K(\alpha)^\top=K(\alpha)$, so the adjoint solve is
$$
\boxed{
K(\alpha)\lambda
= c\,(u^\star - u_{\rm obs})
+ \omega_1\,G^\top W(u^\star) G u^\star.
  }
$$

Variation with respect to $\lambda$ recovers the forward problem,
$$
\delta_\lambda\Phi
= -\big(K(\alpha)u^\star - b\big)^\top \delta\lambda
\quad\Rightarrow\quad
\boxed{K(\alpha)u^\star = b.}
$$

### 3.3 Gradient with respect to $\alpha$

The displacement regularizer $R_u(u^\star)$ does not depend on $\alpha$, so only the PDE and $R_\alpha(\alpha)$ contribute to $\delta_\alpha\Phi$:
$$
\delta_\alpha\Phi
= \omega_2\,\frac{\partial R_\alpha}{\partial\alpha}\cdot\delta\alpha
- \lambda^\top\Big(\frac{\partial K}{\partial\alpha}u^\star\Big)\cdot\delta\alpha.
$$

Thus the gradient that is back-propagated to $\alpha$ is
$$
\boxed{
\frac{\partial\mathcal L}{\partial\alpha}
= -\,\lambda^\top\frac{\partial K}{\partial\alpha}u^\star
+ \omega_2\,\frac{\partial R_\alpha}{\partial\alpha}.
  }
$$

Here $\partial R_\alpha/\partial\alpha$ is exactly the global vector assembled from the per-element gradients $\partial R_k/\partial \alpha_e^{(k)}$ derived in Section 2.2.

If the optimization variables are the log-parameters $\theta = \log\alpha$, apply the chain rule componentwise:
$$
\frac{\partial\mathcal L}{\partial\theta_i}
= \alpha_i\,\frac{\partial\mathcal L}{\partial\alpha_i},
$$
or in vector form
$$
\boxed{
\frac{\partial\mathcal L}{\partial\theta}
= \alpha \odot \frac{\partial\mathcal L}{\partial\alpha},
}
$$
where $\odot$ denotes elementwise (Hadamard) product.

### 3.4 Summary for implementation

Given $\alpha$ (or $\theta = \log\alpha$) and weights $\omega_1,\omega_2$:

1. Solve the forward problem ($K(\alpha)u^\star = b$).
2. Assemble the data-term gradient ($g_{\text{data}} = c\,(u^\star - u_{\rm obs})$).
3. Assemble the displacement regularizer gradient ($r_{\mathrm{reg}}(u^\star) = G^\top W(u^\star)G u^\star$).
4. Solve the adjoint system ($K(\alpha)\lambda = g_{\text{data}} + \omega_1 r_{\mathrm{reg}}(u^\star)$).
5. Assemble the parameter-regularizer gradient ($\partial R_\alpha/\partial\alpha$) using the per-tet expressions and scatter.
6. Form the final gradient
   $$
   \frac{\partial\mathcal L}{\partial\alpha}
   = -\,\lambda^\top\frac{\partial K}{\partial\alpha}u^\star
   + \omega_2\,\frac{\partial R_\alpha}{\partial\alpha},
   $$
   and, if needed, convert to $\partial\mathcal L/\partial\theta$ via the chain rule above.

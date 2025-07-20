## 2.4 Modeling Approach

### 2.4.1 Intersection Points & Coefficient Matrix

In each tetrahedral element $\mathcal V_k$, we locate six intersection points $q_{j}$ by ray-casting from the **barycenter** $x_b$ along the three anisotropy axes to the faces.  At the same time we build a $4\times6$ coefficient matrix $C^k$ whose entries let us reconstruct any intersection $q_j$ from the four vertex positions.

1. **Barycenter**

   $$
   x_b \;=\;\frac{1}{4}\sum_{i=1}^4 x_i
   \tag{2.22}
   $$

   where $x_i$ are the four vertex coordinates.

2. **Point-in-triangle test & barycentric coords**
   A traced point $q_j$ on face $\Delta_{i_1i_2i_3}$ is inside iff

   $$
   S_{\Delta_{i_1i_2i_3}}
   \;=\;
   S_{\Delta_{q_j\,i_2\,i_3}}
   +S_{\Delta_{i_1\,q_j\,i_3}}
   +S_{\Delta_{i_1\,i_2\,q_j}}.
   \tag{2.23}
   $$

   Then its local (area) coordinates on that triangle are

   $$
   \xi=\frac{S_{\Delta_{q_j\,i_2\,i_3}}}{S_{\Delta_{i_1i_2i_3}}},\quad
   \eta=\frac{S_{\Delta_{q_j\,i_1\,i_3}}}{S_{\Delta_{i_1i_2i_3}}},
   \quad
   1-\xi-\eta=\frac{S_{\Delta_{i_1\,i_2\,q_j}}}{S_{\Delta_{i_1i_2i_3}}}.
   \tag{2.24}
   $$

3. **Building the coefficient matrix $C^k$**
   For each intersection $q_j$ we evaluate the four linear shape-functions $N_i$ of the tetrahedron’s nodes $i=1\ldots4$.  On the face containing $q_j$, those coincide with the barycentric coordinates:

   $$
   \begin{cases}
     N_{i_1}(q_j)=1-\xi-\eta,\\
     N_{i_2}(q_j)=\xi,\\
     N_{i_3}(q_j)=\eta,\\
     N_{i_4}(q_j)=0,
   \end{cases}
   $$

   where $\{i_1,i_2,i_3\}$ are the face nodes and $i_4$ is the opposite vertex.  We then set

   $$
   C^k_{ij}\;=\;N_i\bigl(q_j\bigr),
   $$

   assembling a $4\times6$ matrix whose $j$th column holds the four shape-function values at $q_j$.

4. **Updating intersections**
   At runtime, once the current vertex positions $x^t_i$ are known, each intersection moves as

   $$
   x^t_j
   \;=\;\sum_{i=1}^4 C^k_{ij}\;x^t_i
   \tag{2.25}
   $$

   reproducing the straight-sided mapping of a linear tetrahedron.

![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-13.jpg?height=664&width=1286&top_left_y=779&top_left_x=374)

Fig. 2.6 Intersection points in a tetrahedral volume element: The tetrahedron with three axes of anisotropy set at the barycenter and the six intersection points that they define (a), a triangular face of the element containing an intersection point and the coefficients $\xi_{0}$ and $\eta_{0}$ related to the intersection point. Note that $\xi$ increases with the cyan color gradient starting from $\xi=0$ at the line segment ( $p_{1}, p_{2}$ ) and is equal to $\xi=1$ at $p_{3}$, while $\eta$ increases along the orange color gradient starting from $\eta=0$ at ( $p_{2}, p_{3}$ ) until it reaches $\eta=1$ at $p_{1}$ (b).

---

### 2.4.2 Internal Forces

Internal (“deformation”) forces in each tetrahedron are computed by **three axial springs** along the anisotropy axes, plus **three torsion springs** coupling each pair of axes.  See Fig. 2.12.
the angle $\alpha_{\mathrm{lm}}^{t}$ between the axes $\zeta_{\mathrm{l}}$ and $\zeta_{\mathrm{m}}$ can be given by
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-22.jpg?height=720&width=863&top_left_y=394&top_left_x=551)

Fig. 2.12 A tetrahedron with three axial springs (in cyan) along the axes of anisotropy and three torsion springs in the barycenter of the tetrahedron (in violet).
#### 2.4.2.1 Axial Springs

* **Axis vectors**
  Along axis $\ell\in\{1,2,3\}$, let the two intersection points be $q_{\ell,1}$ and $q_{\ell,2}$.  Their current axis vector and length are

  $$
    \zeta_\ell^t
    = x^t_{q_{\ell,1}} - x^t_{q_{\ell,2}},
     .
  $$
* **Initial length** (at $t=0$):

  $$
    l^0_\ell
    = \|\zeta_\ell^0\|
    = \bigl\|\,x^0_{q_{\ell,1}}-x^0_{q_{\ell,2}}\,\bigr\|.
    \tag{2.30}
  $$
* **Unit direction**:

  $$
    \hat\zeta_\ell^t
    = \frac{\zeta_\ell^t}{\|\zeta_\ell^t\|}.
    \tag{2.31}
  $$
* **Hooke’s law** (linear axial force):

  $$
    \boxed{
      f^{t}_{\ell,\,\mathrm{axial}}
      = -\,k_\ell\bigl(\|\zeta_\ell^t\| - \|\zeta_\ell^0\|\bigr)\,\hat\zeta_\ell^t
    }
    \tag{2.35}
  $$

  where $k_\ell$ is the stiffness constant.

---

#### 2.4.2.2 Torsion Springs

To capture bending resistance between each pair of anisotropy axes in a tetrahedron, we introduce **torsion springs**. These springs penalize deviations of the angles between axes from their rest values.

---

##### 1. Angle between two axes

For any two axes $\ell$ and $m$, the angle is

$$
  \alpha^t_{\ell m}
  = \arccos\bigl(\hat\zeta_\ell^t \!\cdot\! \hat\zeta_m^t\bigr),
  \quad
  \alpha^0_{\ell m} = \alpha_{\ell m}^{t=0}
  \tag{2.32}
$$

- $\hat\zeta_\ell^t$ and $\hat\zeta_m^t$ are the unit–direction vectors at time $t$.  
- $\alpha^0_{\ell m}$ is the **rest angle**, measured in the undeformed configuration.

---

##### 2. Decomposing the torsion force

At each intersection point on axis $\ell$, the net torsion force $f_{\ell,1}$ splits into three orthogonal components:

$$
  f_{\ell,1}
  = f_S(\zeta_\ell,\alpha_{\ell m},\alpha_{\ell n})\,\hat\zeta_\ell
  + f_\tau(\zeta_\ell,\alpha_{\ell m},\alpha_{\ell n})\,\hat\zeta_m
  + f_\tau(\zeta_\ell,\alpha_{\ell m},\alpha_{\ell n})\,\hat\zeta_n,
  \tag{2.33}
$$

with

$$
  f_{\ell,2} = -\,f_{\ell,1},
  \quad
  \{m,n\}  \text{are the other two axes.}
$$

- **Axial** component $f_S$ acts along $\hat\zeta_\ell$.  
- **Torsional** components $f_\tau$ lie in the planes $(\hat\zeta_\ell,\hat\zeta_m)$ and $(\hat\zeta_\ell,\hat\zeta_n)$.

---

###### 2a. Expressions for $f_S$ and $f_\tau$

We derive both from simple spring energies aspect like: $f_S = -\,\dfrac{\mathrm dU_S}{\mathrm d\|\zeta_\ell\|}$

**:  
   In a conservative spring model, the force along a single coordinate $x$ is the negative derivative of its potential energy:

   $$
     F(x) = -\,\frac{\mathrm d}{\mathrm d x}\,U(x).
   $$

   Here our “coordinate” is the current length $\|\zeta_\ell\|$, so the axial force magnitude is

   $$
     f_S = -\,\frac{\mathrm d}{\mathrm d \|\zeta_\ell\|}\,U_S.
   $$

1. **Axial term**

   Define

   $$
     U_S
     = \tfrac12\,k_\ell\bigl(\|\zeta_\ell^t\| - \|\zeta_\ell^0\|\bigr)^2.
   $$

   Then

   $$
     f_S
     = -\,\frac{\mathrm d}{\mathrm d \|\zeta_\ell\|}
       \Bigl[\tfrac12\,k_\ell(\|\zeta_\ell\|-\|\zeta_\ell^0\|)^2\Bigr]
     = -\,k_\ell\bigl(\|\zeta_\ell^t\|-\|\zeta_\ell^0\|\bigr),
   $$

   and the vector is $\mathbf f_S = f_S\,\hat\zeta_\ell$.

2. **Torsional terms**

   Define

   $$
     U_\tau
     = \tfrac12\sum_{p\in\{m,n\}}
       k_{\ell p}\,\bigl(\alpha^t_{\ell p}-\alpha^0_{\ell p}\bigr)^2.
   $$

   Differentiating with respect to each angle gives

   $$
     f_\tau(\zeta_\ell,\alpha_{\ell m},\alpha_{\ell n})
     = -\,k_{\ell m}\,\bigl(\alpha^t_{\ell m}-\alpha^0_{\ell m}\bigr),
   $$

   and similarly for $(\ell,n)$.

---

##### 3. Linear torsion‐spring model

$$
  \boxed{
    f^t_{\ell\to m}
    = -\,k_{\ell m}\,\bigl(\alpha^t_{\ell m}-\alpha^0_{\ell m}\bigr)\,\hat\zeta_m^t,
  }
  \quad
  f^t_{m\to \ell}=-\,f^t_{\ell\to m}.
  \tag{2.40–2.41}
$$

---

##### 4. Cosine‐approximation (small‐angle)

When axes remain near orthogonal,

$$
  \alpha^t_{\ell m}-\alpha^0_{\ell m}
  \approx (\hat\zeta_\ell^t\!\cdot\!\hat\zeta_m^t)
         - (\hat\zeta_\ell^0\!\cdot\!\hat\zeta_m^0).
$$

Thus

$$
  \boxed{
    f^t_{\ell\to m}
    = -\,k_{\ell m}\,\bigl((\hat\zeta_\ell^t\!\cdot\!\hat\zeta_m^t)
      - (\hat\zeta_\ell^0\!\cdot\!\hat\zeta_m^0)\bigr)\,\hat\zeta_m^t,
  }
  \quad
  f^t_{m\to \ell}
  = -\,k_{\ell m}\,\bigl((\hat\zeta_\ell^t\!\cdot\!\hat\zeta_m^t)
    - (\hat\zeta_\ell^0\!\cdot\!\hat\zeta_m^0)\bigr)\,\hat\zeta_\ell^t.
  \tag{2.44–2.45}
$$

---

**Assembly:**  
Each tetrahedron contributes  
- 6 axial‐spring forces, and  
- 6 torsion‐spring forces,  

which are then distributed to the four vertices via the shape‐function coefficients $C^k$ and summed with any body forces before time integration.  

---

### 2.4.3 Simplified Volume Preservation (Barycentric Volume Springs)

To control tetrahedral volume without full tensors, we use **barycentric springs** \[Eqns. 2.76–2.77]:

1. **Current barycenter**:

   $$
     x_b^t=\frac{1}{4}\sum_{i=1}^4 x_i^t.
   $$
2. **Radial vectors**:
   $\xi^t_j=x_b^t - x_j^t$, with lengths $\|\xi^t_j\|$.
3. **Rest lengths** $\|\xi^0_j\|$ computed at $t=0$.
4. **Total length error**:

   $$
     \Delta L=\sum_{j=1}^4\|\xi_j^t\|
     \;-\;\sum_{j=1}^4\|\xi_j^0\|.
   $$
5. **Barycentric spring force** on node $j$:

   $$
     f^t_j
     = -\,k_s\,\Delta L\,\frac{\xi^t_j}{\|\xi^t_j\|}
     \;-\;c\,(v_j^t - v_b^t),
   $$

   where $k_s$ is the bulk‐modulus‐based stiffness, $c$ a damping coefficient, and $v_b^t$ the barycenter velocity.
6. **Adaptive stiffness update** (LMS, Eq. 2.81):

   $$
     k_s^{t+\Delta t}
     = k_s^t + \mu\,\Delta V\,\sum_{j=1}^4\|\xi_j^t\|,
   $$

   clamped to $[k_{\min},k_{\max}]$, with $\Delta V=V^t-V^0$ the volume error.

---

This completes the fully‐spring‐based model:

1. mesh topology & intersection (2.22–2.25),
2. internal forces via axial+torsion springs (2.30–2.45),
3. volume control via barycentric springs (2.76–2.81).
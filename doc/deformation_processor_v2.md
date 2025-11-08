# Deformation Processor v2 — SMS Quadratic Measurements (Reference Configuration)



## Overview

We precompute, for each tetrahedron `k` in the reference configuration, linear
measurement row vectors so that quadratic SMS measurements can be evaluated as
simple dot products with the stacked nodal displacement vector `u_hat ∈ R^{12}`
(4 nodes × 3 DOFs):

- Axial strain along principal axis `e_ℓ`:
  - Measurement: `q_{ℓℓ} = r_axis[k,ℓ] · u_hat`.
  - Target continuum quantity: `q_{ℓℓ} ≈ ε_{ℓℓ} = e_ℓ^T (∇u + (∇u)^T)/2 e_ℓ`.
  - Discrete approximation used: `q_{ℓℓ} ≈ (e_ℓ^T Δu_ℓ) / L^0_ℓ`.

- Shear strain between axes `(e_L, e_M)` with `(L,M) ∈ {(0,1), (1,2), (2,0)}`:
  - Measurement: `q_{LM} = r_shear[k,s] · u_hat`.
  - Target: `q_{LM} ≈ ε_{LM} = e_L^T (∇u + (∇u)^T)/2 e_M`.
  - Discrete formula: `q_{LM} ≈ 1/2 ( (e_M^T Δu_L) / L^0_L + (e_L^T Δu_M) / L^0_M )`.

- Volumetric strain:
  - Measurement: `q_vol = r_vol[k] · u_hat`.
  - Target: `q_vol ≈ δV / V ≈ tr(ε)`.

All three families are assembled in the reference configuration, enabling a
static solver to treat them as “constant stiffness” rows.

## Notation

- Reference positions: `X_i`, deformed positions: `x_i = X_i + u_i`.
- Reference edge matrix: `D = [X_1 − X_0, X_2 − X_0, X_3 − X_0] ∈ R^{3×3}`.
- Deformed edge matrix: `d = [x_1 − x_0, x_2 − x_0, x_3 − x_0]`.
- Deformation gradient: `F = d D^{-1}`; SVD `F = U Σ V^T`.
- Principal directions: columns of `V` give `e_0, e_1, e_2` made right-handed.
- Tet barycenter: `b = (X_0 + X_1 + X_2 + X_3)/4`.

### Axis intersections and barycentric lifting

For each axis `e_ℓ`, we cast a line through the barycenter in directions `± e_ℓ`.
For a triangle face with normal `n` and ray `p(t) = b + t d`, use
`t = (a − b)·n / (d·n)` for any vertex `a` on that face when `|d·n| > 0`. We
accept an intersection when its 2D barycentric coordinates within the face are
all ≥ `−ε`.

Let `P_{ℓ,1}` and `P_{ℓ,2}` be the two valid intersections (if both exist). We
convert each to tet-vertex barycentric coefficients `C_k[a, p]` (with a zero at
the node opposite the face). We then define the signed nodal weights for axis `ℓ`:

- `s_{ℓ,a} = C_k[a, p_2] − C_k[a, p_1]`.
- Rest chord length: `L^0_ℓ = || P_{ℓ,2} − P_{ℓ,1} ||`.

### Volume gradient blocks

For a linear tet, the gradient of volume with respect to nodal positions is the
4-vector of 3D vectors `[g_0, g_1, g_2, g_3]` where, for example, the face
opposite node 0 uses:

- `g_0 = (1/6) (X_1 − X_2) × (X_3 − X_2)`.

Cyclic permutations yield `g_1, g_2, g_3`. We assemble the volumetric row as

- `r_vol = [g_a / V]_{a=0..3}` (each `g_a/V` contributes a 1×3 block).

## Measurement Row Assembly

- Axial rows `r_axis[ℓ]`:

  Distribute `(e_ℓ / L^0_ℓ)` onto the 12 DOFs using weights `s_{ℓ,a}`:

  - For node `a`, its 3 DOFs block gets `s_{ℓ,a} (e_ℓ / L^0_ℓ)`.

- Shear rows `r_shear[(L,M)]` for `(0,1), (1,2), (2,0)`:

  Use the symmetric average:

  - `1/2 ( (e_M / L^0_L) s_{L,·} + (e_L / L^0_M) s_{M,·} )`.

- Volume row `r_vol`:

  Concatenate the 1×3 blocks `[g_a/V]` over the 4 nodes in node order 0..3.

## Implementation Notes

- Degeneracy tolerance: `det(D)` and various dot products use small epsilons to
  guard against parallel or near-degenerate cases.
- We compute intersections and barycentric coefficients on the reference
  geometry. Displacements only affect the axes via `F` (SVD frame extraction).
- Right-handed correction of `(e_0, e_1, e_2)` is enforced by checking
  `sign((e_0 × e_1)·e_2)`.

## Pipeline in Code

Given displacements `u_i`:

1. Update deformed positions `x_i = X_i + u_i`.
2. Build `D, d`; compute `F = d D^{-1}` and extract `e_ℓ` from `V` in SVD.
3. For each axis, compute two boundary intersections and map them to tet-vertex
   barycentric coefficients `C_k[a,p]`.
4. Store rest chord lengths `L^0_ℓ` and the signed nodal weights `s_{ℓ,a}`.
5. Assemble measurement rows `r_axis`, `r_shear`, `r_vol`.

## Exported Data

Calling `get_results()` returns a dictionary with keys:

- `mesh_points`, `tetrahedra`, `labels`.
- `volume`, `mass`.
- `anisotropy_axes`.
- `intersection_points`, `intersection_valid`, `coefficient_matrix`.
- `rest_lengths`, `boundary_nodes`.
- `initial_positions`, `displacement_field`.
- `r_axis`, `r_shear`, `r_vol`.

These arrays are intended for direct consumption by a static equilibrium solver.

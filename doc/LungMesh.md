## Chapter 2

## Adamss: Advanced Anisotropic Mass-Spring System

### 2.1 Motivation

Many deformable models based on the solid foundation of continuum mechanics are currently used in a variety of applications that require accurate physically based modeling of deformable objects. The most prominent of these methods is the finite element method (FEM) which is widely used in computational science for solving partial differential equations on irregular grids.

Despite the dominance of the FEM, interest in developing other accurate models still exists. This interest is fueled by the search for real-time methods for producing realistic deformations in computer animations where the high computational costs of FEM make it unsuitable. Interest exists also within other fields of research where accurate but computationally efficient deformable models are needed.

Deformable models are gaining more and more interest in medical environments as models evolve and the computational capacities of hardware increases. For example virtual surgery planning is one of the promising application where deformable models are applied. With virtual surgery planning surgeons may gain the opportunity to test different critical surgery scenarios in a low cost environment without ethical restriction. It may also allow for making predictions about the outcome of a surgery prior to actually performing that surgery, giving the surgeon the opportunity to avoid undesirable outcomes $[17,18,19,20,21]$. Modeling soft tissue mechanics is another and important research area in medical engineering where deformable models are used to better understand the behavior of these tissues, especially in cases where experimentation is either unethical or beyond the physical possibility [22, 23, 24, 25, 26].

In many cases, when investigating specific phenomena, a large number of deformation simulations must be conducted. That makes computational efficiency a major factor when choosing a modeling technique. The faster the modeling algorithm is the more simulations can be evaluated within the same timeframe. A tradeoff between accuracy and computational efficiency can be considered in or-
der to rapidly gain a coarse understanding of the general trend of the investigated phenomena. This approach can simplify the further analysis of these phenomena using a more accurate, but also more time consuming modeling methods.

Mass-spring systems are considered the simplest and most intuitive of all deformable models. They are computationally efficient as they require only solving a system of coupled ordinary differential equations. They also handle large deformations and large displacements with ease.

Mass-spring systems are the method of choice for cloth animation [27,28,29, $30,31,32,33]$, they were also used for facial modeling [34, 17], for modeling muscle [35, 36, 37, 38, 39], and in virtual surgery planning [17, 40, 19, 20], and also in segmentation and image registration [41, 42].

In this work, a physically based deformable model is presented. The model is a modified mass-spring system that addresses and solves the problems present in ordinary mass-spring systems.

### 2.1.1 Mass-Spring Systems

In order to model an object using a mass-spring system, the object is discretized to mass particles $p_{\mathrm{i}}(i=1, \ldots, n)$, then a network of massless springs connecting the particles together is installed.

Mass-spring systems vary according to the discretization mesh, the way the springs are set between the particles, and the functions used to model the springs.

For example, a simple mass-spring system can be built by discretizing the object using a regular hexahedral mesh, setting the mass particles to the vertices of the hexahedrons and finally setting springs along the sides of the hexahedrons connecting these particles (Fig. 2.1(a)). Another system can be created by adding springs that connect the particles diagonally in each of the hexahedron faces (Fig. 2.1(b)) or diagonally through the volume of the hexahedron connecting particles sitting on opposite vertices in relation to the barycenter (Fig. 2.1(c)).

At any given time $t$, the state of the system is defined by the positions $\mathbf{x}_{\mathrm{i}}$ and the velocities $\mathbf{v}_{\mathrm{i}}$ of the particles. The force $\mathbf{f}_{\mathrm{i}}$ at a particle $p_{\mathrm{i}}$ is computed according to its spring connections with its neighbors, in addition to external forces such as gravity or friction. The Newton's second law of motion is used to calculate the motion of each particle:

$$
\begin{equation*}
m_{\mathrm{i}} \frac{d \mathbf{x}_{\mathrm{i}}}{d t}=\mathbf{f}_{\mathrm{i}} \tag{2.1}
\end{equation*}
$$

and the Newton's second law for the entire particles system can be expressed as
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-03.jpg?height=495&width=1357&top_left_y=364&top_left_x=401)

Fig. 2.1 Single hexahedral elements of mass-spring systems with different springs network topologies. Structural springs (a), structural and surface springs (b), structural and diagonal springs (c). Many of these elements are stacked together along a regular three-dimensional mesh to model the geometry and mechanics of the deformable object.

$$
\begin{equation*}
\mathbf{M} \frac{d \mathbf{x}}{d t}=\mathbf{f}(\mathbf{x}, \mathbf{v}) \tag{2.2}
\end{equation*}
$$

where M is a $3 n \times 3 n$ diagonal matrix also called the mass tensor, $\mathbf{x}$ is a $3 n$ vector of the coordinates of all particles.
By solving this system of coupled ordinary differential equations, the coordinates of the particles can be updated as the model deforms.

### 2.1.2 Problems Associated with Mass-Spring Systems

Ordinary mass-spring systems suffer from several intrinsic limitations that restrict their use in physical modeling. The most troubling of these limitations are listed here:

- In comparison with models based on elasticity theory like the finite element or the finite differences methods, ordinary mass-spring system are not necessarily accurate. Most such systems are not convergent, that is, as the mesh is refined the simulation does not converge on the true solution[9].
- The behavior of these systems depends heavily on the topology and the resolution of the mesh. If the mesh changes the simulation does not converge to the same solution obtained using the original mesh [9].
- Finding the right springs functions and parameters to obtain an accurate model is a very difficult and application dependent process.
- Setting the masses correctly to model homogeneous materials is somewhat troublesome.
- Using ordinary mass-spring systems neither isotropic nor anisotropic materials can be generated and controlled easily.

Concerning the last point, if all springs are set to the same stiffness, anisotropies, which correspond to the mesh geometry chosen for the mass-spring system, will be generated. These anisotropies are generally undesirable.

The anisotropic effect related to the mesh topology decreases with increasing mesh density, if the tiling of the object volume was computed from the triangulation of random uniformly-distributed sample points. However, using an extremely dense mesh contradicts the objective on which using mass-spring systems was based, namely computational efficiency.

It is possible to generate the isotropic or the anisotropic effect in ordinary massspring systems by tuning individual spring stiffnesses to reproduce the desired effect [43, 44], or by designing the mesh in order to align springs on some direction of interest [45, 46]. Both methods are time consuming and not applicable to different geometries in a straight-forward manner.

Ordinary mass-spring systems cannot enforce a constant-volume constraint on the modeled object.

Since modeling the deformation of ventricular myocardial tissue is at the heart of this work, and since this tissue exhibits anisotropic mechanical properties along fiber, sheet and sheet-normal directions (this will be discussed in details in Chapter 4) and maintain a constant volume under deformation, it follows that ordinary mass-spring systems are not suitable.

### 2.1.3 Preliminary Groundwork

Bourguignon et al. proposed a method to control anisotropy in mass-spring systems [38] where anisotropies of a deformable object are specified independently from the underlying mesh topology used for mass discretization. In the same paper, Bourguignon proposed a method for volume preservation loosely related to the soft volume preservation constraint of Lee et al. [47]. Using this method the volume variations during deformation depend on the parameters used for the materials and on the type of deformation the model undergoes. In applications where these volume variations are considered too high, a hard-constraint on volume preservation must be implemented.
M. Mohr [39] used Bourguignon's method to control anisotropy in combination with an implementation of elasticity theory for every individual voxel to enforce volume preservation and to model the passive mechanical properties of the myocardial tissue. The concept of combining the mass-spring system and the elasticity theory solution using FEM for every voxel has proven to be fruitful. However, the hybrid model of M. Mohr suffered from a major problem which is the fur-
ther use of mesh springs, including springs along the hexahedral mesh lines, two diagonal springs on each surface and four diagonal springs in each hexahedron. Not only that these springs are incompatible with Bourguignon's method, because they tend to generate unwanted anisotropies, but also, they require new parameters every time the model is scaled or a different mesh density is used, or when a new object is modeled [48]. Furthermore, the use of these springs in combination with the FEM for modeling passive mechanical properties of tissues results in a systematic error in reproducing the passive mechanical properties.

### 2.1.4 Introducing Adamss

In this work a modified mass-spring system, called Adamss (Advanced anisotropic mass-spring system), based on the work of Bourguignon et al. [38] is presented. The system is composed of several building blocks with defined interfaces between the blocks. Figure 2.2 shows the building blocks constituting the system. This design allows for the development of building blocks based on completely
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-05.jpg?height=418&width=1446&top_left_y=1282&top_left_x=345)

Fig. 2.2 The building blocks of Adamss (Advanced anisotropic mass-spring system), the physically based deformable model presented in this work.
different concepts as long as the interfaces with other blocks are intact making the system flexible and extendible.

During the course of the development of the system, the concept introduced by M. Mohr of combining the theory of elasticity to model the passive mechanical properties of materials and to enforce volume preservation was adopted and extended in the corresponding building blocks. Although the concepts are very similar, the implementations, detailed later in this chapter, differ significantly.

In the following sections the building blocks of the system are presented and the different implementations of each of the building blocks are detailed. Gradually, we will show how the proposed system takes on the mentioned problems associated with mass-spring systems (see Section 2.1.2) and solve them elegantly. Additionally, we will show that the system's behavior is independent of the mesh
resolution, and partly independent of the mesh topology. At the end we will show that the overall system is of $O(n)$ complexity.

### 2.2 Simulation Scheme

Figure 2.3 shows the basic simulation scheme of Adamss.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-06.jpg?height=1209&width=752&top_left_y=783&top_left_x=606)

Fig. 2.3 A flowchart of the basic simulation scheme of Adamss.

The simulation start with an initialization step, where the model geometry files including information about anisotropies and boundary conditions, and parameters regulating the mass-spring system generation process are loaded.

After the mass-spring system has been initialized, the system starts the simulation loop. In every iteration of the simulation loop, the different forces working on the particles of the system are calculated. Then time integration of the equations of motion is used to calculate the velocities and offsets of the system's particles.

Depending on the chosen data exporting rate, information about the forces, velocities, and offsets of the system can be exported. Additionally general information like the system's total volume or the system's total kinetic energy can be exported in this step. After that the coordinates of the particles are updated using the offsets and the loop iterates until a condition for stopping the simulation is fulfilled. These conditions can be simulation time crossing a maximum simulation duration $t_{\text {max }}$ or kinetic energy dropping below a threshold $E_{\mathrm{k}, \text { min }}$. Numerical instability and external interception also stops a running simulation.

The following sections will detail each steps of the simulation scheme presented here.

### 2.3 Structure Initialization

To initialize the structure of the modified mass-spring system, files containing information about the geometry of the modeled object, and properties of that geometry are imported to the framework.

According to the chosen discretization mesh topology, the modeled object mass is discretized to mass particles as in ordinary mass-spring systems (see Section 2.1.1). Nonetheless, springs are not set on the edge of the discretization mesh.

Instead, the space occupied by the modeled object is also discretized to volume elements according to the selected mesh topology. These resulting volume elements are actually defined by the particles generated during the discretization of the object's mass. Each of the elements encloses a geometrical domain of the object where parameters, important for the modeling, are set and considered homogeneously distributed within that domain. These parameters include the material type of the volume element, the mass-density, different anisotropies, different stiffnesses, the bulk modulus, and other parameters related to the specific material type. The volume elements can take different geometrical shapes depending on the mesh topology used. For example, if a hexahedral mesh topology was used, the resulting volume elements will then be hexahedra. The number of vertices or particles that define a volume element depends on the element type. The faces of a volume element defined by the vertices are called facets.

As soon as defining volume elements is completed, the optional defining of surfaces of enclosed cavities starts. Each of the cavities is defined by the vertices of the facets that enclose the volume of a cavity. These cavities can be used to add constraints to the deformation, for example by enforcing a volume preservation of the cavity volume during deformation. It is also possible to set a constant homogeneous stress to all facets of a cavity, to model ventricular cavities pressure for example.

There are many possibilities to calculate the volume of an enclosed region, providing the surface enclosing the region is known. In this work we used a Delaunay tetrahedralizations technique to generate a tetrahedral mesh of the cavity starting with the vertices marking the cavity. The volume of the cavity can then be calculated by summing the volume of the resulting tetrahedrons.

### 2.3.1 Mesh Topologies

During the course of development of the modeling framework, the hexahedral mesh topology and several different tetrahedral mesh topologies were implemented. In the following, the implementation of these topologies is presented. The advantages and drawbacks of each of the methods will be discussed later in Section 2.9.

### 2.3.1.1 Hexahedral Mesh Topology

The hexahedral mesh topology is a regular grid, where after first initialization all mesh elements are identical rectangular hexahedra (Fig. 2.4(b)). In the special case when the length of all hexahedra sides are equal, the mesh elements are called voxels (Fig. 2.4(a)).
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-08.jpg?height=577&width=1204&top_left_y=1519&top_left_x=386)

Fig. 2.4 Hexahedral mesh topologies: voxels mesh with 1 mm voxel side(a), rectangular hexahedral mesh of resolution $1 \mathrm{~mm} \times 2 \mathrm{~mm} \times 1 \mathrm{~mm}$ (b).

Here, the mass of the modeled object is discretized according to the mesh into particles that define the vertices of the volume elements of the modeled object. This is not only true for the hexahedral mesh but also for the different tetrahedral meshes.

In order to setup a homogeneous material, the mass of each particle is computed according to the volume of the Voronoi region around it [43]. In the general case
it can be calculated with

$$
\begin{equation*}
m_{\mathrm{i}}=\sum_{\mathrm{k}=1}^{n} \frac{\rho_{\mathrm{k}} V_{\mathrm{k}}}{N_{\mathrm{k}}} \tag{2.3}
\end{equation*}
$$

where $m_{\mathrm{i}}$ is the mass of the particle $p_{\mathrm{i}}, n$ is the number of volume elements neighboring the particle. $V_{\mathrm{k}}, \rho_{\mathrm{k}}$ and $N_{\mathrm{k}}$ are respectively the volume, the mass density, and the number of vertices of the volume element $\mathcal{V}_{\mathrm{k}}$.
For a hexahedral mesh $m_{\mathrm{i}}$ is given by

$$
\begin{equation*}
m_{\mathrm{i}}=\frac{V_{\mathrm{v}}}{8} \sum_{\mathrm{k}=1}^{n} \rho_{\mathrm{k}} \tag{2.4}
\end{equation*}
$$

where $n$ is the number of hexahedra neighboring $p_{\mathrm{i}}$ and $\rho_{\mathrm{k}}$ is the mass density of the hexahedron $\mathcal{H}_{\mathrm{k}} . V_{\mathrm{v}}$ is the volume of a hexahedron of the hexahedral mesh.

To control anisotropy, the method presented by Bourguignon et al. [38] is used. The basic idea is to define several axes of mechanical anisotropy in the barycenter of each volume element of the mesh. Forces generated due to the deformation of the model will act only in the direction of these axes. For example, to model the mechanical deformation of a muscle, the direction of anisotropy axes must be set in the direction of the fibers of the muscle in each of the volume elements of the model's mesh. Although the method allows the definition of several axes of anisotropy, only three were defined in each of volume elements. This was enough to model the anisotropic mechanical behavior of all types of materials modeled with the developed framework.

In a volume element, each axis intersects with the faces of that element in two points, called intersection points (Fig. 2.5(a)). During deformation, the axes evolve with the volume elements to which they belong. At any given moment $t$, the orientation of an axis $\zeta_{\mathrm{l}}$ in a volume element $\mathcal{V}_{\mathrm{k}}$ can be determined using the line extended between the pair of intersection points the axis defines. This can be done numerically by calculating the vector between the pair ( $q_{21}, q_{21+1}$ ) regardless of its direction. The vector is given by

$$
\begin{equation*}
\boldsymbol{\zeta}_{1}^{t}=\mathbf{x}_{21}^{t}-\mathbf{x}_{21+1}^{t} \tag{2.5}
\end{equation*}
$$

For this reason, intersection points are used to track the axes of anisotropies during the deformation of the model.

To compute the coordinates of the intersection points at any moment $t$, and hence the orientation of the anisotropy axes, the coordinates $\mathrm{x}_{\mathrm{j}}^{t}$ of each intersection point $q_{\mathrm{j}}$ are given by a linear interpolation of the coordinates of the face vertices to which it belongs, using the rectangle linear interpolation shape function
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-10.jpg?height=783&width=1338&top_left_y=354&top_left_x=276)

Fig. 2.5 Intersection points in a hexahedral volume element: The 3D hexahedral element with three axes of anisotropy set at the barycenter and the six intersection points they define (a), a rectangle face of the element containing the intersection point $q_{0}$ and the coefficients $\xi_{0}$ and $\eta_{0}$ related to $q_{0}$, the vertices of that face are denoted with $p_{\mathrm{i}}(i=1, \ldots, 4)(\mathrm{b})$.

$$
\begin{equation*}
\mathbf{x}_{\mathrm{j}}^{t}=\sum_{\mathrm{i}=1}^{4} N_{\mathrm{i}}\left(\xi_{\mathrm{j}}, \eta_{\mathrm{j}}\right) \mathbf{x}_{\mathrm{i}}^{t} \tag{2.6}
\end{equation*}
$$

where $\mathbf{x}_{\mathrm{i}}^{t}$ are the coordinates of the vertices of the rectangular face $\mathcal{F}_{\mathrm{j}}$ to which $q_{\mathrm{j}}$ belongs and $N_{\mathrm{i}}\left(\xi_{\mathrm{j}}, \eta_{\mathrm{j}}\right)$ are the rectangle linear interpolation shape functions [49, 1] and are given with

$$
\begin{align*}
& N_{1}\left(\xi_{\mathrm{j}}, \eta_{\mathrm{j}}\right)=\left(1-\xi_{\mathrm{j}}\right)\left(1-\eta_{\mathrm{j}}\right) \\
& N_{2}\left(\xi_{\mathrm{j}}, \eta_{\mathrm{j}}\right)=\xi_{\mathrm{j}}\left(1-\eta_{\mathrm{j}}\right)  \tag{2.7}\\
& N_{3}\left(\xi_{\mathrm{j}}, \eta_{\mathrm{j}}\right)=\xi_{\mathrm{j}} \eta_{\mathrm{j}} \\
& N_{4}\left(\xi_{\mathrm{j}}, \eta_{\mathrm{j}}\right)=\left(1-\xi_{\mathrm{j}}\right) \eta_{\mathrm{j}}
\end{align*}
$$

where $\xi_{\mathrm{j}}, \eta_{\mathrm{j}}$ are the interpolation coefficients associated with $q_{\mathrm{j}}$ (Fig. 2.5(b)).
During structure initialization, the axes of mechanical anisotropies are defined at the barycenter of each hexahedron of the model. The intersection points and the corresponding shape functions are also computed at this stage using the method described here:

1. For each hexahedron $\mathcal{H}_{\mathrm{k}}$ of the mesh, repeat the following steps:
2. For each of the three axes $\zeta_{1}$ of $\mathcal{H}_{\mathrm{k}}$, repeat the following steps:
3. For each face $\mathcal{F}_{\mathrm{j}}$ of $\mathcal{H}_{\mathrm{k}}$, repeat the following steps to locate intersection points axis $\zeta_{1}$ defines:
4. Compute the point where the line extending from the barycenter $p_{\mathrm{bk}}$ of $\mathcal{H}_{\mathrm{k}}$ and which is parallel to $\zeta_{1}$ intersects with the plane containing $\mathcal{F}_{\mathrm{j}}$
5. If the computed point was outside $\mathcal{F}_{\mathrm{j}}$, it is not a valid intersection point, go to step 3 (iterate over faces)
6. If the point is in $\mathcal{F}_{\mathrm{j}}$, it is a valid intersection point, compute the related shape function values
7. If two valid intersection points were found for $\zeta_{\mathrm{l}}$, go to step 2 (iterate over axes)

Step 4 is actually a ray tracing task that requires computing the barycenter, which can be done using

$$
\begin{equation*}
\mathbf{x}_{\mathrm{b}}=\frac{1}{8} \sum_{\mathrm{i}=1}^{8} \mathbf{x}_{\mathrm{i}} \tag{2.8}
\end{equation*}
$$

where $\mathbf{x}_{\mathrm{i}}$ are the hexahedron's vertices. The vector equation of the ray starting from $p_{\text {bk }}$ in the direction of axis $\zeta_{1}$ can be given with

$$
\begin{equation*}
\mathbf{v}=d \widehat{\boldsymbol{\zeta}}_{1}+\mathbf{x}_{\mathrm{bk}} \tag{2.9}
\end{equation*}
$$

where $\widehat{\zeta}_{1}$ is the unit vector in direction of $\zeta_{1}$ and $d$ is a distance along $\zeta_{1}$. The plane of $\mathcal{F}_{\mathrm{j}}$ can be expressed in vector notation with

$$
\begin{equation*}
\left(\mathbf{v}-\mathbf{x}_{\mathbf{i}}\right) \cdot \mathbf{n}=0 \tag{2.10}
\end{equation*}
$$

where $\mathbf{x}_{\mathrm{i}}$ is one of the face's vertices, i. e. a random point in the plane. $\mathbf{n}$ is the normal on the plane and can be simply calculated using the face's vertices with

$$
\begin{equation*}
\mathbf{n}=\frac{\left(\mathbf{x}_{1}-\mathbf{x}_{2}\right) \times\left(\mathbf{x}_{2}-\mathbf{x}_{3}\right)}{\left\|\left(\mathbf{x}_{1}-\mathbf{x}_{2}\right) \times\left(\mathbf{x}_{2}-\mathbf{x}_{3}\right)\right\|} \tag{2.11}
\end{equation*}
$$

By substituting Eq. (2.9) in Eq. (2.10) and solving for $d$ we obtain the distance along $\zeta_{1}$ to the plane starting from $p_{\mathrm{bk}}$ :

$$
\begin{equation*}
d=\frac{\left(\mathbf{x}_{\mathrm{i}}-\mathbf{x}_{\mathrm{bk}}\right) \cdot \mathbf{n}}{\zeta_{\mathbf{l}} \cdot \mathbf{n}} \tag{2.12}
\end{equation*}
$$

Now by substituting $d$ in Eq. (2.9) the point where the ray intersects with the plane is obtained.

A special method has been developed, not only to check if a specific point which is coplanar to a rectangle is inside that rectangle (step 5), but also to compute the corresponding shape functions (step 6).

First the surface $S_{\square}$ of the face defined by vertices $p_{\mathrm{i}}(i=1, \ldots, 4)$ is calculated with

$$
\begin{equation*}
S_{\square}=\left\|\mathbf{x}_{1}-\mathbf{x}_{2}\right\| \cdot\left\|\mathbf{x}_{1}-\mathbf{x}_{3}\right\| \tag{2.13}
\end{equation*}
$$

where $\mathbf{x}_{\mathrm{i}}$ are the rectangle vertices' coordinates. Then the surfaces of the four different triangles the intersection point $q_{\mathrm{j}}$ defines with the rectangle's vertices, namely $S_{\triangle \mathrm{j} 12}, S_{\triangle \mathrm{j} 14}, S_{\triangle \mathrm{j} 23}$ and $S_{\triangle \mathrm{j} 34}$ are calculated (see Fig. 2.5(b)). The surface of a triangle can be calculated using

$$
\begin{equation*}
S_{\triangle 123}=\frac{1}{2}\left\|\left(\mathbf{x}_{1}-\mathbf{x}_{2}\right) \times\left(\mathbf{x}_{2}-\mathbf{x}_{3}\right)\right\| \tag{2.14}
\end{equation*}
$$

here $\mathbf{x}_{\mathrm{i}}(i=1, \ldots, 3)$ are the triangle vertices' coordinates. If the statement:

$$
\begin{equation*}
S_{\square}=S_{\triangle \mathrm{j} 12}+S_{\triangle \mathrm{j} 14}+S_{\triangle \mathrm{j} 23}+S_{\triangle \mathrm{j} 34} \tag{2.15}
\end{equation*}
$$

is true, then the point $q_{\mathrm{j}}$ is located inside the rectangle, and the shape functions coefficients $\xi_{\mathrm{j}}$ and $\eta_{\mathrm{j}}$ can be calculated with

$$
\begin{align*}
& \xi=2 \cdot S_{\triangle \mathrm{j} 23} / S_{\square}  \tag{2.16}\\
& \eta=2 \cdot S_{\triangle \mathrm{j} 34} / S_{\square}
\end{align*}
$$

The shape functions of each of the intersection points of a hexahedron $\mathcal{H}_{\mathrm{k}}$ can be arranged in a matrix $\mathbf{C}_{k}$, we call the coefficient matrix, according to

$$
C_{\mathrm{ij}}=\left\{\begin{array}{l}
N_{\mathrm{ij}} \text { where } p_{\mathrm{i}} \text { is a vertex of } \mathcal{H}_{\mathrm{k}} \text { and also the }  \tag{2.17}\\
\quad \text { face } \mathcal{F}_{\mathrm{j}} \text { containing intersection point } q_{\mathrm{j}} \\
0 \text { otherwise }
\end{array}\right.
$$

$N_{\mathrm{ij}}$ is the shape function associated with $q_{\mathrm{j}}$ and $p_{\mathrm{i}}$ and calculated using Eqs. (2.7) and (2.16). There are six intersection points and eight nodes, that means the coefficients matrix $\mathbf{C}_{\mathrm{k}}$ is an $8 \times 6$ matrix. Using $\mathbf{C}_{\mathrm{k}}$, it is possible to calculate the coordinates $\mathbf{x}_{\mathrm{i}}^{t}$ of intersection points $p_{\mathrm{j}}^{t}$ at time $t$ by calculating

$$
\begin{equation*}
\mathbf{x}_{\mathrm{j}}^{t}=\sum_{\mathrm{i}=1}^{8} C_{\mathrm{ij}} \mathbf{x}_{\mathrm{i}}^{t} \tag{2.18}
\end{equation*}
$$

### 2.3.1.2 Tetrahedral Mesh Topologies

Tetrahedral mesh topologies are topologies where the mesh elements are tetrahedra.

As in hexahedral mesh topologies, the mass of each particle is computed according to the volume of the Voronoi region around it according to Eq. (2.3), which can be rewritten for a the special case of a tetrahedral mesh as

$$
\begin{equation*}
m_{\mathrm{i}}=\frac{1}{4} \sum_{\mathrm{k}=1}^{n} \rho_{\mathrm{k}} V_{\mathrm{k}} \tag{2.19}
\end{equation*}
$$

where $m_{\mathrm{i}}$ is the mass of the particle $i, n$ is the number of tetrahedra neighboring the particle $i, \rho_{\mathrm{k}}$ and $V_{\mathrm{k}}$ are the mass density and the volume of the tetrahedron $\mathcal{T}_{\mathrm{k}}$.

Similarly to the method used to control mechanical anisotropies in objects modeled using a hexahedral mesh, three axes of anisotropy are defined at the barycenter of each tetrahedron. Each axis defines two intersection points on the triangle faces of the tetrahedron (Fig. 2.6(a)).
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-13.jpg?height=664&width=1286&top_left_y=779&top_left_x=374)

Fig. 2.6 Intersection points in a tetrahedral volume element: The tetrahedron with three axes of anisotropy set at the barycenter and the six intersection points that they define (a), a triangular face of the element containing an intersection point and the coefficients $\xi_{0}$ and $\eta_{0}$ related to the intersection point. Note that $\xi$ increases with the cyan color gradient starting from $\xi=0$ at the line segment ( $p_{1}, p_{2}$ ) and is equal to $\xi=1$ at $p_{3}$, while $\eta$ increases along the orange color gradient starting from $\eta=0$ at ( $p_{2}, p_{3}$ ) until it reaches $\eta=1$ at $p_{1}$ (b).

To track the axes as the model and the underlying tetrahedra deform, the coordinates of the intersection points are calculated as a linear interpolation of the vertices coordinates of the triangle faces to which the intersection points belong with

$$
\begin{equation*}
\mathbf{x}_{\mathrm{j}}^{t}=\sum_{\mathrm{i}=1}^{3} N_{\mathrm{i}}\left(\xi_{\mathrm{j}}, \eta_{\mathrm{j}}\right) \mathbf{x}_{\mathrm{i}}^{t} \tag{2.20}
\end{equation*}
$$

Here, the triangle linear interpolation shape functions $N_{\mathrm{i}}\left(\xi_{\mathrm{j}}, \eta_{\mathrm{j}}\right)$ are used [1]:

$$
\begin{align*}
& N_{1}\left(\xi_{\mathrm{j}}, \eta_{\mathrm{j}}\right)=1-\xi_{\mathrm{j}}-\eta_{\mathrm{j}} \\
& N_{2}\left(\xi_{\mathrm{j}}, \eta_{\mathrm{j}}\right)=\xi_{\mathrm{j}}  \tag{2.21}\\
& N_{3}\left(\xi_{\mathrm{j}}, \eta_{\mathrm{j}}\right)=\eta_{\mathrm{j}}
\end{align*}
$$

where $\mathbf{x}_{\mathrm{j}}^{t}$ is the coordinate of the intersection point $q_{\mathrm{j}}$ at a time $t$, and $\xi_{\mathrm{j}}, \eta_{\mathrm{j}}$ are the interpolation coefficients associated with $q_{\mathrm{j}}$ (Fig. 2.6(b)).

During structure initialization, the axes of mechanical anisotropies are defined at the barycenter of each tetrahedron of the model. The intersection points and the corresponding shape functions are also computed at this stage using the same method described in Section 2.3.1.1 for finding intersection points and calculating shape functions, however, after adapting the method to support tetrahedral meshes.

Here also a ray tracing task is performed starting from the barycenter of a tetrahedron in the direction of the anisotropy axes in order to search for intersection points.

The first difference between the hexahedral and the tetrahedral mesh implementation of the method is calculating the barycenter of the mesh element which must be calculated for tetrahedral elements with

$$
\begin{equation*}
\mathrm{x}_{\mathrm{b}}=\frac{1}{4} \sum_{\mathrm{i}=1}^{4} \mathrm{x}_{\mathrm{i}} \tag{2.22}
\end{equation*}
$$

To check if a specific traced point $q_{\mathrm{j}}$ which is coplanar to one of the triangular faces of the tetrahedron $\mathcal{T}_{\mathrm{k}}$ is inside that triangle and to compute the corresponding shape functions at the same time, the surfaces of triangles $S_{\triangle 123}, S_{\triangle \mathrm{j} 12}, S_{\triangle \mathrm{j} 13}$ and $S_{\triangle \mathrm{j} 23}$ are calculated using Eq. (2.14). The triangle $S_{\triangle 123}$ is defined by the vertices $p_{\mathrm{i}}(i=1, \ldots, 3)$, while $S_{\triangle \mathrm{j} 12}, S_{\triangle \mathrm{j} 13}$ and $S_{\triangle \mathrm{j} 23}$ are the triangles which the point $q_{\mathrm{j}}$ defines with the face vertices.

If the statement:

$$
\begin{equation*}
S_{\triangle 123}=S_{\triangle \mathrm{j} 12}+S_{\triangle \mathrm{j} 13}+S_{\triangle \mathrm{j} 23} \tag{2.23}
\end{equation*}
$$

was true, then the point $q_{\mathrm{j}}$ is located inside the triangular face, and the shape functions coefficients $\xi_{\mathrm{j}}$ and $\eta_{\mathrm{j}}$ can be given by

$$
\begin{align*}
\xi & =S_{\triangle \mathrm{j} 13} / S_{\triangle 123}  \tag{2.24}\\
\eta & =S_{\triangle \mathrm{j} 12} / S_{\triangle 123}
\end{align*}
$$

Setting the shape functions using Eq. (2.24) is legitimate only for linear tetrahedra i.e. straight-sided triangles. The coefficients in this case are also called area or areal coordinates. Eq. (2.24) does not carry over to general isoparameteric higher order triangles i.e. with curved sides.

Here also, the shape functions of each of the intersection points of a tetrahedron $\mathcal{T}_{\mathrm{k}}$ can be arranged in a coefficient matrix $\mathbf{C}_{\mathrm{k}}$, according to Eq. (2.17) where $N_{\mathrm{ij}}$ is the shape function associated with intersection point $q_{\mathrm{j}}$ and the vertex $p_{\mathrm{i}}$ and calculated using Eqs. (2.21) and (2.24). In each tetrahedron, there are six intersection points and four nodes, that means the coefficients matrix $\mathbf{C}_{\mathrm{k}}$ is a $4 \times 6$ matrix.

Using the matrix $\mathbf{C}_{\mathrm{k}}$, it is possible to calculate the coordinates of intersection points $q_{\mathrm{j}}^{t}$ at time $t$ by calculating

$$
\begin{equation*}
\mathbf{x}_{\mathrm{j}}^{t}=\sum_{\mathrm{i}=1}^{4} C_{\mathrm{ij}} \mathbf{x}_{\mathrm{i}}^{t} \tag{2.25}
\end{equation*}
$$

Tetrahedral topologies can be generated by systematically dividing a regular grid of the modeled object to tetrahedrons or by using unstructured grids where the modeled object is broken down to irregular tetrahedra using a mesh generation algorithm.

### 2.3.1.3 Tetrahedral Mesh Topologies Based on Regular Grids

This method take the advantages that regular grids offer, especially the analogy to medical images data structures, and at the same time benefit of the use of tetrahedra to solve problems associated with hexahedra. Each hexahedron of the hexahedral mesh is divided to a number of non-intersecting linear tetrahedra according to a specific division scheme. This strategy was inspired by a method to compute the volume of a deformed hexahedron by dividing it to six non-intersecting tetrahedra and then summing up the computed volume of each of the tetrahedra [50].

Schemes to divide each hexahedron to five or six non-intersecting tetrahedra were implemented. These schemes do not alter the number of vertices of the model nor the masses the hexahedral mass discretization defines.

Figure 2.7 shows a hexahedron divided to five tetrahedra. It is notable that tetrahedron $\mathcal{T}_{5}$ has a bigger volume in comparison with the remaining tetrahedra.

Figure 2.8 shows a hexahedron divided to six tetrahedra. Using this division scheme all resulting tetrahedra have the same volume.

Figures 2.9(a) and 2.9(b) shows two different mesh topologies based on five tetrahedra per hexahedron schemes. In the the first scheme 2.9(a) all hexahedra were divided identically, while in the second 2.9(b) a division scheme that has a cubic pattern kernel of $2 \times 2 \times 2$ hexahedra is used. In each of the kernel the division scheme produces mirrored tetrahedra.

Figures 2.10(a) and 2.10(b) show two different mesh topologies based on the six tetrahedra per hexahedron scheme. In the the first scheme (Fig. 2.10(a)) all hexahedra were divided identically, while in the second (Fig. 2.10(b)) a division scheme that has a cubic pattern kernel of $2 \times 2 \times 2$ hexahedra is used. In each of the kernel the division scheme produces mirrored tetrahedra.

The schemes presented in Figures 2.9(b) and 2.10(b) were implemented to eliminate anisotropies resulting from the implementation of volume preservation forces that consider a homogeneous elements volume distribution.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-16.jpg?height=427&width=1361&top_left_y=586&top_left_x=299)
(a)
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-16.jpg?height=1040&width=1312&top_left_y=1239&top_left_x=315)
(b)

Fig. 2.7 In (a), A hexahedron (left) divided to five tetrahedra in two different ways (middle and right). The lines connecting the vertices of the hexahedra define the different tetrahedra resulting from the division scheme. In (b), the tetrahedra resulting from the first division scheme (a, middle) are shown.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-17.jpg?height=438&width=1380&top_left_y=369&top_left_x=389)
(a)
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-17.jpg?height=1086&width=1198&top_left_y=1019&top_left_x=506)
(b)

Fig. 2.8 In (a), A hexahedron (left) divided to six tetrahedra in two different ways (middle and right), two possible way are not depicted here. The lines connecting the vertices of the hexahedra define the different tetrahedra resulting from the division scheme. In (b), the tetrahedra resulting from the second division scheme (a, right) are shown.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-18.jpg?height=686&width=1412&top_left_y=368&top_left_x=276)

Fig. 2.9 Different mesh topologies based on the hexahedron to five tetrahedra division scheme with a model of a cube of $18 \times 18 \times 18=5832$ voxels thus to 29160 tetrahedra. All hexahedrons are divided identically (a). A division scheme with a cubic pattern kernel generates a symmetric mesh topology (b).
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-18.jpg?height=684&width=1406&top_left_y=1240&top_left_x=279)

Fig. 2.10 Different mesh topologies based on the hexahedron to six tetrahedra division scheme with a model of a cube of $18 \times 18 \times 18=5832$ voxels thus to 34992 tetrahedra. All hexahedrons are divided identically (a). A division scheme with a cubic pattern kernel generate a symmetric mesh topology (b).

### 2.3.1.4 Tetrahedral Mesh Based on Unstructured Grids

Unstructured grids are widely used in computational modeling specially when the modeled object has an irregular shape. In this work, CGAL (Computer Geometry Algorithm Library) [51] was used to generate the tetrahedral meshes out of the lattice image datasets. CGAL is an open source project that provides access to many geometric algorithms in form of a C++ library which is a popular library that is used in many projects [52, 53, 54].

### 2.4 Forces Calculation

In this section, the different models of forces that act on the particles of the modeled object are discussed. In our implementation forces are divided to internal and external forces.

Internal forces are generated in the volume elements of the modeled object either in response to deformation, or as the result of internal processes active within the element, like the contraction forces of a muscle for example. External forces are forces affecting the particles regardless of the state of the elements to which these particles belong. This group includes forces resulting from external acceleration, initial velocity, external pressures or stresses on the defined cavities of the model. Friction is also part of this group.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-19.jpg?height=672&width=1469&top_left_y=1097&top_left_x=345)

Fig. 2.11 The organization of the force calculating module, showing the models used for the calculation of internal and external forces

Usually, modeling complex objects composed of several materials is of high interest. Modeling cardiac mechanics is one example related to this work. These materials exhibit a variety of mechanical properties. Therefore different forces models must be used in combination to reproduce the object's complex mechanical properties. That also applies for the different internal processes that translate into forces. Objects are often under the effect of several external forces acting at the same time. A combination of different forces models must also be used to model these external forces. Interchangeable internal and external forces models were developed. Each of the models has its area of application, advantages and surely disadvantages. Figure 2.11 shows the organization of these different models.

Before we go through the mentioned models, the method used to control anisotropy in this framework must be explained.

### 2.4.1 Controlling Anisotropy

In ordinary mass-spring systems, the length of each structural spring set between two particles changes when the model deforms. This results in forces acting on the particles in the direction opposite to the change, trying to bring the spring to its initial length, thus, bringing the system back to equilibrium. However these forces give rise to undesirable anisotropies working in the direction of mesh lines, as mentioned in Section 2.3.

In this work, the method proposed by Bourguignon et al. [38] to control anisotropy in mass-spring systems is used. This method provides the possibility to let forces act along pre-defined axes of interest, i.e. along pre-defined axes of anisotropy.

As described in details in Section 2.3, in each volume element $\mathcal{V}_{\mathrm{k}}$ of the model, three axes of anisotropy $\zeta_{1}(l=1, \ldots, 3)$ are defined at the barycenter. These axes intersect with the surfaces of the volume element in intersection points. For a volume element $\mathcal{V}_{\mathrm{k}}$, the coordinate $\mathbf{x}_{\mathrm{j}}$ of an intersection point $q_{\mathrm{j}}$ can be given using the coefficient matrix $\mathbf{C}_{\mathrm{k}}$ of that element using Eq. (2.18) or Eq. (2.25), according to the mesh topology. For both the hexahedral and the tetrahedral, we can write

$$
\begin{equation*}
\mathbf{x}_{\mathrm{j}}^{t}=\sum_{\mathrm{i}=1}^{n} C_{\mathrm{ij}} \mathbf{x}_{\mathrm{i}}^{t} \tag{2.26}
\end{equation*}
$$

where $n$ is the number of vertices of $\mathcal{V}_{\mathrm{k}}$. Additionally, it is possible to calculate the velocity of intersection points with

$$
\begin{equation*}
\frac{d \mathbf{x}_{\mathrm{j}}^{t}}{d t}=\sum_{\mathrm{i}=1}^{n} C_{\mathrm{ij}} \frac{d \mathbf{x}_{\mathrm{i}}^{t}}{d t} \tag{2.27}
\end{equation*}
$$

To control anisotropy, forces are first calculated at intersection points and then distributed to the particles. For instance, the force $\mathbf{f}_{\mathrm{j}}$ calculated at intersection point $q_{\mathrm{j}}$ is distributed to the particles $p_{\mathrm{i}}$ of the face to which $q_{\mathrm{j}}$ belongs according to the shape functions $N_{\mathrm{i}}\left(\xi_{\mathrm{j}}, \eta_{\mathrm{j}}\right)$ calculated for $q_{\mathrm{j}}$ using Eq. (2.6) or Eq. (2.20) depending on the mesh topology. The portion of $\mathbf{f}_{\mathrm{j}}$ acting on vertex $p_{\mathrm{i}}$ is given with

$$
\begin{equation*}
\mathbf{f}_{\mathrm{ij}}=N_{\mathrm{i}}\left(\xi_{\mathrm{j}}, \eta_{\mathrm{j}}\right) \mathbf{f}_{\mathrm{j}} \tag{2.28}
\end{equation*}
$$

The force $\mathbf{f}_{\mathrm{i}}$ active at vertex $p_{\mathrm{i}}$ of a volume element $\mathcal{V}_{\mathrm{k}}$ is the accumulation of portions of the forces $f_{i j}$ acting on intersection points belonging to faces of which $p_{\mathrm{i}}$ is a vertex. By making use of the coefficient matrix $\mathbf{C}_{\mathrm{k}}$, and since we have
six intersection points per volume element, the last statement can be formulated mathematically as

$$
\begin{equation*}
\mathbf{f}_{\mathrm{i}}^{t}=\sum_{\mathrm{j}=0}^{5} C_{\mathrm{ij}} \mathbf{f}_{\mathrm{j}}^{t} \tag{2.29}
\end{equation*}
$$

To model an anisotropic behavior, stresses $\sigma_{\mathrm{j}}$ acting on faces of $\mathcal{V}_{\mathrm{k}}$ should be transformed to forces $\mathbf{f}_{\mathrm{j}}$ at the intersection points $q_{\mathrm{j}}$ of the faces, and then distributed using 2.29 to the particles $p_{\mathrm{i}}$ of $\mathcal{V}_{\mathrm{k}}$. To model isotropic behavior, stresses $\sigma_{\mathrm{j}}$ acting on faces of $\mathcal{V}_{\mathrm{k}}$ should be transformed to forces acting directly on particles $p_{\mathrm{i}}$ of $\mathcal{V}_{\mathrm{k}}$ ignoring the axes of anisotropies, intersection points and forces distribution rule.

### 2.4.2 Internal Forces

Internal forces can either refer to forces generated in the volume elements of the model due deformation (deformation forces), or forces generated due to internal processes in the volume elements of the model (active forces).

### 2.4.2.1 Axial and Torsion Springs

One method to calculate deformation forces is using axial and torsion springs as presented in the work of Bourguignon et al. [38].

In each volume element $\mathcal{V}_{\mathrm{k}}$, three axial springs $\mathcal{S}_{1}$ are defined between each pair of intersection points ( $q_{21}, q_{21+1}$ ). A torsion spring $\tau_{\mathrm{lm}}$ is defined between each pair of these axial springs $\left(\mathcal{S}_{1}, \mathcal{S}_{\mathrm{m}}\right)$. In total, three axial springs and three torsion springs are defined in each volume element that uses this method to calculate internal forces. Figure 2.12 shows a tetrahedron with three axial springs and three torsion springs. The initial length of an axial spring can be computed using Eq. (2.5) with $t=0$ and then calculating the Euclidean norm

$$
\begin{equation*}
l_{\mathcal{S}_{1}}^{0}=\left\|\zeta_{1}^{0}\right\|=\left\|\mathrm{x}_{21}^{0}-\mathrm{x}_{21+1}^{0}\right\| \tag{2.30}
\end{equation*}
$$

where $\mathbf{x}_{21}^{0}$ and $\mathbf{x}_{21+1}^{0}$ are the initial coordinates of the intersection points defined by the anisotropy axis $\zeta_{1}$.

By defining the unit vector $\hat{\zeta_{1}^{t}}$ in the direction of axis $\zeta_{1}^{t}$ at time $t$ with

$$
\begin{equation*}
\hat{\zeta}_{1}^{t}=\frac{\zeta_{1}^{t}}{\left\|\zeta_{1}^{t}\right\|} \tag{2.31}
\end{equation*}
$$

the angle $\alpha_{\mathrm{lm}}^{t}$ between the axes $\zeta_{\mathrm{l}}$ and $\zeta_{\mathrm{m}}$ can be given by
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-22.jpg?height=720&width=863&top_left_y=394&top_left_x=551)

Fig. 2.12 A tetrahedron with three axial springs (in cyan) along the axes of anisotropy and three torsion springs in the barycenter of the tetrahedron (in violet).

$$
\begin{equation*}
\alpha_{\mathrm{lm}}^{t}=\arccos \left(\hat{\boldsymbol{\zeta}}_{1}^{t} \cdot \hat{\boldsymbol{\zeta}_{\mathrm{m}}^{t}}\right) \tag{2.32}
\end{equation*}
$$

and the initial angle $\alpha_{1 \mathrm{~m}}^{0}$ is computed by setting $t=0$ in Eq. (2.32).
Forces $\mathbf{f}_{21}$ and $\mathbf{f}_{21+1}$ the springs exert on intersection points $q_{21}$ and $q_{21+1}$ defined by $\zeta_{1}$ are composed of an axial component $\mathbf{f}_{\mathcal{S}}$ and two torsion components $\mathbf{f}_{\tau}$ each. A general formulation for these forces can be given by

$$
\begin{align*}
\mathbf{f}_{21} & =\mathbf{f}_{\mathcal{S}}\left(\boldsymbol{\zeta}_{1}, \alpha_{\operatorname{lm}}, \alpha_{\ln }\right) \hat{\boldsymbol{\zeta}}_{1}+\mathbf{f}_{\tau}\left(\boldsymbol{\zeta}_{1}, \alpha_{\operatorname{lm}}, \alpha_{\ln }\right) \hat{\boldsymbol{\zeta}_{\mathrm{m}}}+\mathbf{f}_{\tau}\left(\boldsymbol{\zeta}_{1}, \alpha_{\operatorname{lm}}, \alpha_{\ln }\right) \hat{\boldsymbol{\zeta}_{\mathrm{n}}}  \tag{2.33}\\
\mathbf{f}_{21+1} & =-\mathbf{f}_{21} \tag{2.34}
\end{align*}
$$

with $l \neq m \neq n$.
These components vary depending on spring functions used. Different spring functions can be used to reflect the mechanical properties of the modeled materials. In the matter of fact, spring functions can be specifically designed to obtain a better fit to mechanical properties of the modeled material.

## Linear Axial Springs

Linear springs are commonly used in ordinary mass-spring systems. They are simply an implementation of Hooke's law where force is a linear function of the spring's length:

$$
\begin{equation*}
\mathbf{f}_{21}^{t}=-k_{1}\left(\left\|\zeta_{1}^{t}\right\|-\left\|\zeta_{1}^{0}\right\|\right) \hat{\zeta}_{1}^{t} \tag{2.35}
\end{equation*}
$$

where $k_{1}$ is the stiffness constant of the linear spring.

## Higher-Order Axial Springs

Most materials exhibit a non-linear strain-stress relationship, and can be considered linear only for short strain ranges. For instance, the relation can follow a quadratic or a logarithmic curve. Spring functions can be designed depending on the properties of the modeled material by means of curve fitting. By fitting a polynomial function to the strain-stress curve, a polynomial spring function would have the following form:

$$
\begin{equation*}
\mathbf{f}_{2 \mathrm{l}}^{t}=-\left(\sum_{i=1}^{n} k_{1, \mathrm{i}}\left(\left\|\zeta_{1}^{t}\right\|-\left\|\zeta_{1}^{0}\right\|\right)^{i}\right) \hat{\zeta}_{1}^{t} \tag{2.36}
\end{equation*}
$$

where $n$ is the rank of the function and the parameters $k_{\mathrm{i}}$ are the different stiffness parameters of the polynomial function. From Eq. (2.36) one can easily conclude that more parameters have to be found for spring functions of high order $n$, complicating the parameterization task. In this work only quadratic and cubic spring functions were implemented.

The parameters $k_{\mathrm{i}}$ of the spring functions can be obtained by fitting the spring function to the strain-stress curves after converting the spring force to stress using a unit surface. But the resulting parameters in this case would be correct only if faces had a unit surface which is clearly not the case. Therefore, the parameters must be scaled using the surface of faces on which the springs exert their forces. The scaled stiffness parameters must be then used in the spring functions mentioned above.

That means, from the software engineering point of view, that memory has to be allocated for the scaled stiffness parameters in each volume element of the model. Additionally, scaling the parameters at the structure initialization phase makes the forces calculation module dependent on the mesh topology module (see Fig. 2.2) which is undesirable because it reduces the flexibility of the framework. By incorporating the surface of the face on which the spring force applies in the forces calculation step, it is possible to avoid the mentioned issues on the cost of increasing the computation time.

The unscaled stiffness parameters $E_{1}$ found by fitting the spring function to the strain-stress relation have a unit of pressure $\left[\mathrm{N} / \mathrm{m}^{2}\right]$. Using these parameters, Eq. (2.35) can be rewritten as

$$
\begin{align*}
\mathbf{f}_{21}^{t} & =-k_{1}^{t}\left(\left\|\zeta_{1}^{t}\right\|-\left\|\zeta_{1}^{0}\right\|\right) \hat{\zeta}_{1}^{t}  \tag{2.37}\\
k_{1}^{t} & =E_{1} A_{\mathrm{j}}^{t}\left|\mathbf{n}_{\mathrm{j}}^{t} \cdot \hat{\zeta}_{1}^{t}\right| \tag{2.38}
\end{align*}
$$

where $A_{\mathrm{j}}^{t}$ is the surface of the face $\mathcal{F}_{\mathrm{j}}$ on which the force acts and $\mathbf{n}_{\mathrm{j}}^{t}$ is the normal on $\mathcal{F}_{\mathrm{j}}$ all calculated at time $t$. Using Eq. (2.37) the notion $\mathbf{f}_{21}=-\mathbf{f}_{21+1}$ of Eq. (2.34) becomes invalid, because, for this case, $k_{1}^{t}$ depends on the area and normal of the face $\mathcal{F}_{\mathrm{j}}$. Therefore, the spring forces must be calculated independently for the intersection point $q_{21}$ and $q_{21+1}$ defined by axis $\zeta_{1}$. The same can be applied to Eq. (2.36) by substituting $k_{1}$ with $k_{1}^{t}$ given in Eq. (2.38). And the resulting polynomial spring function can be given with

$$
\begin{equation*}
\mathbf{f}_{21}^{t}=-\left(\sum_{i=1}^{n} k_{1, i}^{t}\left(\left\|\zeta_{1}^{t}\right\|-\left\|\zeta_{1}^{0}\right\|\right)^{i}\right) \hat{\zeta}_{1}^{t} \tag{2.39}
\end{equation*}
$$

## Linear Torsion Springs

Torsion springs act between two pairs of intersection points as demonstrated in figure 2.12.

The linear torsion spring function is given by

$$
\begin{align*}
& \mathbf{f}_{21}^{t}=-k_{\operatorname{lm}}\left(\alpha_{\operatorname{lm}}^{t}-\alpha_{\operatorname{lm}}^{0}\right) \hat{\zeta_{\mathrm{m}}^{t}}  \tag{2.40}\\
& \mathbf{f}_{2 \mathrm{~m}}^{t}=-k_{\operatorname{lm}}\left(\alpha_{\operatorname{lm}}^{t}-\alpha_{\operatorname{lm}}^{0}\right) \hat{\zeta_{1}^{t}}  \tag{2.41}\\
& \mathbf{f}_{21+1}^{t}=-\mathbf{f}_{21}^{t}  \tag{2.42}\\
& \mathbf{f}_{2 \mathrm{~m}+1}^{t}=-\mathbf{f}_{2 \mathrm{~m}}^{t} \tag{2.43}
\end{align*}
$$

where Eq. (2.32) is used to calculate angles $\alpha_{\mathrm{lm}}^{t}$ and $\alpha_{\mathrm{lm}}^{0}$.
Here, instead of using a unit vector normal to the anisotropy axis and in the plane where the angle is measured, a unit vector parallel to the second axis is used to reduce computational costs. In order to further reduce computational costs, the angle between two pairs of intersection points is around $90^{\circ}$ and can be approximated with the cosine of that angle assuming small angle variations during deformation [38].

Using the cosine of the angle instead of the angle in Eqs. (2.40) and (2.41), the torsion spring functions can be rewritten as

$$
\begin{align*}
& \mathbf{f}_{21}^{t}=-k_{\operatorname{lm}}\left(\hat{\zeta_{1}^{t}} \cdot \hat{\zeta_{\mathrm{m}}^{t}}-\hat{\zeta_{1}^{0}} \cdot \hat{\zeta_{\mathrm{m}}^{0}}\right) \hat{\zeta_{\mathrm{m}}^{t}}  \tag{2.44}\\
& \mathbf{f}_{2 \mathrm{~m}}^{t}=-k_{\operatorname{lm}}\left(\hat{\zeta_{1}^{t}} \cdot \hat{\zeta_{\mathrm{m}}^{t}}-\hat{\zeta_{1}^{0}} \cdot \hat{\zeta_{\mathrm{m}}^{0}}\right) \hat{\zeta_{1}^{t}} \tag{2.45}
\end{align*}
$$

with noticing that Eqs. (2.42) and (2.43) remain unchanged.

## Cubic Torsion Springs

The cubic torsion spring function are:

$$
\begin{align*}
& \mathbf{f}_{2 \mathrm{l}}^{t}=-\left(\sum_{\mathrm{i}=1}^{3} k_{\mathrm{lm}, \mathrm{i}}\left(\alpha_{\mathrm{lm}}^{t}-\alpha_{\mathrm{lm}}^{0}\right)^{i}\right) \hat{\zeta_{\mathrm{m}}^{t}}  \tag{2.47}\\
& \mathbf{f}_{2 \mathrm{~m}}^{t}=-\left(\sum_{\mathrm{i}=1}^{3} k_{\mathrm{lm}, \mathrm{i}}\left(\alpha_{\mathrm{lm}}^{t}-\alpha_{\mathrm{lm}}^{0}\right)^{i}\right) \hat{\zeta_{\mathrm{l}}^{t}}  \tag{2.48}\\
& \mathbf{f}_{2 \mathrm{l}+1}^{t}=-\mathbf{f}_{2 \mathrm{l}}^{t}  \tag{2.49}\\
& \mathbf{f}_{2 \mathrm{~m}+1}^{t}=-\mathbf{f}_{2 \mathrm{~m}}^{t} \tag{2.50}
\end{align*}
$$

The stiffness parameters for the torsion spring functions can be found by fitting the functions to shear strain-stress curves after converting the spring force to stress using a unit surface.

Using the same approach used for axial springs to avoid scaling the stiffness parameters, the linear torsion spring function can be given by

$$
\begin{align*}
& \mathbf{f}_{2 \mathrm{l}}^{t}=-k_{\mathrm{lm}}^{t}\left(\alpha_{\mathrm{lm}}^{t}-\alpha_{\mathrm{lm}}^{0}\right) \hat{\zeta_{\mathrm{m}}^{t}}  \tag{2.51}\\
& \mathbf{f}_{2 \mathrm{~m}}^{t}=-k_{\mathrm{lm}}^{t}\left(\alpha_{\mathrm{lm}}^{t}-\alpha_{\mathrm{lm}}^{0}\right) \hat{\zeta_{1}^{t}}  \tag{2.52}\\
& \mathbf{f}_{2 \mathrm{l}+1}^{t}=k_{\mathrm{lm}}^{t}\left(\alpha_{\mathrm{lm}}^{t}-\alpha_{\mathrm{lm}}^{0}\right) \hat{\zeta_{\mathrm{m}}^{t}}  \tag{2.53}\\
& \mathbf{f}_{2 \mathrm{~m}+1}^{t}=k_{\mathrm{lm}}^{t}\left(\alpha_{\mathrm{lm}}^{t}-\alpha_{\mathrm{lm}}^{0}\right) \hat{\zeta_{\mathrm{l}}^{t}} \tag{2.54}
\end{align*}
$$

where

$$
\begin{equation*}
k_{\operatorname{lm}}^{t}=k_{\operatorname{lm}} A_{\mathrm{j}}^{t}\left|\hat{n_{\mathrm{j}}^{t}} \cdot \hat{\zeta_{\mathrm{l}}^{t}}\right| \tag{2.55}
\end{equation*}
$$

### 2.4.2.2 Deformation Forces using Continuum Mechanics

Ordinary mass-spring systems suffer intrinsically from the problem of spring functions parameterization. The task of parameterization is not straight forward and depends usually on optimization techniques of the parameters values.

Although the parameterization method presented earlier in 2.4.2.1 replaces the optimization process with curves fitting, solving practically the parameterization problem, a big interest in incorporating the calculation of forces via the means of continuum mechanics exists. Because it offers the possibility of calculating the deformation forces using constitutive laws of the modeled materials directly removing the need for parameterization or fitting completely.

For this reason, a method to calculate the deformation forces using continuum mechanics and the theory of finite elasticity was developed. The method, we call the method of the virtual hexahedron, can be seen as a way to analytically generate spring functions that incorporate the constitutive laws of the modeled materials. The generated spring functions perform the task of bridging the worlds of continuum mechanics and mass-spring systems .

In the method of the virtual hexahedron, in every volume element, whether a tetrahedron or a hexahedron, a local coordinate system at time $t=0$ is defined. It is assumed that a deformation tensor describing the transformations of the intersection points also describes the transformations of the volume element's vertices. For that reason a virtual hexahedron with the intersection points in the middle of its surfaces and edges parallel to the three axes is defined (see Fig. 2.13). This virtual hexahedron is used to determine the forces which are applied to the intersection points.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-26.jpg?height=718&width=1426&top_left_y=1534&top_left_x=272)

Fig. 2.13 The method of virtual hexahedron, in case of a hexahedral mesh (a), the case of a tetrahedral mesh (b), the intersection points are in the middles of the surfaces of the virtual hexahedron and the edges are parallel to the axes of anisotropies.

Starting with a constitutive law defined by an energy density function $W$, the forces $\mathbf{f}_{\mathrm{i}}$ which are applied to the intersection points of the virtual hexahedron can be calculated with

$$
\begin{equation*}
\mathbf{f}_{2 \mathrm{l}}^{t}=R_{\mathrm{v}} A_{1}^{0} \mathbf{F} \mathbf{B S}^{\mathrm{T}} \mathbf{n}_{2 \mathrm{l}}^{0} \tag{2.56}
\end{equation*}
$$

where $A_{1}^{0}$ is the surface of the virtual hexahedron face containing the intersection point $q_{21}$ defined by axis $\zeta_{1}$ at time $t=0, \mathbf{n}_{21}^{0}$ is the normal on that face. $A_{1}^{0}$ can be given with

$$
\begin{equation*}
A_{1}^{0}=\left|\boldsymbol{\zeta}_{\mathrm{m}} \times \boldsymbol{\zeta}_{\mathrm{n}}\right| \quad \text { with } \quad l \neq m \neq n \tag{2.57}
\end{equation*}
$$

$R_{\mathrm{v}}$ is the ratio of the initial volume $V^{0}$ of the volume element and the initial volume of the virtual hexahedron $V_{\mathrm{h}}^{0}$ :

$$
\begin{equation*}
R_{\mathrm{v}}=\frac{V^{0}}{V_{\mathrm{h}}^{0}} \tag{2.58}
\end{equation*}
$$

B is the transformation tensor from the local volume element coordinates to the global spatial coordinates. S is the Piola-Kirchhoff stress tensor and is given with Eq. (1.7), E is the Green strain tensor which is given with Eq. (1.6)

In Eqs. (2.56) and (1.6), the deformation tensor $\mathbf{F}$ is needed. $\mathbf{F}$ is used to transform a volume element from the undeformed initial state at $t=0$ to the state at time $t$. To determine $\mathbf{F}$, two different methods have been implemented:

## Aproximation of the Deformation Tensor F

The first method has been developed to determine an approximation of the deformation tensor using exclusively the strains of axes $\zeta_{1}, \zeta_{2}$ and $\zeta_{3}$ in addition to the angles between these axes. The deformation tensor $\mathbf{F}$ can be written as the product of principle strain and shear strain:

$$
\mathbf{F}=\left(\begin{array}{ccc}
\lambda_{1} & 0 & 0  \tag{2.59}\\
0 & \lambda_{2} & 0 \\
0 & 0 & \lambda_{3}
\end{array}\right) \cdot\left(\begin{array}{ccc}
1 & \gamma_{12} & \gamma_{13} \\
\gamma_{21} & 1 & \gamma_{23} \\
\gamma_{31} & \gamma_{32} & 1
\end{array}\right)=\left(\begin{array}{ccc}
\lambda_{1} & \lambda_{1} \gamma_{12} & \lambda_{1} \gamma_{13} \\
\lambda_{2} \gamma_{21} & \lambda_{2} & \lambda_{2} \gamma_{23} \\
\lambda_{3} \gamma_{31} & \lambda_{3} \gamma_{32} & \lambda_{3}
\end{array}\right)
$$

The principle strain components $\lambda_{\mathrm{i}}$ can be approximated using the strain of the axes $\zeta_{\mathrm{i}}$ and the angles $\alpha_{\mathrm{ij}}$ between each of them (see Fig. 2.14):

$$
\begin{gather*}
\beta_{\mathrm{ij}}=\beta_{\mathrm{ji}}=\frac{1}{2}\left(\frac{\pi}{2}-\alpha_{\mathrm{ij}}\right)  \tag{2.60}\\
\lambda_{\mathrm{i}}=\frac{\left\|\boldsymbol{\zeta}_{\mathrm{i}}\right\|}{\left\|\boldsymbol{\zeta}_{\mathrm{i}}^{0}\right\|} \prod_{\mathrm{j}=1, \mathrm{j} \neq \mathrm{i}}^{3} \cos \left(\beta_{\mathrm{ij}}\right) \tag{2.61}
\end{gather*}
$$

and the shear components $\gamma_{\mathrm{ij}}$ are calculated as follows:

$$
\begin{equation*}
\gamma_{\mathrm{ij}}=\gamma_{\mathrm{ji}}=\tan \left(\beta_{\mathrm{ij}}\right), \quad i \neq j \tag{2.62}
\end{equation*}
$$

Eq. (2.61) is an approximation. Obtaining the exact solution for the principle strain components using the axis strains and the angles is complex. However, if at least one axis is orthogonal to one of the other axis, the exact solution is obtained.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-28.jpg?height=752&width=732&top_left_y=740&top_left_x=599)

Fig. 2.14 Deformation composed of axial strain and shear strain.

Determining the deformation tensor using shape functions
The second method is a standard procedure known from the finite elements method (FEM). It uses shape functions to calculate a linear interpolation of the deformation tensor from the position of the vertices of the volume element.

The deformation of a linear (8-nodes) hexahedron can be interpolated using shape functions [49, 1]. After transforming the global spatial coordinates of vertices to the local coordinate system of the hexahedron, the transformed coordinates ( $x_{\mathrm{i}}, y_{\mathrm{i}}, z_{\mathrm{i}}$ ) of the vertices $p_{\mathrm{i}}$ numbered as shown in Figure 2.15 are used to calculate the shape functions $N_{\mathrm{i}}$ according to

$$
\begin{equation*}
N_{\mathrm{i}}(\boldsymbol{\xi})=N_{\mathrm{i}}(\varsigma, \eta, \mu)=\frac{1}{8}\left(1+\varsigma_{\mathrm{i}} \varsigma\right)\left(1+\eta_{\mathrm{i}} \eta\right)\left(1+\mu_{\mathrm{i}} \mu\right) \tag{2.63}
\end{equation*}
$$

And the deformation tensor is given by [49]
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-29.jpg?height=517&width=558&top_left_y=461&top_left_x=486)
(a)

| I | $\xi_{I}$ | $\eta_{I}$ | $\varsigma_{I}$ |
| :---: | :---: | :---: | :---: |
| 1 | -1 | -1 | -1 |
| 2 | 1 | -1 | -1 |
| 3 | 1 | 1 | -1 |
| 4 | -1 | 1 | -1 |
| 5 | -1 | -1 | 1 |
| 6 | 1 | -1 | 1 |
| 7 | 1 | 1 | 1 |
| 8 | -1 | 1 | 1 |

(b)

Fig. 2.15 Linear hexahedral element and its nodal natural coordinates. Linear hexahedral element (a), nodal natural coordinates for linear hexahedron element (b).

$$
\begin{equation*}
\mathbf{F}=\sum_{\mathrm{i}=1}^{8} \mathbf{x}_{\mathrm{i}} \otimes \frac{\partial N_{\mathrm{i}}}{\partial \mathbf{X}} \tag{2.64}
\end{equation*}
$$

where

$$
\begin{align*}
\frac{\partial N_{\mathrm{i}}}{\partial \mathbf{X}} & =\left(\frac{\partial \mathbf{X}}{\partial \boldsymbol{\xi}}\right)^{-\mathrm{T}} \frac{\partial N_{\mathrm{i}}}{\partial \boldsymbol{\xi}}  \tag{2.65}\\
\frac{\partial \mathbf{X}}{\partial \boldsymbol{\xi}} & =\sum_{\mathrm{i}=1}^{8} \mathbf{X}_{\mathrm{i}} \otimes \frac{\partial N_{\mathrm{i}}}{\partial \boldsymbol{\xi}} \tag{2.66}
\end{align*}
$$

and therefore the elements of the deformation tensor are given by

$$
\begin{equation*}
F_{\mathrm{IJ}}=\sum_{\mathrm{i}=1}^{8} x_{\mathrm{i}, \mathrm{I}} \frac{\partial N_{\mathrm{i}}}{\partial X_{\mathrm{J}}} \tag{2.67}
\end{equation*}
$$

The deformation tensor of a 4-node tetrahedron can be interpolated using shape functions $[49,1]$. The local numbering of the vertices of the tetrahedrons is defined by choosing an arbitrary node as the first and then numbering the remaining nodes counterclockwise as seen from the first node in Figure 2.16. As in the hexahedral case, the global spatial coordinates vertices have to be transformed to the local coordinate system. Using the transformed coordinates ( $x_{\mathrm{i}}, y_{\mathrm{i}}, z_{\mathrm{i}}$ ) of the ordered vertices $p_{\mathrm{i}}(i=1, \ldots, 4)$ the shape functions $N_{\mathrm{i}}$ are defined by

$$
\mathbf{X}=\left(\begin{array}{l}
1  \tag{2.68}\\
X \\
Y \\
Z
\end{array}\right)=\mathbf{A}\left(\begin{array}{l}
N_{1} \\
N_{2} \\
N_{3} \\
N_{4}
\end{array}\right)
$$

where

$$
\mathbf{A}=\left(\begin{array}{cccc}
1 & 1 & 1 & 1  \tag{2.69}\\
x_{1} & x_{2} & x_{3} & x_{4} \\
y_{1} & y_{2} & y_{3} & y_{4} \\
z_{1} & z_{2} & z_{3} & z_{4}
\end{array}\right)
$$

By inverting matrix A, the shape functions $N_{\mathrm{i}}$ can be written as a linear combination of the components of X with

$$
\begin{equation*}
N_{\mathrm{i}}(x, y, z)=m_{1, \mathrm{i}}+m_{2, \mathrm{i}} X+m_{3, \mathrm{i}} Y+m_{4, \mathrm{i}} Z \tag{2.70}
\end{equation*}
$$

The elements $m_{\mathrm{IJ}}$ of $\mathbf{A}^{-1}$ can be calculated using the following relation:

$$
\begin{equation*}
m_{\mathrm{IJ}}=(-1)^{I+J} \frac{1}{\operatorname{det}(\mathbf{A})} \hat{\mathbf{A}}_{\mathrm{IJ}} \tag{2.71}
\end{equation*}
$$

where $\hat{\mathbf{A}}_{\mathrm{IJ}}$ are the minors of $\mathbf{A}$. Finally, the tetrahedron deformation tensor can then be given by

$$
\begin{equation*}
F_{\mathrm{IJ}}=\sum_{\mathrm{i}=1}^{4} x_{\mathrm{i}, \mathrm{I}} \frac{\partial N_{\mathrm{i}}}{\partial X_{\mathrm{J}}} \tag{2.72}
\end{equation*}
$$

![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-30.jpg?height=603&width=712&top_left_y=1826&top_left_x=632)

Fig. 2.16 A 4-node tetrahedral element and the nodes numbering scheme

Unlike the approximation method described above, this method is well-established and has less computational costs. Therefore, it is used as the method of choice for calculating the deformation tensor. Back to Eq. (2.56):

$$
\mathbf{f}_{21}^{t}=R_{\mathrm{v}} A_{1}^{0} \mathbf{F B S}^{\mathrm{T}} \mathbf{n}_{21}^{0}
$$

$\mathbf{S}$ is defined for the local coordinate system in the volume element, the normal vectors $\mathbf{n}_{21}^{0}$ on the surfaces of the virtual hexahedron are:

$$
\begin{align*}
& \mathbf{n}_{0}^{0}=-\mathbf{n}_{1}^{0}=\left(\begin{array}{l}
1 \\
0 \\
0
\end{array}\right)=\mathbf{B}^{-1} \hat{\zeta_{1}^{0}} \\
& \mathbf{n}_{2}^{0}=-\mathbf{n}_{3}^{0}=\left(\begin{array}{l}
0 \\
1 \\
0
\end{array}\right)=\mathbf{B}^{-1} \hat{\zeta_{2}^{0}}  \tag{2.73}\\
& \mathbf{n}_{4}^{0}=-\mathbf{n}_{5}^{0}=\left(\begin{array}{l}
0 \\
0 \\
1
\end{array}\right)=\mathbf{B}^{-1} \hat{\zeta_{3}^{0}}
\end{align*}
$$

Using these normal vectors and that $\mathbf{S}=\mathbf{S}^{\mathrm{T}}$, the force $\mathbf{f}_{21}$ acting on intersection point $q_{\mathrm{j}}$ can be decomposed in three force components along the axes. Substituting in Eq. (2.56) and using $\lambda=l+1$ gives:

$$
\begin{align*}
\mathbf{f}_{21} & =R_{\mathrm{v}} A_{1}^{0} \mathbf{F} \mathbf{B} \mathbf{S n}_{21}^{0} \\
& =R_{\mathrm{v}} A_{1}^{0} \mathbf{F} \mathbf{B}\left(S_{1 \lambda} \mathbf{n}_{1}^{0}+S_{2 \lambda} \mathbf{n}_{3}^{0}+S_{3 \lambda} \mathbf{n}_{5}^{0}\right) \\
& =R_{\mathrm{v}} A_{1}^{0} \mathbf{F B}\left(S_{1 \lambda} \mathbf{B}^{-1} \hat{\boldsymbol{\zeta}_{1}^{0}}+S_{2 \lambda} \mathbf{B}^{-1} \hat{\boldsymbol{\zeta}_{2}^{0}}+S_{3 \lambda} \mathbf{B}^{-1} \hat{\boldsymbol{\zeta}_{3}^{0}}\right) \\
\mathbf{f}_{21} & =R_{\mathrm{v}} A_{1}^{0}\left(S_{1 \lambda} \hat{\boldsymbol{\zeta}_{1}}+S_{2 \lambda} \hat{\boldsymbol{\zeta}_{2}}+S_{3 \lambda} \hat{\boldsymbol{\zeta}_{3}}\right) \tag{2.74}
\end{align*}
$$

which is a function suitable for code implementation in contrast to Eq. (2.56).
In comparison to the concept of axial and angular springs, the terms $A_{1}^{0} S_{11} \hat{\boldsymbol{\zeta}}_{1}$, $A_{2}^{0} S_{22} \hat{\boldsymbol{\zeta}_{2}}$ and $A_{3}^{0} S_{33} \hat{\boldsymbol{\zeta}_{3}}$ represent the forces of axial springs, while the remaining terms represent components of torsion spring forces.

### 2.4.2.3 Volume Preservation

Developing a method for volume preservation was a very important part of this work, due to high interest in modeling objects that retain constant volume during deformation, like cardiac myocytes.

Bourguignon et al. [38] used volume preservation springs acting on the volume elements particles. The function Bourguignon used to calculate the volume preservation force at vertex $p_{\mathrm{i}}$ of a hexahedron is given with

$$
\begin{equation*}
\mathbf{f}_{\mathrm{i}}^{t}=-\left(k_{\mathrm{s}}\left(\left\|\boldsymbol{\xi}_{\mathrm{i}}^{t}\right\|-\left\|\boldsymbol{\xi}_{\mathrm{i}}^{0}\right\|\right)+k_{\mathrm{d}} \boldsymbol{\vartheta}_{\mathrm{i}}^{t} \cdot \hat{\boldsymbol{\xi}}_{\mathrm{i}}^{t}\right) \hat{\boldsymbol{\xi}}_{\mathrm{i}}^{t} \tag{2.75}
\end{equation*}
$$

In case of using tetrahedrons, the function is given with

$$
\begin{equation*}
\mathbf{f}_{\mathrm{i}}^{t}=-k_{\mathrm{s}}\left(\sum_{\mathrm{j}=1}^{4}\left\|\boldsymbol{\xi}_{\mathrm{j}}^{t}\right\|-\sum_{\mathrm{j}=1}^{4}\left\|\boldsymbol{\xi}_{\mathrm{j}}^{0}\right\|\right) \hat{\boldsymbol{\xi}_{\mathrm{i}}^{t}} \tag{2.76}
\end{equation*}
$$

where $\boldsymbol{\xi}_{\mathrm{i}}^{t}=\mathrm{x}_{\mathrm{b}}^{t}-\mathrm{x}_{\mathrm{i}}^{t}$, and $\mathrm{x}_{\mathrm{b}}$ is the barycenter of the volume element and can be calculated using Eq. (2.8) for hexahedrons and Eq. (2.22 ) for tetrahedrons. The unit vector $\hat{\boldsymbol{\xi}}_{\mathrm{i}}^{t}$ is given by

$$
\begin{equation*}
\hat{\boldsymbol{\xi}}_{\mathrm{i}}^{t}=\frac{\boldsymbol{\xi}_{\mathrm{i}}^{t}}{\left\|\boldsymbol{\xi}_{\mathrm{i}}^{t}\right\|} \tag{2.77}
\end{equation*}
$$

and for the relative velocity of an intersection point $\boldsymbol{\vartheta}^{t}$ we can write:

$$
\begin{equation*}
\boldsymbol{\vartheta}_{\mathrm{i}}^{t}=\frac{d \mathbf{x}_{\mathrm{i}}^{t}}{d t}-\frac{d \mathbf{x}_{\mathrm{b}}^{t}}{d t} \tag{2.78}
\end{equation*}
$$

These volume preservation springs exert penalty forces on particles proportional to the strain of the springs.

This approach did not lead to satisfying results. The variations of volume of the elements were too high. The parameterization of the functions was an additional drawback of this method.

An adaptive volume preservation spring function was proposed with the hope of obtaining a better volume preservation than achieved using the Bourguignon spring functions in Eqs. (2.75) and (2.76).

The main idea behind the adaptive spring functions method is to make the value $k_{\mathrm{s}}$ in Eq. (2.76) adaptive with time, and to include an adaptation algorithm that continuously updates the value of $k_{\mathrm{s}}$ to reduce the volume difference $\Delta V$ between the original and the current volume of the element, $V^{0}$ and $V^{t}$ respectively:

$$
\begin{equation*}
\Delta V=V^{t}-V^{0} \tag{2.79}
\end{equation*}
$$

We will use the symbol $\hat{k_{\mathrm{s}}}$ to distinguish the adaptive $k_{\mathrm{v}}$ from the constant counterpart.

The Least Mean Squares (LMS) algorithm was used to update the value of $k_{\mathrm{s}}$, using $\Delta V$ as an adaptation variable. For example for a tetrahedron:

$$
\begin{align*}
& \mathbf{f}_{\mathrm{i}}^{t}=-{\hat{k_{\mathrm{s}}}}^{t}\left(\sum_{\mathrm{j}=1}^{4}\left\|\boldsymbol{\xi}_{\mathrm{j}}^{t}\right\|-\sum_{\mathrm{j}=1}^{4}\left\|\boldsymbol{\xi}_{\mathrm{j}}^{0}\right\|\right) \hat{\boldsymbol{\xi}}_{\mathrm{i}}^{t}  \tag{2.80}\\
& {\hat{k_{\mathrm{s}}}}^{t+h}={\hat{k_{\mathrm{s}}}}^{t}+\mu \Delta V \sum_{\mathrm{j}=1}^{4}\left\|\boldsymbol{\xi}_{\mathrm{j}}^{t}\right\| \tag{2.81}
\end{align*}
$$

where $\hat{k}_{\mathrm{s}}^{t}$ is the current adaptive stiffness, $\hat{k}_{\mathrm{s}}^{t+h}$ is the adaptive stiffness after a time step $h, \mu$ is the adaptation constant or the LMS step size.

Although this adaptive springs stiffness showed promising results in initial experiments with a small number of elements, this method was not further investigated.

To enforce volume preservation in each element of the model, a method used in continuum mechanics is finally applied to our mass-spring system. According to continuum mechanics, volume preservation can be introduced to an element by adding a volumetric energy density term $W_{\mathrm{v}}$ to the potential energy density function of the element $W[49,1]$ :

$$
\begin{equation*}
\hat{W}=W+W_{\mathrm{v}} \tag{2.82}
\end{equation*}
$$

$W_{\mathrm{v}}$ depends on the change of volume and can be formulated using the ratio of the current to the initial volume. There are many possible ways to formulate $W_{\mathrm{v}}$. In this work the following form was used:

$$
\begin{equation*}
W_{\mathrm{v}}=p\left(\frac{\Delta V}{V^{0}}\right)^{2} \tag{2.83}
\end{equation*}
$$

We know that the determinant of the deformation tensor $\mathbf{F}$ of a an element is

$$
\begin{equation*}
\operatorname{det}(\mathbf{F})=\frac{V^{t}}{V^{0}} \tag{2.84}
\end{equation*}
$$

and we can then write

$$
\begin{equation*}
\frac{\Delta V}{V^{0}}=\frac{V^{t}-V^{0}}{V^{0}}=\operatorname{det}(\mathbf{F})-1 \tag{2.85}
\end{equation*}
$$

By substituting Eq. (2.85) in Eq. (2.83) we get

$$
\begin{equation*}
W_{\mathrm{v}}=p(\operatorname{det}(\mathbf{F})-1)^{2} \tag{2.86}
\end{equation*}
$$

The volumetric term $W_{\mathrm{v}}$ represents the potential energy resulting from a change in volume.

A common method to achieve incompressibility is to minimize the total energy $\hat{W}$ with respect to the displacement. In that case a variation calculus problem with constraints has to be solved, and the coefficient $p$ can be seen as a Lagrange multiplier [49, 1].

In this work, a different approach adapted to the mass-spring system is used. Here, volume preservation penalty forces, that depend on the $\Delta V$, are derived from a Piola-Kirchhoff tensor $\mathbf{S}$ of volume preservation which is in turn derived from the volumetric energy density term $W_{\mathrm{v}}$ according to

$$
\begin{equation*}
S_{\mathrm{ij}}=\frac{\partial W_{\mathrm{v}}}{\partial E_{\mathrm{ij}}} \tag{2.87}
\end{equation*}
$$

where $E_{\mathrm{ij}}$ are the Green strain tensor elements.
To compute $S_{\mathrm{ij}}, W_{\mathrm{v}}$ must be transformed from a function of the deformation tensor elements $F_{\mathrm{ij}}$ to a function of the Green strain tensor elements $E_{\mathrm{ij}}$. By expanding $W_{\mathrm{v}}$ :

$$
\begin{equation*}
W_{\mathrm{v}}=p(\operatorname{det}(\mathbf{F})-1)^{2}=p\left(\operatorname{det}(\mathbf{F})^{2}-2 \operatorname{det}(\mathbf{F})+1\right) \tag{2.88}
\end{equation*}
$$

In general, given two matrices $\mathbf{A}$ and $\mathbf{B}$, one can write:

$$
\begin{align*}
\operatorname{det}(\mathbf{A}) & =\operatorname{det}\left(\mathbf{A}^{\mathrm{T}}\right)  \tag{2.89}\\
\operatorname{det}(\mathbf{A B}) & =\operatorname{det}(\mathbf{A}) \operatorname{det}(\mathbf{B})  \tag{2.90}\\
\operatorname{det}(\mathbf{A A}) & =\operatorname{det}(\mathbf{A}) \operatorname{det}(\mathbf{A})  \tag{2.91}\\
& =\operatorname{det}(\mathbf{A}) \operatorname{det}\left(\mathbf{A}^{\mathrm{T}}\right)  \tag{2.92}\\
& =\operatorname{det}\left(\mathbf{A} \mathbf{A}^{\mathrm{T}}\right)=\operatorname{det}(\mathbf{A})^{2} \tag{2.93}
\end{align*}
$$

The Green strain tensor and the deformation tensor are related as follows:

$$
\begin{align*}
& \mathbf{E}=\frac{1}{2}\left(\mathbf{F F}^{\mathrm{T}}-\mathbf{I}\right) \Rightarrow  \tag{2.94}\\
& \mathbf{F F}^{\mathbf{T}}=2 \mathbf{E}+\mathbf{I} \tag{2.95}
\end{align*}
$$

Using Eq. (2.93) and Eq. (2.95) we obtain:

$$
\begin{equation*}
\operatorname{det}(\mathbf{F})^{2}=\operatorname{det}\left(\mathbf{F F}^{\mathrm{T}}\right)=\operatorname{det}(2 \mathbf{E}+\mathbf{I}) \tag{2.96}
\end{equation*}
$$

and Eq. 2.88 can be rewritten as

$$
\begin{align*}
W_{\mathrm{v}} & =p(\operatorname{det}(2 \mathbf{E}+\mathbf{I})-2 \sqrt{\operatorname{det}(2 \mathbf{E}+\mathbf{I})}+1)  \tag{2.97}\\
& =p(\Omega-2 \sqrt{\Omega}+1) \tag{2.98}
\end{align*}
$$

where

$$
\begin{align*}
\Omega=\operatorname{det}(2 \mathbf{E}+\mathbf{I})= & \left(2 E_{11}+1\right)\left(2 E_{22}+1\right)\left(2 E_{33}+1\right) \\
& +8\left(E_{12} E_{23} E_{31}\right)+8\left(E_{21} E_{32} E_{13}\right) \\
& -4\left(E_{23} E_{32}\left(2 E_{11}+1\right)\right)  \tag{2.99}\\
& -4\left(E_{13} E_{31}\left(2 E_{22}+1\right)\right) \\
& -4\left(E_{12} E_{21}\left(2 E_{33}+1\right)\right)
\end{align*}
$$

and finally, the elements of the tensor $\mathbf{S}$ can be given with

$$
\begin{equation*}
S_{\mathrm{ij}}=p \frac{\partial \Omega}{\partial E_{\mathrm{ij}}}\left(1-\frac{1}{\sqrt{\Omega}}\right) \tag{2.100}
\end{equation*}
$$

where

$$
\frac{\partial \Omega}{\partial E_{\mathrm{ij}}}= \begin{cases}2\left(2 E_{\mathrm{jj}}+1\right)\left(2 E_{\mathrm{kk}}+1\right)-8 E_{\mathrm{kj}} E_{\mathrm{jk}} & i=j \\ 8\left(E_{\mathrm{jk}} E_{\mathrm{ki}}\right)-4 E_{\mathrm{ji}}\left(2 E_{\mathrm{kk}}-1\right) & i \neq j\end{cases}
$$

The resulting volume preservation forces can be calculated using the method of virtual hexahedron represented by Eq. (2.56) or Eq. (2.74) and the volume preservation Piola-Kirchhoff tensor elements $S_{\mathrm{ij}}$ can be calculated with Eq. (2.100).

If the volume preservation effect should act in the direction of the defined anisotropies in an element, these forces can be applied to the intersections points and then distributed to the vertices using Eq. (2.29). In the other case where the volume preservation effect should be isotropic, a coefficient matrix $C$ that guarantees an equal share of volume preservation forces to all vertices of an element is generated and used in Eq. (2.29) to distribute the forces to the vertices.

It is important to mention that some constitutive laws include terms to enforce volume preservation. In that case using the constitutive law with the method of the virtual hexahedron to calculate the deformation forces is enough to ensure the volume preservation, meaning that no additional separate treatment is needed.

### 2.4.2.4 Volume Control

In some applications a method to control the volume of the element is desired (e.g. see Section 5.4.3). In these applications the volume of the element should not remain constant, but rather change towards a predefined value. To add that possibility to our framework, the method used to enforce volume preservation by
adding a volume preservation energy density term was improved to allow for volume control.

Starting with Eq. (2.86), one can write:

$$
\begin{equation*}
W_{\mathrm{v}}=p(\operatorname{det}(\mathbf{F})-r)^{2} \tag{2.101}
\end{equation*}
$$

where $r$ is the ratio of the target volume $V^{\infty}$ to the initial volume $V^{0}$ of the element:

$$
\begin{equation*}
r=\frac{V^{\infty}}{V^{0}} \tag{2.102}
\end{equation*}
$$

By setting the target volume to the initial volume in Eq. (2.102) we get $r=1$, turning Eq. (2.101) back to Eq. (2.86).

By following the same procedure used to derive the volume preservation forces, we can obtain the equations for the volume control forces. We start by expanding Eq. (2.101):

$$
\begin{equation*}
W_{\mathrm{v}}=p\left(\operatorname{det}(\mathbf{F})^{2}-2 r \operatorname{det}(\mathbf{F})+r^{2}\right) \tag{2.103}
\end{equation*}
$$

Then by substituting the terms of the deformation tensor $\mathbf{F}$ with the Green strain tensor E using the Eq. (2.96) we get:

$$
\begin{align*}
W_{\mathrm{v}} & =p\left(\operatorname{det}(2 \mathbf{E}+\mathbf{I})-2 r \sqrt{\operatorname{det}(2 \mathbf{E}+\mathbf{I})}+r^{2}\right)  \tag{2.104}\\
& =p\left(\Omega-2 r \sqrt{\Omega}+r^{2}\right) \tag{2.105}
\end{align*}
$$

where $\Omega$ is given in Eq. (2.99). Finally the Piola-Kirchhoff stress tensor elements can be calculated using

$$
\begin{equation*}
S_{\mathrm{ij}}=p \frac{\partial \Omega}{\partial E_{\mathrm{ij}}}\left(1-\frac{r}{\sqrt{\Omega}}\right) \tag{2.106}
\end{equation*}
$$

and the resulting intersection points forces can be calculated using the method of virtual hexahedron represented by Eq. (2.56) or Eq. (2.74).

### 2.4.2.5 Material Specific Friction

Anisotropic friction specific for the different material in each volume element of the modeled object was implemented using the relative velocity of the intersection points:

$$
\begin{equation*}
\dot{\zeta}_{1}^{t}=\frac{d}{d t}\left(\mathbf{x}_{21}^{t}-\mathbf{x}_{21+1}^{t}\right) \tag{2.107}
\end{equation*}
$$

Bourguignon et al. introduced linear friction by applying friction forces to the intersection points using the following function:

$$
\begin{equation*}
\mathbf{f}_{21}^{t}=-k_{\mathrm{d}}\left(\dot{\zeta_{1}^{t}} \cdot \hat{\zeta}_{1}^{t}\right) \hat{\zeta}_{1}^{t} \tag{2.108}
\end{equation*}
$$

where $k_{\mathrm{d}}$ is the friction coefficient, or as Bourguignon calls it, the damping parameter of spring $\mathcal{S}_{1}$. This kind of friction was also used in the work of M. Mohr [39] for anisotropy springs in his hybrid-mass-spring model.

In our model, the following function was used to introduce linear friction:

$$
\begin{equation*}
\mathbf{f}_{\mathrm{j}}^{t}=-k_{\mathrm{d}}^{t}\left(\dot{\zeta_{1}^{t}} \cdot \hat{\zeta}_{1}^{t}\right) \hat{\zeta}_{1}^{t} \tag{2.109}
\end{equation*}
$$

where $k_{\mathrm{d}}^{t}$ is given by

$$
\begin{equation*}
k_{\mathrm{d}}^{t}=k_{\mathrm{d}} S_{\mathrm{j}}^{t}\left|\hat{n_{\mathrm{j}}^{t}} \cdot \hat{\zeta_{\mathrm{l}}^{t}}\right| \tag{2.110}
\end{equation*}
$$

$S_{\mathrm{j}}^{t}$ is the surface of the face containing intersection point $q_{\mathrm{j}}$ that the axis $\zeta_{\mathrm{l}}$ defines and $n_{\mathrm{j}}^{t}$ is the normal on that face. When using the method of the virtual hexahedron the following formula can be used:

$$
\begin{equation*}
\mathbf{f}_{21}^{t}=-R_{\mathrm{v}}\left(k_{\mathrm{d}} \dot{\zeta_{1}^{t}} \cdot \hat{\zeta_{1}^{t}}\right) A_{1}^{0} \hat{\zeta_{1}^{t}} \tag{2.111}
\end{equation*}
$$

where $A_{1}^{0}$ is the surface of the virtual hexahedron face containing $q_{21}$ of axis $\zeta_{1}$ defined at time $t=0$. $A_{1}^{0}$ can be calculated using Eq. (2.57).

### 2.4.3 Active Forces

To model tensions generated by internal processes, like the contraction tension in a muscle along its fibers, mathematical models of these internal processes can be used to calculate the tension these processes generate in each volume element. These tensions can also be measured using special measurement techniques and then introduced to the mechanical modeling framework. Depending on the process, the resulting force might be isotropic affecting all particles of the volume element in the same way, or anisotropic. In the isotropic case, the tension $\mathbf{t}_{\mathrm{j}}$ calculated for each face of the volume element is transformed to forces using the surface of the face $S_{\mathrm{j}}$ and distribute it equally to the particles at the vertices $p_{\mathrm{i}}$ of the face $S_{\mathrm{j}}$ :

$$
\begin{equation*}
\mathbf{f}_{\mathrm{i}}=\frac{1}{n} \mathbf{t}_{\mathrm{j}} \mathbf{n}_{\mathrm{j}} \tag{2.112}
\end{equation*}
$$

where $n$ is the number of vertices defining the face, and $\mathbf{n}_{\mathrm{j}}$ is the normal on that face. In the anisotropic case, tension $\mathbf{t}_{1}$ generated along the axis $\zeta_{1}$ is set to the intersection points defined by the axis with

$$
\begin{align*}
\mathbf{f}_{2 \mathrm{l}} & =\frac{1}{2} \mathbf{t}_{1} S_{2 \mathrm{l}}^{t} \hat{\boldsymbol{\zeta}}_{1}  \tag{2.113}\\
\mathbf{f}_{2 \mathrm{l}+1} & =-\frac{1}{2} \mathbf{t}_{1} S_{2 \mathrm{l}+1}^{t} \hat{\boldsymbol{\zeta}}_{1} \tag{2.114}
\end{align*}
$$

where, $S_{21}^{t}$ is the surface of the face where axis $\zeta_{1}$ defines the intersection point $q_{21}$, and $S_{21}^{t}$ is the surface of the face where the same axis defines the second intersection point $q_{21+1}$, both at time $t$. And when using the method of virtual hexahedron we get:

$$
\begin{align*}
\mathbf{f}_{2 \mathrm{l}} & =\frac{1}{2} R_{\mathrm{v}} \mathbf{t}_{\mathrm{l}} A_{2 \mathrm{l}}^{t} \hat{\zeta}_{\mathrm{l}}^{t}  \tag{2.115}\\
\mathbf{f}_{2 \mathrm{l}+1} & =-\frac{1}{2} R_{\mathrm{v}} \mathbf{t}_{\mathrm{l}} A_{2 \mathrm{l}+1}^{t} \hat{\zeta}_{\mathrm{l}}^{t} \tag{2.116}
\end{align*}
$$

where $A_{21}^{t}$ and $A_{21+1}^{t}$ are the surfaces of the virtual hexahedron faces containing intersection points $q_{21}$ and $q_{21+1}$ respectively, that axis $\zeta_{1}$ defines computed at time $t$.

Internal processes that generate force and not tension, like the myocardial tension development where the count of myofibrils is independent of the deformation, the initial surface $S^{0}$ (in spring function formulation) or $A^{0}$ (in the virtual hexahedron method formulation) is used instead of the surface at time $t$ to calculate the force resulting from the internal process.

### 2.4.4 External Forces

In contrast with internal forces, external forces are set directly to mass particles. They do not depend on the state of volume elements to which these particles belong. These forces can be part of the environment like gravity forces, or part of the simulation setup, like initial velocity.

### 2.4.4.1 External Acceleration

If the application should require the setting of external acceleration to the mass particles, this can be done by calculating the forces that correspond to the acceleration and assign the forces to the particles. For a particle $p_{\mathrm{i}}$, the force resulting from an external acceleration a can be given according to Newton's second law of motion:

$$
\begin{equation*}
\mathbf{f}_{\mathrm{i}}=m_{\mathrm{i}} \mathbf{a} \tag{2.117}
\end{equation*}
$$

where $m_{\mathrm{i}}$ is the mass of $p_{\mathrm{i}}$.
Loading the particles with gravity is just a special case of the general external acceleration where the acceleration vector $\mathbf{a}$ is the gravitational acceleration vector g .

### 2.4.4.2 Global Friction

Linear friction can also be introduced to the system using a friction coefficient $\mu \geq 0$. When $\mu=0$ no friction is modeled. In the case $\mu>0$, each particle $p_{\mathrm{i}}$ becomes subject to a global friction force $\mathbf{f}_{\mathrm{i}}$ given with

$$
\begin{equation*}
\mathbf{f}_{\mathrm{i}}=-\mu \mathbf{v}_{\mathrm{i}} \tag{2.118}
\end{equation*}
$$

where $\mathbf{v}_{\mathrm{i}}$ is the velocity of $p_{\mathrm{i}}$.

### 2.4.4.3 External Pressure

Constant or time dependent pressure $P^{t}$ can be set in the defined cavities. The pressure is transformed to forces that affect vertices of the cavity faces. For a face $\mathcal{F}_{\mathrm{j}}$ defined by vertices $p_{\mathrm{i}}(i=1, \ldots, n)$ the following rule is used:

$$
\begin{equation*}
\mathbf{f}_{\mathrm{i}}=-\frac{1}{n} P^{t} A_{\mathrm{j}}^{t} \mathbf{n}_{\mathrm{j}} \tag{2.119}
\end{equation*}
$$

where $A_{\mathrm{j}}^{t}$ is the surface of the face $\mathcal{F}_{\mathrm{j}}$ at time $t$ and $\mathbf{n}_{\mathrm{j}}$ is the normal on the face pointing towards the cavity.

### 2.4.4.4 Initial Velocity

Many mechanical modeling applications require giving the modeled object an initial velocity right at the beginning of the simulation. This can be done by setting the velocity vectors of all particles directly to the chosen initial velocity vector $\mathbf{v}^{\mathbf{0}}$ which is a $3 n$ vector and $n$ is the number of particles.

### 2.5 Time Integration

In this section, the equations of motion and the resulting system of coupled ordinary differential equations (ODE)s, implemented schemes for numerical time integration and several other topics related to time integration are presented.

### 2.5.1 Equations of Motion

The coordinates $\mathbf{x}_{\mathrm{i}}=\left(x_{\mathrm{i}}, y_{\mathrm{i}}, z_{\mathrm{i}}\right)$ of particles $p_{\mathrm{i}}(i=1, \ldots, n)$, where $n$ is the count of all particles of the object, along with the coordinates $\mathbf{x}_{\text {i,init }}$ describe the deformation state of the object at any given time $t$. Using $\mathbf{x}_{\mathrm{i}}$, we can build the $3 n$ dimensional vector $\mathbf{u}$ :

$$
\begin{equation*}
\mathbf{u}=\left(x_{1}, y_{1}, z_{1}, x_{2}, y_{2}, z_{2}, \ldots, x_{\mathrm{n}}, y_{\mathrm{n}}, z_{\mathrm{n}}\right) \tag{2.120}
\end{equation*}
$$

$\mathbf{u}$ describes the trajectory of all particles. Similarly, we can define the initial trajectories vector $\mathbf{u}_{\text {init }}$.

The particles' motion is defined by a set of $3 n$ coupled second-order differential equations:

$$
\begin{equation*}
\mathbf{M} \frac{d^{2}}{d t^{2}} \mathbf{u}+\mathbf{F}\left(\mathbf{u}, \mathbf{u}_{\text {init }}, \frac{d}{d t} \mathbf{u}, t\right)=0 \tag{2.121}
\end{equation*}
$$

with the boundary conditions:

$$
\begin{aligned}
\left.\mathbf{u}\right|_{(\mathrm{t}=0)} & =\mathbf{u}^{0} \\
\left.\frac{d}{d t} \mathbf{u}\right|_{(\mathrm{t}=0)} & =\left.\mathbf{v}\right|_{(\mathrm{t}=0)}=\mathbf{v}^{0}
\end{aligned}
$$

where

$$
\begin{equation*}
\mathbf{F}\left(\mathbf{u}, \mathbf{u}_{\text {init }}, \frac{d}{d t} \mathbf{u}, t\right)=\mathbf{F}_{\mu}\left(\frac{d}{d t} \mathbf{u}\right)+\mathbf{F}_{\mathrm{d}}\left(\mathbf{u}, \mathbf{u}_{\text {init }}\right)+\mathbf{F}_{\mathrm{a}}(t) \tag{2.122}
\end{equation*}
$$

and $\mathbf{M}$ is the system's $3 n \times 3 n$ mass tensor, $\mathbf{F}_{\mu}$ expresses friction forces, $\mathbf{F}_{\mathrm{d}}$ expresses passive forces resulting from the deformation of the body and $\mathbf{F}_{a}$ the time-dependent active forces. Here, a distigtion between the initial trajectories $\mathbf{u}_{\text {init }}$ resulting during the structure initialization phase, and $\mathbf{u}^{0}$ which is the trajectories vector at $t=0$ is made. $\mathbf{u}^{0}$ could differ from $\mathbf{u}_{\text {init }}$ if an initial displacement $\Delta \mathbf{u}$ was introduced to the system:

$$
\begin{equation*}
\mathbf{u}_{\text {init }}=\mathbf{u}^{0}+\Delta \mathbf{u} \tag{2.124}
\end{equation*}
$$

$\mathbf{u}_{\text {init }}$ does not play a role in the derivations of the following equations, therefore it will be ommited from the notation. The nature of these boundary conditions makes the ordinary differential equation Eq. (2.121) an initial value problem.

Every ordinary differential equation of order $m$ can be separated into a system of $m$ coupled partial differential equation of first order. According to this, the motion equations of the particles can be rewritten as $2 \times 3 n$ first-order equations:

$$
\begin{align*}
\frac{d}{d t} \mathbf{u} & =\mathbf{v}  \tag{2.125}\\
\frac{d}{d t} \mathbf{v} & =\mathbf{M}^{-1} \mathbf{F}\left(\mathbf{u}, \frac{d}{d t} \mathbf{u}, t\right) \tag{2.126}
\end{align*}
$$

Time integration is solving these equations for $\mathbf{u}$.
There are several methods for numerical time integration of initial value problem ODEs. However, the underlying idea of any of these numerical methods is rewriting $d \mathbf{u}$ and $d t$ as finite steps $\Delta \mathbf{u}$ and $\Delta t$, and multiply the equation by $\Delta t$ derivatives in the form of differences. The smaller $\Delta t$ is, the better approximation of the differential equation can be achieved. The forward Euler method is a literal implementation of this procedure. The implementation of the forward Euler method is discussed in the following.

### 2.5.2 Explicit Euler Method

The Euler method is a first order method for solving ODEs. The function $u(t)$ is evaluated at equidistant time points $t_{\mathrm{i}}$ where $t_{\mathrm{i}+1}=t_{\mathrm{i}}+h$ and $h$ is the time step:

$$
\begin{equation*}
\left.\mathbf{v}\right|_{\left(\mathrm{t}=\mathrm{t}_{\mathrm{i}}\right)}=\mathbf{v}^{0}+\sum_{\mathrm{k}=0}^{i-1} \int_{\mathrm{t}_{\mathrm{k}}}^{t_{\mathrm{k}}+h} \mathbf{M}^{-1} \mathbf{F}(\mathbf{u}, \mathbf{v}, t) d t \tag{2.127}
\end{equation*}
$$

The integration can be approximated by

$$
\begin{equation*}
\int_{\mathrm{t}_{\mathrm{k}}}^{t_{\mathrm{k}}+h} \mathbf{M}^{-1} \mathbf{F}(\mathbf{u}, \mathbf{v}, t) d t=\mathbf{M}^{-1} \mathbf{F}\left(\mathbf{u}_{\mathrm{k}}, \mathbf{v}_{\mathrm{k}}, t\right) h \tag{2.128}
\end{equation*}
$$

using this approximation, the Euler steps can be obtained as follows:I

$$
\begin{align*}
\mathbf{v}^{i+1} & =\mathbf{v}^{i}+\mathbf{M}^{-1} \mathbf{F}\left(\mathbf{u}^{i}, \mathbf{v}^{i}, t_{\mathrm{i}}\right) h  \tag{2.129}\\
\mathbf{u}^{i+1} & =\mathbf{u}^{i}+\mathbf{v}^{i} h \tag{2.130}
\end{align*}
$$

The magnitude of the errors that occur between Euler step $\mathbf{v}^{n}$ and $\mathbf{v}^{n+1}$ can be estimated by making a comparison with a Taylor expansion. By assuming $\mathbf{v}\left(t_{\mathrm{n}}\right)$ is exactly known, the Taylor expansion gives:

$$
\begin{align*}
\left.\mathbf{v}\right|_{\left(\mathrm{t}=\mathrm{t}_{\mathrm{i}}+\mathrm{h}\right)} & =\left.\mathbf{v}\right|_{\left(\mathrm{t}=\mathrm{t}_{\mathrm{i}}\right)}+\left.h \frac{d}{d t} \mathbf{v}\right|_{\left(\mathrm{t}=\mathrm{t}_{\mathrm{i}}\right)}+O\left(h^{2}\right)+O\left(h^{3}\right)+\ldots  \tag{2.131}\\
& =\mathbf{v}^{i+1}+O\left(h^{2}\right)+O\left(h^{3}\right)+\ldots \tag{2.132}
\end{align*}
$$

For small $h$ the error resulting from the Euler method is proportional to $h^{2}$.

### 2.5.3 Stiff Differential Equations

When using the forward Euler method as well as many other methods, $h$ must be chosen sufficiently small to obey an absolute stability restriction. The absolute stability restriction ensures that the distance between the exact and numerical solutions decreases. Otherwise, the algorithm becomes unstable. It is important to understand that the absolute stability restriction does not ensure accuracy [55].

Ideally, the choice of the time step $h$ should be dictated by the approximation accuracy requirement. But the additional absolute stability requirement can dictate a much smaller $h$ [55].

Loosely speaking, differential equations of an initial value problem are called stiff, if the absolute stability requirements of the equations dictates a much smalled $h$ than the approximation requirements demands [55].

Generally, mass-spring systems generate stiff differential equations. That applies for Adamss as well. In almost all the applications where Adamss was implemented the resulting differential equations turned out to be stiff for a fairly small time step. That means that even smaller time steps were needed to ensure the stability of the time integration algorithm. That also means much more integration steps were needed to obtain the results of a simulation, making the system very expensive computationally.

Actually, the integration algorithms became unstable so many times while trying to find the biggest possible time step the numerical time integration can take without becoming unstable. And so many times, the unstable system produced impossible deformations that could be elevated to the status of computational art. In an attempt to find beauty in failures. Figure 2.17 depicts a case of unphysical deformation. Because of the stiffness problem, the saying: "My mood is oscillating like a Mass-Spring System" made perfect sense.
*****

Stiffness of ODEs is a well known issue in numerical integration, and much effort has been done in the development of methods that try to extend the region of stability of the differential equations. To deal with the stiffness problem, two approaches were implemented in this work. The first is the Adams-Bashforth Moulton predictor-corrector method with adaptive time- stepping, and the second is the backward Euler method which is an implicit method. The computational cost of methods that solve the stiffness problem is usually high. But, overall the use of
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-43.jpg?height=861&width=1486&top_left_y=352&top_left_x=336)

Fig. 2.17 A Snapshots of a simulations right before (left) and right after (right) time integration of the stiff ODEs became unstable.
these methods may be much cheaper than taking the large number of steps required for explicit methods to work successfully [56].

### 2.5.4 Adams-Bashforth Moulton Predictor-Corrector Method

To be able to take larger time-steps, two multistep methods, namely AdamsBashforth which is an explicit method and Adams-Moulton which is an implicit methods were used as a Predictor-Corrector pair with adaptive time-stepping as described in [57].

The explicit Adams-Bashforth method of order $k+1$ is given for $\mathbf{u}$ with

$$
\begin{align*}
\mathbf{u}^{i+1} & =\mathbf{u}^{i}+h \sum_{\mathrm{j}=0}^{k} \beta_{\mathrm{j}} \mathbf{v}^{i-j}  \tag{2.133}\\
\beta_{\mathrm{j}} & =(-1)^{j} \sum_{\mathrm{i}=\mathrm{j}}^{k}\binom{i}{j} \gamma_{\mathrm{i}}  \tag{2.134}\\
\gamma_{\mathrm{i}} & =(-1)^{i} \int_{0}^{1}\binom{-s}{i} d s \tag{2.135}
\end{align*}
$$

and the different values for $\beta_{\mathrm{j}}$ can be also found in tables in the literature [58, 57]. Adams-Bashforth methods are explicit methods with very small regions of absolute stability. The implicit version of Adams methods are called the AdamsMoulton methods of order $\mathrm{k}+2$ :

$$
\begin{equation*}
\mathbf{u}^{i+1}=\mathbf{u}^{i}+h \sum_{\mathbf{j}=-1}^{k} \beta_{\mathbf{j}} \mathbf{v}^{i-j} \tag{2.137}
\end{equation*}
$$

To make use of both methods in a predictor-corrector pair, the explicit AdamsBashforth method is used to predict the value of the next step. By using the predicted value on the right side of the implicit Adams-Moulton method, a better estimation of the next step is made, i.e. the value is corrected, as presented in these formulas where first a prediction is made using the Adams-Bashforth method of the 3rd order:

$$
\begin{align*}
\tilde{\mathbf{v}}^{i+1}= & \mathbf{v}^{i}+h\left(\frac{23}{12} \mathbf{M}^{-1} \mathbf{F}\left(\mathbf{u}^{i}, \mathbf{v}^{i}, t_{\mathrm{i}}\right)-\frac{16}{12} \mathbf{M}^{-1} \mathbf{F}\left(\mathbf{u}^{i-1}, \mathbf{v}^{i-1}, t_{\mathrm{i}-1}\right)\right. \\
& \left.+\frac{5}{12} \mathbf{M}^{-1} \mathbf{F}\left(\mathbf{u}^{i-2}, \mathbf{v}^{i-2}, t_{\mathrm{i}-2}\right)\right)  \tag{2.138}\\
\tilde{\mathbf{u}}^{i+1}= & \mathbf{u}^{i}+h\left(\frac{23}{12} \mathbf{v}-\frac{16}{12} \mathbf{v}^{i-1}+\frac{5}{12} \mathbf{v}^{i-2}\right) \tag{2.139}
\end{align*}
$$

Then, the resulting values are used in the Adams-Moulton method of the same order in the correction step:

$$
\begin{align*}
& \mathbf{v}^{i+1}=\mathbf{v}^{i}+h\left(\frac{5}{12} \tilde{\mathbf{v}}^{i+1}+\frac{8}{12} \mathbf{M}^{-1} \mathbf{F}\left(\mathbf{u}^{i}, \mathbf{v}^{i}, t_{\mathrm{i}}\right)-\frac{1}{12} \mathbf{M}^{-1} \mathbf{F}\left(\mathbf{u}^{i-1}, \mathbf{v}^{i-1}, t_{\mathrm{i}-1}\right)\right)  \tag{2.140}\\
& \mathbf{u}^{i+1}=\mathbf{u}^{i}+h\left(\frac{5}{12} \tilde{\mathbf{u}}^{i+1}+\frac{8}{12} \mathbf{v}^{i}-\frac{1}{12} \mathbf{v}^{i-1}\right) \tag{2.141}
\end{align*}
$$

The predicted and the corrected coordinates are compared, and a decision about the accuracy at the specified time-step is taken. If all the differences were smaller than a predefined lower-error threshold, the time-step is doubled (reducing time-mesh). If one of the differences was greater than a predefined maximal-error threshold, the time-step is divided by two (refining time-mesh). Otherwise, the time-step remains unchanged. The method is depicted in the flowchart 2.18.

Adams-Bashforth and Adams-Moulton methods are multi-points methods, old values should be available to the time integration method. These values must be taken by equidistant time-steps. When the time-step is halved, extra values must be calculated between the old ones. To do so the old values are linearly interpolated. The
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-45.jpg?height=1189&width=1474&top_left_y=359&top_left_x=345)

Fig. 2.18 Flowchart of the Adams-Bashforth-Moulton time integration scheme, the time step adaptation mechanism occurs in the Adams Bashforth Moulton evalutation block
refine time-mesh concept is described in Figure 2.19.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-45.jpg?height=401&width=1494&top_left_y=1830&top_left_x=338)

Fig. 2.19 The time mesh refinement procedure is displayed, $i$ is the discrete time axis, the big circles represent the old values, they are at equidistance of $h$ the old timestep, the little circles represent the new values aligned with the new time-step $\frac{h}{2}$, new values at $Z^{-3}$ and $Z^{-1}$ are linearly interpolated with the old values at $Z, Z^{-1}$ and $Z^{-2}$.

The implemented method is of the 3rd order. That means that three previous points
should be available at any given time for the method to work properly. But at the beginning of the simulation no function evaluations are available. Therefore the first 2 steps are done using the forward Euler method with a small enough timestep.

The prediction-correction method offered a huge advantage over the explicit Euler method concerning stability. However, the timestep plunges often into very small values to ensure stability and accuracy of the solution which results in reduction of the efficiency. On the other hand if the thresholds controlling the adaptation of time steps were not set properly the system could take larger steps that lead eventually towards instability. This approach did not lead to satisfying results. The variations of volume of the elements were too high. The parameterization of the functions was an additional drawback of this method.

### 2.5.5 Backward Euler Method

The backward Euler method uses an approach different from the forward Euler method. Hereby, the solution for the next timestep depends only on the forces which will arise at timestep $t_{\mathrm{n}+1}$ :

$$
\begin{align*}
\mathbf{v}^{i+1} & =\mathbf{v}^{i}+\Delta \mathbf{v}  \tag{2.142}\\
\Delta \mathbf{v} & =\mathbf{M}^{-1} F\left(\mathbf{u}^{i+1}, \mathbf{v}^{i+1}, t_{\mathrm{i}+1}\right) h  \tag{2.143}\\
\mathbf{u}^{i+1} & =\mathbf{u}^{i}+\mathbf{v}^{i+1} \cdot h \tag{2.144}
\end{align*}
$$

The backward Euler method is an implicit method, since a set of implicit equations has to be solved.

Figure 2.20 shows the exact solution for a differential equation of a harmonic oscillation and the approximated solutions of the explicit and implicit Euler methods for different time step sizes. Depending on the time step size, the amplitude of the solution resulting from the explicit Euler method grows in time without bound, which leads to numerical instability. By contrast, the solution of the backward Euler method shows numerical damping, which is characteristical for implicit methods allowing the use of bigger time steps.

By applying a Taylor series expansion to $\mathbf{F}\left(\mathbf{u}^{i+1}, \mathbf{v}^{i+1}, t_{\mathbf{i}+1}\right)$ and making the first order approximation, we get:
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-47.jpg?height=1155&width=1514&top_left_y=345&top_left_x=331)

Fig. 2.20 Comparison of explicit and implicit method for different time step sizes. The exact solution is displayed in blue, the explicit method in red and the implicit method in green. In this example, for a relative big timestep $h=1$ the explicit method does not converge to the exact solution and became unstable while the implicit method does not converge to the exact solution either but the related curve remains under the curve of the exact solution.

$$
\begin{align*}
\mathbf{F}_{\mathrm{d}, \mu}\left(\mathbf{u}^{i+1}, \mathbf{v}^{i+1}, t_{\mathrm{i}+1}\right)= & \mathbf{F}\left(\mathbf{u}^{i}, \mathbf{v}^{i}, t_{\mathrm{i}}\right) \\
& +\frac{\partial}{\partial \mathbf{u}} \mathbf{F}\left(\mathbf{u}^{i}, \mathbf{v}^{i}, t_{\mathrm{i}}\right)\left(\mathbf{u}^{i+1}-\mathbf{u}^{i}\right) \\
& +\frac{\partial}{\partial \mathbf{v}} \mathbf{F}\left(\mathbf{u}^{i}, \mathbf{v}^{i}, t_{\mathrm{i}}\right)\left(\mathbf{v}^{i+1}-\mathbf{v}^{i}\right)  \tag{2.145}\\
& +\frac{\partial}{\partial t} \mathbf{F}\left(\mathbf{u}^{i}, \mathbf{v}^{i}, t_{\mathrm{i}}\right) h
\end{align*}
$$

Then by substituting in the Eq. (2.143), then using Eqs. (2.144) and (2.142), and finally arranging for $\Delta \mathbf{v}$ a linear system of the form

$$
\begin{equation*}
\mathbf{A} \Delta \mathbf{v}=\mathbf{b} \tag{2.146}
\end{equation*}
$$

is obtained, where $\mathbf{A}$ is the $3 n \times 3 n$ system matrix, and $\mathbf{b}$ is the $3 n$ right-hand side vector:

$$
\begin{align*}
\mathbf{A}= & 1-h \mathbf{M}^{-1} \frac{\partial}{\partial \mathbf{v}} \mathbf{F}\left(\mathbf{u}^{i}, \mathbf{v}^{i}, t_{\mathrm{i}}\right)-h^{2} \mathbf{M}^{-1} \frac{\partial}{\partial \mathbf{u}} \mathbf{F}\left(\mathbf{u}^{i}, \mathbf{v}^{i}, t_{\mathrm{i}}\right)  \tag{2.147}\\
\mathbf{b}= & h \mathbf{M}^{-1} \mathbf{F}_{\mathrm{a}}\left(t_{\mathrm{i}+1}\right)+h \mathbf{M}^{-1} \mathbf{F}\left(\mathbf{u}^{i}, \mathbf{v}^{i}, t_{\mathrm{i}}\right)  \tag{2.148}\\
& +h^{2} \mathbf{M}^{-1} \frac{\partial}{\partial \mathbf{u}} \mathbf{F}\left(\mathbf{u}^{i}, \mathbf{v}^{i}, t_{\mathrm{i}}\right) \mathbf{v}^{i}+h^{2} \mathbf{M}^{-1} \frac{\partial}{\partial t} \mathbf{F}\left(\mathbf{u}^{i}, \mathbf{v}^{i}, t_{\mathrm{i}}\right)
\end{align*}
$$

To assemble $\mathbf{A}$ and $\mathbf{b}$, the Jacobian matrices $\frac{\partial}{\partial \mathbf{u}} \mathbf{F}$ and $\frac{\partial}{\partial \mathbf{v}} \mathbf{F}$ have to be determined. By acknowledging that $h^{2}$ is a very small value, the term $h^{2} \mathbf{M}^{-1} \frac{\partial}{\partial t} \mathbf{F}\left(\mathbf{u}^{i}, \mathbf{v}^{i}, t_{\mathrm{i}}\right)$ can be neglected for simplification.

To obtain the Jacobian matrix of the deformation forces $\mathbf{F}_{\mathrm{d}}$, for each particle $p_{\mathrm{l}}$, the deformation forces at current timestep $\mathbf{F}_{\mathrm{d}}(\mathbf{u})$ are calculated. Then a small but finite value $\delta u$ is added to each element $\mathbf{u}_{\mathrm{m}}$ of the vector $\mathbf{u}$, one element at a time, that results in the varied configurations $\hat{\mathbf{u}}(m)$. Eventually the varied passive forces $\hat{\mathbf{F}}_{\mathrm{d}}(\hat{\mathbf{u}}(m))$ of all particles $p_{1}$ are calculated. The Jacobian matrix $\mathbf{J}_{\mathrm{u}}$ is then built by calculating the difference quotients:

$$
\begin{equation*}
J_{\mathrm{u}, \mathrm{~m}}=\frac{\mathbf{F}_{\mathrm{d}}-\hat{\mathbf{F}}_{\mathrm{d}}(\hat{\mathbf{u}}(m))}{\delta u} \tag{2.149}
\end{equation*}
$$

In a similar way, the Jacobian matrix $\mathbf{J}_{\mathrm{v}}$ of the damping forces $\mathbf{F}_{\mu}$ is calculated by adding small but finite values $\delta v$ to each of the element of the velocity vector v and repeating the steps detailed above. Both Jacobians are then replaced in Eqs. (2.147) and (2.148).

Since any particle of the system is only connected to a maximal number of 26 neighboring particles (voxel's vertices in 3D space) the matrix $\mathbf{A}$ is sparse and it can be shown that every row of the matrix contains a constant count of non zero elements. That allows the use of efficient methods of computational complexity $O(n)$ to solve the linear system (2.146). An iterative solver (GMRES) that takes advantage of the matrix A sparsity provided with the PETSc package [59] is used.

To take advantage of the possibility of taking bigger time steps which the implicit integration provides without risking the algorithm to becomes unstable, an adaptive time-stepping mechanics is used.

It was observed that when the system shows high dynamics, caused by large forces for example, the number of iterations $N_{\mathrm{s}}$ the iterative solver needs to converge when solving the linear system in Eq. (2.146) increases. If the system shows monotonic behavior, the number of iterations decreases. Furthermore, the number of iterations increases massively just a few steps before the system becomes unstable.

In general the number of iterations appeared to be a good parameter to control the timestep size used in the implicit integration algorithm as shown in Figure 2.21.

By defining a lower and an upper limit for the number of iterations, the implicit solver can adapt the timestep size. If the number of iterations reaches the lower limit, the timestep size is doubled. If the number of iterations reaches the upper limit the timestep size is halved.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-49.jpg?height=880&width=1446&top_left_y=848&top_left_x=351)

Fig. 2.21 A good correlation between the normalized active forces sum (blue) and the iteration counts of the GMRES iterative solver of the PETSc package (red).

This technique is not completely legitimate, because the mathematics behind it was not carefully analyzed. However, so far experiments are showing this method is able to manage the timestep size with great success.

### 2.5.6 Critical Timestep Size

To obtain a realistic simulation, the maximum timestep must be smaller than the time, a mechanical wave needs to pass the modeled object. Otherwise the propagation of a mechanical wave cannot be simulated properly. The critical timestep size is given by

$$
\begin{equation*}
h_{\mathrm{c}}=\frac{L}{c} \tag{2.150}
\end{equation*}
$$

where $L$ is the length of the smallest side of the volume element and c is the propagation velocity of a mechanical wave (the speed of sound). The propagation velocity depends on the damping behavior of the object where the wave is propagating. An appropriate timestep size can be obtained, by comparing the results of the simulation for different timestep sizes. A timestep is sufficiently small, if the results of a simulation using that timestep and the results of a simulation using a timestep which is orders of magnitude smaller do not differ substantially. This method was used to justify the timestep used for a given application.

### 2.6 Boundary Conditions

To constrain the movement of the modeled object, particles can be marked with different fixation and control tags. A file containing the indices of the particles and the associated tags is processed during the initialization of the system to mark the particles.

Fixation tags can be one of three possible tags and their combination. These tags are:

- Fixed in $x$-axis direction
- Fixed in $y$-axis direction
- Fixed in $z$-axis direction

Here, the axes correspond to the global coordinate system.
For example, a particle $p_{\mathrm{i}}$ marked with the $x$-axis and the $y$-axis fixation tags, can only move in the $z$-axis direction. In the case $p_{\mathrm{i}}$ was marked with all fixation tags, it will have the same position during the entire simulation.

The code implementation of the fixation boundary condition involves iterating over particles and setting the offset of the particle according to the fixation tags to zero. However in the case when the implicit time integration is used, the $\mathrm{x}, \mathrm{y}, \mathrm{z}$ mass components of the mass tensor M are set to a very large value according to the fixation tags, preventing it from moving.

Control tags can be one of three possible tags and their combination. These tags are:

- Controlled in $x$-axis direction
- Controlled in $y$-axis direction
- Controlled in $z$-axis direction

For example, a particle $p_{\mathrm{i}}$ marked with the $x$-axis control tag, will be moved every simulation step with a predefined step in the direction of the $x$-axis.

Using both fixation and control tags gives the possibility of conducting strainstress simulations. For example a uni-axial strain-stress experiment of a patch of elastic material can be conducted by fixing the model from both ends in all directions, and controlling one end in the direction of one of the axis.

### 2.7 System's Output

To visualize and evaluate the deformation, the system exports periodically sets of data related to the simulation. The export-period can be set at the beginning of the simulations depending on the simulation's circumstances.

The system can export the coordinates, offsets, velocity and acceleration vectors of all particles. Other general data, not related to individual particles, can also be exported. These comprises the simulation time step, the particles total kinetic energy, the system's barycenter coordinates, the system's total volume.

Beside plain text data files, the system supports several file formats like the well known Visualization Toolkit (VTK) file format [60].

### 2.8 System Verification

In this section, several simulations designed to test the functionality of the system and verify its implementation are demonstrated.

### 2.8.1 Reproducing Mechanical Properties

To verify the ability of the system to reproduce the mechanical properties of a modeled material represented by its constitutive law, a series of uniaxial stretch and simple shear simulations were conducted and the resulting stress strain relations were compared with the curves calculated theoretically. In cases where the theoretical calculation of the stress strain relation is not trivial, like in simple shear experiments, the applied work to the system is compared with the potential deformation energy. If the stresses were calculated correctly, the potential deformation energy must be equal to the applied work. The deformation energy can be calculated easily using the deformation tensor and the energy density function. The linear approximation of work $W_{\mathrm{i}+1}$ applied between stretch steps $s_{\mathrm{i}}$ and $s_{\mathrm{i}+1}$ is given by

$$
\begin{equation*}
W_{\mathrm{i}+1}=\sum_{\mathrm{n}=1}^{N}\left(\mathbf{x}_{\mathrm{n}}\left(s_{\mathrm{i}+1}\right)-\mathbf{x}_{\mathrm{n}}\left(s_{\mathrm{i}}\right)\right) \cdot \frac{\mathbf{f}_{\mathrm{n}}\left(s_{\mathrm{i}+1}\right)+\mathbf{f}_{\mathrm{n}}\left(s_{\mathrm{i}}\right)}{2} \tag{2.151}
\end{equation*}
$$

where $\mathbf{x}_{\mathrm{n}}$ are the coordinates of the controlled point $p_{\mathrm{n}}, \mathbf{f}_{\mathrm{n}}$ is the force acting on $p_{\mathrm{n}}$, and $N$ is the number of controlled particles. The total work is given by

$$
\begin{equation*}
W_{\text {total }}=\sum_{\mathrm{i}=1}^{i_{\text {total }}} W_{\mathrm{i}} \tag{2.152}
\end{equation*}
$$

For the simulations, the constitutive law of myocardial tissue proposed by Hunter et al. [61] was used:

$$
\begin{align*}
W & =k_{11} \frac{E_{11}^{2}}{\left|a_{11}-E_{11}\right|^{b_{11}}}+k_{22} \frac{E_{22}^{2}}{\left|a_{22}-E_{22}\right|^{b_{22}}}+k_{33} \frac{E_{33}^{2}}{\left|a_{33}-E_{33}\right|^{b_{33}}}  \tag{2.153}\\
& +k_{12} \frac{E_{12}^{2}}{\left|a_{12}-E_{12}\right|^{b_{12}}}+k_{13} \frac{E_{13}^{2}}{\left|a_{13}-E_{13}\right|^{b_{13}}}+k_{23} \frac{E_{23}^{2}}{\left|a_{23}-E_{23}\right|^{b_{23}}}
\end{align*}
$$

where $E_{\mathrm{ij}}$ are the elements of the Green strain tensor $\mathbf{E}$. They are related to the deformation tensor $\mathbf{F}$ according to Eq. 1.6. $k_{\mathrm{ij}}$ and $b_{\mathrm{ij}}$ are parameters obtained by experimental data. The parameters used in this simulation are given in Table 2.1. For more about the implementation of this constitutive law see Section 4.7.5.

Table 2.1 Parameters for the energy density function proposed by Hunter et al. adapted to canine myocardium (Parameters from [61, 23])

| $k_{11}$ | $k_{22}$ | $k_{33}$ | $k_{12}$ | $k_{13}$ | $k_{23}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 2.842 | 0.063 | 0.31 | 1.0 | 1.0 | 1.0 |
| $a_{11}$ | $a_{22}$ | $a_{33}$ | $a_{12}$ | $a_{13}$ | $a_{23}$ |
| 0.318 | 0.429 | 1.037 | 0.731 | 0.731 | 0.886 |
| $\beta_{11}$ | $\beta_{22}$ | $\beta_{33}$ | $\beta_{12}$ | $\beta_{13}$ | $\beta_{23}$ |
| 0.624 | 2.48 | 0.398 | 2.0 | 2.0 | 2.0 |

It is easy to recognize from Eq. (2.153) that this material model has three directions of passive mechanical anisotropy, along the material local coordinate system that represents the fiber, sheet and sheet-normal directions of myocardial tissue, which we will call the fiber coordinate system.

First deformation forces calculation using axial and torsion springs is put to test. Polynomial axial springs parameters were obtained by fitting a third order function to the strain-stress curves of Eq. (2.153) for the uniaxial stretch experiments (see Section 2.4.2.1). For torsion springs, the values were chosen to give a good approximation of the shear curves (see Section 2.4.2.1). The parameters are listed in Table 2.2, the values were fitted to maximum strain of $20 \%$.

Table 2.2 Parameters (kPa) of third-order polynomial axial and torsion springs. The third-order polynomial functions were fitted to the constitutive law of Hunter et al. [61]

| $k_{11}$ | $k_{13}$ | $k_{21}$ | $k_{23}$ | $k_{31}$ | $k_{33}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 6.358 | 768 | 0 | 288.525 | 0.305 | 5.699 |
| $k_{12,1}$ | $k_{12,3}$ | $k_{13,1}$ | $k_{13,3}$ | $k_{23,1}$ | $k_{23,3}$ |
| 3.355 | 97.374 | 97.374 | 3.355 | 45.896 | 2.261 |

Uniaxial stretch experiments of a patch of $6 \times 6 \times 10$ hexahedra were conducted. The model's resolution was 5 mm and the patch was divided to 2160 tetrahedra using a tetrahedral mesh topology. Forces were calculated using the axial and torsion springs of the third order and the parameters in Table 2.2. The resulting stress strain curves are plotted in Figure 2.22 alongside the theoretical calculation of stress strain relation of Eq. (2.153). It is important to remember that the parameters are related to the stress values in volume elements and not to forces, therefore these curves show that the behavior of the models is independent of the chosen resolution. However, a better behavior is generally expected at higher resolutions specially in case of modeling objects with smooth or sharp edges. Then the method of virtual hexahedron was put here test. Uniaxial stretch simulations of a patch of myocardial tissue along the fiber, sheet and sheet-normal axes were conducted. Hereby, stretching is performed with small steps $s_{\mathrm{i}}$, and the resulting stress at each stretch step is determined after the system reaches an equilibrium state. In all simulations, deformation tensors are calculated using the shape functions method (see Section 2.4.2.2).

First stretch simulation of a hexahedron, using tetrahedral mesh topology where each hexahedron was divided to six tetrahedra, were conducted (Fig. 2.23). Using the same mesh topology, simulations of a cuboid composed of $4 \times 4 \times 8$ hexahedra (768 tetrahedra) were performed (Fig. 2.25). In all these simulations, the fiber coordinate system was set parallel to the regular grid. To show that the system can control anisotropy, the fiber coordinates system was rotated by $\phi=21.25^{\circ}$ and $\Theta=49.6^{\circ}$ in spherical coordinates, and stretch simulations of the $4 \times 4 \times 8$ cuboid model in the global $z$-axis direction were conducted (Fig. 2.25).

In all simulations, the simulation results are in very good agreement with the theoretical calculations. Other simulations not presented here using the hexahedral topology gave results equal to the results for tetrahedral mesh topologies. Simple shear experiments were conducted, to validate the ability of the system to reproduce shear forces. In order to do that, the top plane of the modeled patch was displaced in a direction parallel to the plane. For each experiment, three different shear procedures were simulated: The plane orthogonal to the fiber axis was displaced in direction of the sheet-normal axis, the plane orthogonal to the sheet axis was displaced in direction of the fiber axis and the plane orthogonal to the sheet-normal axis was displaced in direction of the sheet axis. Models used for the
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-54.jpg?height=1086&width=1489&top_left_y=351&top_left_x=238)

Fig. 2.22 The stress-strain relation of an $6 \times 6 \times 10$ hexahedra object of 5 mm resolution ( 2160 tetrahedra), for strains in fiber (red), sheet (green) and sheet-normal (blue) directions. Forces were calculated using the axial and torsion springs of the third order. The results of the simulations (circles) using third degree spring functions were used are displayed with theoretical calculation (solid lines). The object was stretched to $20 \%$ of its original length.
stretch simulations were used for these experiments. However no fiber coordinates rotation was applied.

To validate the simulation results, in addition the applied work to the system was compared with the deformation energy.

Figures 2.26 and 2.27 show the simulations' outcome. Here the difference between the multiple hexahedra cuboid and the single hexahedron deformation energies is due to the contribution of the axial strain to the deformation energy in the case of the cuboid, resulting in bigger differences between the three curves in comparison with the single hexahedron case. Nonetheless, the resulting applied work in both simulations fits very well to the deformation energy.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-55.jpg?height=918&width=1500&top_left_y=346&top_left_x=329)

Fig. 2.23 Stress-strain curves for uniaxial stretch of a voxel (6 tetrahedra) along all axes. Simulation output (points), theoretical calculation (solid line) for fiber (red), sheet (green) and sheet-normal (blue) directions.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-55.jpg?height=920&width=1503&top_left_y=1382&top_left_x=331)

Fig. 2.24 Stress-strain curves for uniaxial stretch of a $4 \times 4 \times 8$ voxel object ( 768 tetrahedra) along all axes. Simulation output (points), theoretical calculation (solid line) for fiber (red), sheet (green) and sheet-normal (blue) directions.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-56.jpg?height=895&width=1486&top_left_y=352&top_left_x=239)

Fig. 2.25 Uniaxial stretch along z-axis, fiber coordinate system is rotated relative to the hexahedral mesh with $\theta=49.6^{\circ}$ and $\phi=21.75^{\circ}$. Simulation output or applied work (points), theoretical calculation or deformation energy (solid line), $4 \times 4 \times 8$ voxel, 768 tetrahedra.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-56.jpg?height=915&width=1489&top_left_y=1456&top_left_x=238)

Fig. 2.26 Energy-shear curves for simple shear. In the legend the first axis expresses the normal vector of the displaced plane and the second axis expresses the direction of displacement. Simulation output or applied work (points), theoretical calculation or deformation energy (solid line) in fiber (red), sheet (green) and sheet-normal (blue) directions for 1 voxel, 6 tetrahedra.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-57.jpg?height=901&width=1486&top_left_y=349&top_left_x=336)

Fig. 2.27 Energy-shear curves for simple shear. In the legend the first axis expresses the normal vector of the displaced plane and the second axis expresses the direction of displacement. Simulation output or applied work (points), theoretical calculation or deformation energy (solid line) in fiber (red), sheet (green) and sheet-normal (blue) directions for $4 \times 4 \times 8$ voxel, 768 tetrahedra.

Many other validation simulations using different mesh topologies and both methods to calculate the deformation tensor, and different geometries were conducted. In general as long as no hour-glassing occurs (see Section 2.9.1 and Fig. 2.34), the hexahedral mesh topologies can reproduce the mechanical properties of the material almost as good as the tetrahedral mesh topologies. A small difference can be found between the results when hexahedral, or tetrahedral volume elements are used. This discrepancy results from the linear interpolation of the deformation tensors. These simulations show as well that the system is able to calculate the passive forces of myocardial tissue correctly, and to correctly reflect anisotropy using the method of virtual hexahedron, that facilitates to a great extent modeling different deformable materials by using their constitutive laws.

### 2.8.2 Volume Preservation

To demonstrate the ability of the presented mass-spring system to maintain its volume under deformation, the deformation of a cuboid of $6 \times 6 \times 12$ hexahedral elements under gravity loading was simulated using implicit time integration, and the continuum mechanics method for volume preservation (see Section 2.4.2.3). In the presented simulations, different values for the volume preservation coefficient $p$ were used, and for each simulation the relative change of volume $\frac{\Delta V}{V^{0}}$ was measured.

Figure 2.28 shows $\frac{\Delta V}{V^{0}}$ as a function of time for different values of $p$. For this application a value of $p$ bigger than $10^{3} \mathrm{kPa}$ will keep the relative change of volume well below $1 \%$.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-58.jpg?height=875&width=1446&top_left_y=602&top_left_x=254)

Fig. 2.28 The relative volume change curve during gravity loading simulations with coefficients: $p=$ $10^{2} \mathrm{kPa}$ (red), $p=10^{3} \mathrm{kPa}$ (green), $p=10^{4} \mathrm{kPa}$ (blue) and $p=10^{5} \mathrm{kPa}$ (magenta).

### 2.8.3 Computational Complexity

The main advantage of using mass-spring system in physical based modeling of deformable objects is that each time step iteration has a computational complexity of $O(n)$ where $n$ is the number system's particles. The reason is that determining the deformation in each time step requires solving the system of ODEs given in Eqs. (2.125) and (2.126) which can be done in a single sweep over all system's particles using explicit time integration methods.

Implicit time integration requires solving a linear system of the form given in Eq. (2.146) where the system matrix $\mathbf{A}$ is $3 n \times 3 n$ dimensions. Solving such systems is a task of $O\left(n^{2}\right)$ complexity. However, as mentioned in Section 2.5.5, the system matrix is sparse and therefore efficient methods of $O(n)$ complexity that take advantages of the sparsity of the matrix can be used to solve the system.

To demonstrate the computational complexity of the system in the case of using implicit time integration, cuboid-shaped objects of dimensions $6 \times 6 \times 12 \mathrm{~cm}$, sharing the same physical properties but having different mesh resolutions and thus different number of elements, were simulated using the same simulation setup, and the same computational environment. In each simulation the computation time was measured every 1000 timestep iterations. Hereby, a fixed time step was used.

Figure 2.29 shows the average computation time of 1000 iterations for the corresponding volume elements count. For simulations repeated more than once, error bars showing the deviation from the average are shown in the figure. The results show clearly the linear relation between computation time and the number of volume elements simulations with the implemented implicit time integration method.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-59.jpg?height=878&width=1468&top_left_y=1069&top_left_x=354)

Fig. 2.29 The average computation time needed for 1000 timestep iterations with a constant timestep. The measured average and the standard deviation (green) and the linear regression (red).

### 2.9 Discussion

The system is a combination of several modules. In each module, there are many design aspects that affect the implementation and the simulation outcome of the system. Also for each modeling task or application, there are different modules combination that could be used. Choosing between these combinations affects the
system in many areas like the stability of the simulation, accuracy of simulation results, computation duration ... etc. In the following, we will go over the modules while discussing the most important of these issues.

### 2.9.1 About mesh topologies

Each of the mesh topologies implemented in this system has advantages and drawbacks.

For instance, the greatest advantage of using a hexahedral mesh topology is that most medical imaging devices, like CT and MRI, produce images stored in a three dimensional lattice. That makes models generation of objects depicted in medical images a straight-forward task. Nonetheless, in order to model smooth surfaces using hexahedral mesh topology, a high resolution mesh must be used. That means more elements must be modeled leading to the increase of computational costs.

As mentioned in 2.1.1, the deformation of the model is represented by the offset of the vertices from their initial positions. Since the time integration module updates the particles' coordinates regardless of volume elements they belong to, it is possible that a particle of a hexahedron will move across one of that hexahedron's faces, turning it to a non-simple polyhedron as shown in Figure 2.30. Such elements spoil the calculation of inner forces that depends on the volume elements integrity (see Section 2.4.2) therefore they are called corrupt. Corrupt elements
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-60.jpg?height=749&width=1377&top_left_y=1653&top_left_x=285)

Fig. 2.30 A deformed hexahedrom (a) and a non-simple polyhedron where vertex $p_{6}$ of the hexahedron moved across the opposite face of the hexahedron defined by $p_{1}, p_{5}, p_{8}$ and $p_{4}$ (b).
can be either the result of numerical instability or the result of using a timestep for time integration bigger than the critical timestep $h_{\mathrm{c}}$ (see Section 2.5.6). The modeling framework must implement methods to detect these elements and eventually to deal with them. Implementing a method to determine whether during deformation a hexahedron became corrupt is not trivial.

Faces of eight nodes hexahedron (linear hexahedron) are under-determined, because each of the faces is defined by four points. Since the vertices coordinates determine the deformation, there is no way to tell whether a linear hexahedron is in a state shown in Figure 2.31(a) or in state shown in Figure 2.31(b) or in some other state.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-61.jpg?height=806&width=1286&top_left_y=993&top_left_x=419)

Fig. 2.31 Two deformed hexahedral elements sharing the exact same vertices position, the first with concave surfaces (a) and the other with convex surfaces (b).

Determining the surfaces of volume elements'faces, the normals on the faces, and the volumes of the elements are routine tasks essential for the calculation of inner forces (see Section 2.4.2). Every time the surface of a deformed linear hexahedron's face, the normal to that face, and the volume of the hexahedron are computed, a systematic error is made.

Interpolating the coordinates of the intersection points is also prone to this systematic error.

To mitigate the errors in calculating the surface of faces defined by four points, the surface is divided to two triangles in two different ways as shown in Figure
2.32, and the surfaces of the resulting triangles are averaged. Similarly the normal to the face is determined by calculating the average of normals of the four mentioned triangles.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-62.jpg?height=730&width=1351&top_left_y=583&top_left_x=301)

Fig. 2.32 Two different ways to divide a face of a hexahedron in two triangles.

The volume of each hexahedron is computed using a method based on dividing the hexahedron to five or six tetrahedra as in Figures 2.7 and 2.8, and then summing up the computed volumes of the resulting tetrahedra as in in the work of Grandy et al. [50].

Furthermore, the deformation state of a hexahedron is not totally defined by the positions of its intersection points as displayed in figure 2.33. Therefore the forces resulting due to deformation in both cases are equivalent, which does not reflect the physical case.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-62.jpg?height=346&width=1261&top_left_y=2060&top_left_x=346)

Fig. 2.33 One face of three different deformed hexahedra with equal intersection points configuration showing that hexahedra are not uniquely defined by the intersection points.

On top of all that, hexahedra suffer from hourglass modes which are non-physical zero deformation energy modes that can result in non-physical deformations [1]. Figure 2.34 shows the different hourglass modes of linear hexahedron models.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-63.jpg?height=644&width=855&top_left_y=609&top_left_x=646)

Fig. 2.34 Hourglass modes of the linear hexahedron from [62].

For all these reasons several tetrahedral mesh topologies were examined and implemented. Most of the simulations conducted in this work are based on tetrahedral mesh topologies.

Using tetrahedra instead of hexahedra is very beneficial from the implementation and the simulation outcome points of view.

The triangle faces of tetrahedra are always uniquely defined, hence surfaces and normals on surfaces can be computed with no systematic error using Eqs. (2.14) and (2.11). For the same reason, coordinates of intersection points can be calculated uniquely using the vertices of the faces to which the points belong. Reversely, the deformation of a tetrahedron can be uniquely retrieved by the intersection points and the interpolation coefficients related to these intersection points.

A deformed tetrahedron is uniquely defined by its vertices, therefore the volume $V_{\mathrm{k}}$ of a tetrahedron can always be calculated with

$$
\begin{align*}
& V_{\mathrm{h}}=\left(\mathbf{x}_{1}-\mathbf{x}_{4}\right) \cdot\left(\left(\mathbf{x}_{2}-\mathbf{x}_{4}\right) \times\left(\mathbf{x}_{3}-\mathbf{x}_{4}\right)\right)  \tag{2.154}\\
& V_{\mathrm{k}}=\frac{1}{6}\left|V_{\mathrm{h}}\right| \tag{2.155}
\end{align*}
$$

where $V_{\mathrm{h}}$ is the signed volume of the parallelepiped defined by the four tetrahedron vertices $p_{\mathrm{i}}(i=1, \ldots, 4)$.
$V_{\mathrm{h}}$ can be used to detect corrupt tetrahedra. A corrupt tetrahedron is detected if the value of $V_{\mathrm{h}}$ changes sign during deformation. In the case all vertices of a tetrahedron become coplanar, many equations to calculate inner forces become invalid. Therefore flat tetrahedra are equally unpleasant as a corrupt tetrahedron and must also be detected.
![](https://cdn.mathpix.com/cropped/2025_06_18_500e32f01f63b688f2a5g-64.jpg?height=441&width=1437&top_left_y=833&top_left_x=264)

Fig. 2.35 A healthy tetrahedron (a), a tetrahedron with coplanar vertices (b) and a corrupt tetrahedron (c).

Tetrahedra do not have hourglass modes and therefore do not suffer from that phenomenon. But they do suffer from another phenomenon called locking that occurs when a hard constraint on volume preservation is imposed [49]. Locking manifests itself as a catastrophic artificial stiffening of the system. One practical solution to reduce the effect of locking is not to use fully but nearly incompressible material laws. The more the volume is allowed to change the less the effect of locking is observed.

The tetrahedral mesh topologies based on regular grids inherit some of the drawbacks of hexahedral mesh topologies. For instance modeling smooth surfaces needs high resolution meshes. Furthermore, the number of tetrahedra needed to model an object in comparison with a hexahedral mesh topology with the same resolution, is five or six times greater than the number of hexahedra, increasing the computation cost accordingly.

Tetrahedral mesh topology based on unstructured grids, have many advantages over the other implemented topologies. By using a good mesh generator and the right set of mesh optimization and filtering, it is possible to obtain meshes that have smooth surfaces starting from 3D lattice image datasets. That eliminates the need of high resolution models in order to model smooth surfaces needed in topologies based on regular grids.

But as it is often the case, nothing is for free. Here other problems arise, like volumetric locking and inhomogeneous distribution of volume. These problems must be addressed in order to mitigate their undesirable effects. To reduce the effect of inhomogeneous distribution of masses, a maximum limit is imposed on tetrahedron volume during the mesh generation phase, at the same time increasing the number of tetrahedra needed for an object and consequently the related memory and computational cost.

Image information relevant to modeling, for example the anisotropies distribution, are usually available in the form of a lattice dataset related to the original lattice image dataset. Assigning this information to the volume elements when using a hexahedral mesh or a tetrahedral mesh topology based on regular grid is straight-forward, but in the case of unstructured grids, this information must be set for generated tetrahedra that could extend over one or more elements with different values of the information lattice. One possible method to set these values for the generated tetrahedra is using interpolation techniques. In this work linear interpolation was used. However better interpolation techniques could be implemented.

Tetrahedral mesh generating methods which do not impose a limit on resulting tetrahedra's minimal volume, may generate some very skinny tetrahedra. Such tetrahedra are called slivers. Slivers are more prone to becoming corrupt or flat because of their morphology. A sliver removal algorithm or a better mesh generator can be used to avoid problems associated with these tetrahedra.

### 2.9.2 About Forces Calculation

Several issues concerning the calculation of deformation, active and volume preservation forces are mentioned here. Starting with deformation forces, sometimes material's constitutive laws are given only for the stretch direction. These constitutive laws do not provide information about stress in case of compression. If internal deformation forces are set to zero when modeling the compression of such materials, and no other forces can work against the compression like volume preservation forces, the elements can collapse and become corrupt making the results of the simulation invalid. A method to maintain a minimal volume for volume elements that can sustain compression must be implemented.

Some material's constitutive laws contain a pole in their formulas. When strain approaches a pole, stress rises immensely dividing the strain axis to two ranges. The range of valid strains and the range of invalid strains. Because the timestep used for time integration is not infinitesimally small, a volume element might deform from a state where the element's strain is within a valid strain range to a state where the element's strain is invalid. In that case the constitutive law formula might deliver a stress, and the simulation continues, but since the strain is invalid
that stress has no physical meaning, and the simulations results become invalid. Therefore constitutive laws that contain a pole are substituted with a high-order polynomial approximation which is inherently pole-free.

Volume preservation forces based on continuum mechanics solution according to Eq. (2.86) uses the parameter $p$ related to the bulk modulus. The parameter $p$ has to be chosen in a way such that the volume preservation forces resulting from a very small change in volume, have the same magnitude as all the forces participating in the deformation. When the volume deviates from its original value, hydrostatic work is added to the system and eventually, giving rise to penalty forces that trys to keep the change in volume around zero.

Locking associated with tetrahedral mesh topologies can be avoided by modeling the absolutely incompressible material with a nearly incompressible model. This can be done by choosing a value for $p$ that is not too high allowing the volume to vary, but so small that the volume variation won't exceed an accuracy limit. If a suitable value of $p$ cannot be found for a certain application, tetrahedral mesh topologies should be avoided, or different volume preservation force models must be implemented.

Material specific friction detailed in Section 2.4.2.5 was implemented only for axial friction or damping and no shear friction was implemented. If needed, this could be done.

The model used for active forces sets the forces or tension to the main anisotropy direction. To simulate internal processes producing forces with components in all anisotropy direction, a different active forces model must be developed and implemented.

### 2.9.3 About Time Integration

When using the forward Euler method or the prediction-correction method, the volume of the system $V$ oscillates around the original volume of the system $V^{0}$.

An explicit solver considers only the present forces to calculate the solution for the next timestep. Upcoming volume preservation forces caused by a change in volume, are not regarded. Since small changes in volume produce huge volume preservation forces, tiny timesteps must be taken, to avoid instability. However, even for small timesteps the volume preservation forces can cause a strong undesirable oscillation of the volume.

To get around these problems the upcoming volume preservation forces of the next timestep have to be taken into account, for solving the equations of motions.

Therefore the implicit backward Euler time integration method is favorable. Hereby, the solution of the next timestep depends on the forces which will arise at that step. This method ensures volume preservation and reduces the mentioned volume oscillation even for larger timesteps.

Implementing the implicit time integration for solving the equations of dynamics not only increased the stability and allowed for larger timesteps, but also removed the unwanted oscillation of the volume mentioned above. Implicit time integration introduces artificial damping that contaminates the simulation results. The larger the timestep size is the more artificial damping is added to the system. Therefore the timestep should be chosen carefully in order to obtain valid simulation results.

In the implementation of the implicit integration method two approximations are made. The first is making the first order approximation of the Taylor series expansion of $\mathbf{F}\left(\mathbf{u}^{i+1}, \mathbf{v}^{i+1}, t_{\mathrm{i}+1}\right)$ in Eq. (2.145), and the second is neglecting the term $h^{2} \mathbf{M}^{-1} \frac{\partial \mathbf{F}}{\partial t}\left(\mathbf{u}^{i}, \mathbf{v}^{i}, t_{\mathrm{i}}\right)$ in the same equation. Because of these approximations the stability of the implemented implicit time integration method is not unconditional. That means large changes in forces can cause numerical instability even when the implemented implicit time integration method is used for the integration.

Friction in general affects the speed of mechanical waves propagation. According to Eq. (2.150) the critical timestep $h_{\mathrm{c}}$ depends on the mechanical wave speed. It must be kept in mind that changing friction conditions might require a change in timestep used for time integration.


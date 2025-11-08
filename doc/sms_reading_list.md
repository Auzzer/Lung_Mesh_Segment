# 1. basic spring mass system 
1.1 a good intro about sms from taichi: (https://github.com/taichiCourse01/taichiCourse01/blob/main/material/09_implicit_integration.pdf) 
(a tutor video in mandarin: https://www.bilibili.com/video/BV1nr4y1Q73e/?vd_source=cbe82d5c4c737ec7366b7cf57dd2714b)

Note that the officially provided sample code may be excuted if taichi's version is >= 1.5.0. A new one is attached. 

1.2 the classical method used for real-time clothes simulation: 
[Large Steps in Cloth Simulation](https://www.cs.cmu.edu/~baraff/papers/sig98.pdf)

# 2. Anisotropy in SMS
2.1 [Controlling Anisotropy in Mass-Spring Systems](https://inria.hal.science/inria-00537510/file/BC00.pdf)
2.2  a more detaied one: https://www.ibt.kit.edu/download/PRJ_2011-02-10_O_Jarrousse.pdf


# 3. solver
3.1 taichi's linear system solver is also included in [1.1](https://www.bilibili.com/video/BV1nr4y1Q73e?vd_source=cbe82d5c4c737ec7366b7cf57dd2714b&spm_id_from=333.788.videopod.episodes&p=3)

3.2 equilibrium equation: https://manuals.dianafea.com/d108/en/1219784-1220336-incremental-iterative-solution-procedures-for-nonlinear-systems.html

# Some other useful info
Use SVD-based affine to represent the deformation is in https://math.ucdavis.edu/~jteran/papers/ITF04.pdf
You can also find a good tutor containing this section in https://www.bilibili.com/video/BV12Q4y1S73g?vd_source=cbe82d5c4c737ec7366b7cf57dd2714b&spm_id_from=333.788.videopod.episodes&p=8. The slides "GAMES103-08-FEM2.pdf" is attached

Professor Huaming Wang also has very good introduction tutors of simulation in https://games-cn.org/games103/.
The videos of [spring mass system](https://www.bilibili.com/video/BV12Q4y1S73g?vd_source=cbe82d5c4c737ec7366b7cf57dd2714b&spm_id_from=333.788.videopod.episodes&p=5) and fem methods "[GAMES103-07-FEM.pdf](https://www.bilibili.com/video/BV12Q4y1S73g?vd_source=cbe82d5c4c737ec7366b7cf57dd2714b&spm_id_from=333.788.videopod.episodes&p=7), [GAMES103-08-FEM2](https://www.bilibili.com/video/BV12Q4y1S73g?vd_source=cbe82d5c4c737ec7366b7cf57dd2714b&spm_id_from=333.788.videopod.episodes&p=8).pdf" 

An env file based on Matt's env file is attached as well. 
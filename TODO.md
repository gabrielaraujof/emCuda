Bug Fixing
==========

- CPU time accuracy.

New features
============

- Dynamic database size and path.
- Automatic kernel setting (by means of device information)

Kernels
=======
- **_E-Step_**
    - _Marginalization Kernel_: Calculates the marginal probability of each gaussian component.
- **_M-Step_**
    - _Weight Kernel_: Re-estimates the weight of each gaussian component.
    - _Mean Kernel_: Re-estimates the mean of each gaussian component.
    - _Covariance Matrix Kernel_: Re-estimates the covariance matrix of each gaussian component.

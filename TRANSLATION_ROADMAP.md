# HMcode-python + pyhalomodel → Julia translation roadmap

This document records the dependency mapping used for the Julia translation.

## Top-level objective
Reproduce `hmcode.power(k, zs, results, T_AGN)` with pure Julia inputs:
- `Pk_lin(k, z)` callable
- `sigma_R(R, z)` callable
- cosmology struct

## HMcode call graph (main path)

`hmcode.power` depends on:
1. utility checks/logspace
2. growth interpolators + accumulated growth
3. Mead spherical-collapse fits (`dc_Mead`, `Dv_Mead`)
4. non-linear summaries from linear spectrum (`sigma8`, `sigmaV`, `Rnl`, `n_eff`)
5. concentration model + Dolag correction
6. NFW profile Fourier transform (`_win_NFW`, optional baryonic version)
7. `pyhalomodel.model.power_spectrum(..., simple_twohalo=True)` one-halo base term
8. HMcode-2020 transition and damping
9. optional feedback suppression ratio recursion

## Primitive → complex translation order

1. **Utilities**
   - monotonic checks, local derivative from samples, logspace/trapz helpers
2. **Linear cosmology helpers**
   - `Tk_EH_nowiggle`, tophat kernels, `sigmaV`
3. **Growth ODE**
   - `_w`, `_X_w`, `_Hubble2`, `_Omega_m`, `_AH`, `get_growth_interpolator`, `get_accumulated_growth`
4. **HMcode helpers**
   - `get_effective_index`, `get_nonlinear_radius`, `get_halo_collapse_redshifts`, dewiggle, feedback parameter map
5. **pyhalomodel core**
   - `HaloModel` constructor (ST/Tinker params)
   - `mass_function_nu`, `linear_bias_nu`, `_peak_height`, `average`
   - `HaloProfile.Fourier` with `Uk`/`Wk` semantics
6. **pyhalomodel power internals**
   - `_I_2h`, `_Pk_2h`, `_Pk_1h`, `power_spectrum`
7. **HMcode assembly**
   - one-halo damping, dewiggled two-halo, alpha transition, optional feedback ratio

## Numerical method replacements

- `scipy.optimize.root_scalar(..., bracket)` → `Roots.find_zero(..., Bisection())`
- `scipy.integrate.quad` → `QuadGK.quadgk`
- `np.trapz`/`scipy.integrate.trapezoid` → `Trapz.trapz`
- `scipy.interpolate.interp1d` usage replaced by Julia interpolation where needed in tests/input wrappers
- `scipy.ndimage.gaussian_filter1d` behavior approximated with reflect-boundary Gaussian stencil

## Validation target

- function-level checks: tight tolerances for translated primitives
- end-to-end checks against saved Python references (`test_data.npz`, `cosmo_test_data.npz`)

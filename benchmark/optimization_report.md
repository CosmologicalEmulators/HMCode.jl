# HMCode.jl optimization report

## Accuracy regression gate
- Regression reference generated from `k=10.^range(-3,1,100)` and `nz=50` deterministic interpolant case.
- Verified post-optimization: `max_rel_err = 9.72205010432725e-15 < 1e-10`.

## nz sweep benchmark (median wall time)
| nz | Before [ms] | After [ms] | Speedup |
|----|------------:|-----------:|--------:|
| 1 |     11.411 |     4.701 |    2.43x |
| 5 |     63.678 |    21.358 |    2.98x |
| 10 |    124.829 |    42.064 |    2.97x |
| 20 |    258.347 |    89.155 |    2.90x |
| 50 |    654.187 |   228.993 |    2.86x |

## Step-by-step timing checkpoints (nz=50)
| Stage | Median time [ms] | Notes |
|---|---:|---|
| Baseline (pre-optimization) | 654.187 | original implementation |
| After Step 2/3 precomputations | 641.425 | sigma(M,z) + scalar params prepass |
| Final optimized | 228.993 | precomputed NFW tensor + mul! one-halo integration + optional threaded path |

## Per-step timing deltas (nz=50)
| Step | Before [ms] | After [ms] | Delta |
|---|---:|---:|---:|
| Step 2/3: sigma-grid + scalar param prepass | 654.187 | 641.425 | -1.95% |
| Step 4/5: NFW tensor precompute + mul! one-halo integration | 641.425 | 228.993 | -64.30% |
| Step 6: threaded fallback path added (serial default) | 228.993 | 228.993 | ~0% in this environment |

## Threaded mode
- Added `threaded::Bool=false` keyword to `hmcode_power` / `hmcode_power_single` (serial default for AD compatibility).
- In this environment, threaded and serial medians are similar (~226.8 ms for nz=50), likely due thread availability/configuration.

## Allocation snapshot (step 8 check)
- `@allocated hmcode_power(...)` for `nk=100,nz=50`: `124,498,552` bytes.
- `Profile.Allocs` snapshot captured (`allocation_check.jl`), records: `1458`.

## Validation status
- Full unit test suite passes after optimization (`30/30`).
- Existing Python parity tests remain unchanged.
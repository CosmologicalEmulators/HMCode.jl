using BenchmarkTools
using Printf
using NPZ

include(joinpath(@__DIR__, "common_setup.jl"))

# Ensure reference exists (requested regression fixture)
ref = ensure_reference!()

case_full = build_regression_case(nk=100, nz=50)
Pref = ref["Pk_ref"]

# Validate exactness against cached reference before benchmarking
Pchk = HMcode.hmcode_power(case_full.k, case_full.zs, case_full.Pk_lin_interp, case_full.sigma_R_interp, case_full.cosmo)
err = max_rel_err(Pchk, Pref)
@assert err < 1e-10 "Accuracy regression before benchmark: $err"

nzs = [1, 5, 10, 20, 50]
results_ms = Dict{Int, Float64}()

println("\nBenchmarking hmcode_power over nz sweep")
println("(nk = 100, nM = default, T_AGN = default)")

for nz in nzs
    zs_sub = case_full.zs[1:nz]
    trial = @benchmark HMcode.hmcode_power($case_full.k, $zs_sub, $case_full.Pk_lin_interp, $case_full.sigma_R_interp, $case_full.cosmo) evals=1 samples=8 seconds=180
    tmed_ms = median(trial).time / 1e6
    results_ms[nz] = tmed_ms
end

println("\n| nz | wall time median [ms] |")
println("|----|-----------------------:|")
for nz in nzs
    @printf("| %2d | %21.3f |\n", nz, results_ms[nz])
end

# Save machine-readable output for before/after comparisons
npzwrite(joinpath(@__DIR__, "bench_nz_latest.npz"), Dict(
    "nzs" => Float64.(nzs),
    "times_ms" => [results_ms[nz] for nz in nzs],
))

println("\nSaved benchmark file: benchmark/bench_nz_latest.npz")

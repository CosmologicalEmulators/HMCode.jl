using Profile
using Chairmarks

include(joinpath(@__DIR__, "common_setup.jl"))

case = build_regression_case(nk=100, nz=50)

println("Chairmarks summary:")
@b HMcode.hmcode_power(case.k, case.zs, case.Pk_lin_interp, case.sigma_R_interp, case.cosmo)

bytes = @allocated HMcode.hmcode_power(case.k, case.zs, case.Pk_lin_interp, case.sigma_R_interp, case.cosmo)
println("\n@allocated bytes = ", bytes)

println("\nCollecting allocation profile snapshot...")
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=0.001 HMcode.hmcode_power(case.k, case.zs, case.Pk_lin_interp, case.sigma_R_interp, case.cosmo)
res = Profile.Allocs.fetch()
println("Allocation profile records: ", length(res.allocs))

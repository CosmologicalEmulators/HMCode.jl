using NPZ
using Interpolations

include(joinpath(@__DIR__, "..", "src", "HMcode.jl"))
using .HMcode

"""
Build deterministic interpolation-backed inputs for regression/benchmarking.
Returns named tuple with `k`, `zs`, `Pk_lin_interp`, `sigma_R_interp`, `cosmo`.

`zs` is returned in decreasing order to satisfy HMcode input checks.
"""
function build_regression_case(; nk::Int=100, nz::Int=50)
    # Requested benchmark grid in k; z grid created as 0..3 then reversed for HMcode
    k = 10.0 .^ range(-3.0, 1.0, length=nk)
    zs_inc = collect(range(0.0, 3.0, length=nz))
    zs = reverse(zs_inc)

    # Internal interpolation support grids
    k_grid = 10.0 .^ range(-5.0, 2.0, length=600)
    R_grid = 10.0 .^ range(-3.0, 2.0, length=500)

    # Smooth, positive proxy for linear spectrum and sigma(R)
    # (deterministic and stable for regression comparisons)
    Pk_table = Array{Float64}(undef, nz, length(k_grid))
    sigma_table = Array{Float64}(undef, nz, length(R_grid))

    @inbounds for (iz, z) in enumerate(zs)
        D = 1.0 / (1.0 + z)
        # mildly wiggly but smooth spectrum shape
        Pk_table[iz, :] .= @. (k_grid^0.965) * exp(-0.18 * k_grid) * (1.0 + 0.04 * sin(5.0 * log(k_grid + 1e-12))) * D^2 + 1e-12

        # monotonic sigma(R) proxy with sensible dynamic range
        sigma_table[iz, :] .= @. 2.8 * D^0.9 * (R_grid^-0.32) / (1.0 + (R_grid / 9.0)^1.8)
    end

    pk_itp = Vector{Any}(undef, nz)
    sig_itp = Vector{Any}(undef, nz)
    logk_grid = log.(k_grid)
    logR_grid = log.(R_grid)

    @inbounds for iz in 1:nz
        pitp = interpolate((logk_grid,), log.(Pk_table[iz, :]), Gridded(Linear()))
        pk_itp[iz] = extrapolate(pitp, Line())

        sitp = interpolate((logR_grid,), log.(sigma_table[iz, :]), Gridded(Linear()))
        sig_itp[iz] = extrapolate(sitp, Line())
    end

    nearest_z_index = z -> argmin(abs.(zs .- z))

    Pk_lin_interp = (kval, zval) -> begin
        iz = nearest_z_index(zval)
        exp(pk_itp[iz](log(kval)))
    end

    sigma_R_interp = (Rval, zval) -> begin
        iz = nearest_z_index(zval)
        exp(sig_itp[iz](log(Rval)))
    end

    cosmo = HMcode.HMcodeCosmology(
        0.314885,
        0.049,
        0.674,
        0.965,
        0.8,
        -1.0,
        0.0,
        0.00142,
        0.0,
    )

    return (
        k = k,
        zs = zs,
        Pk_lin_interp = Pk_lin_interp,
        sigma_R_interp = sigma_R_interp,
        cosmo = cosmo,
    )
end

"""
Compute and cache baseline reference output once.
"""
function ensure_reference!(; force::Bool=false, ref_file::String=joinpath(@__DIR__, "reference_Pk_ref.npz"))
    if (!force) && isfile(ref_file)
        data = npzread(ref_file)
        return data
    end

    case = build_regression_case(nk=100, nz=50)
    Pk_ref = HMcode.hmcode_power(case.k, case.zs, case.Pk_lin_interp, case.sigma_R_interp, case.cosmo)

    npzwrite(ref_file, Dict(
        "k" => case.k,
        "zs" => case.zs,
        "Pk_ref" => Pk_ref,
    ))
    return npzread(ref_file)
end

"""
Max relative error utility.
"""
max_rel_err(Pnew, Pref) = maximum(abs.((Pnew .- Pref) ./ Pref))

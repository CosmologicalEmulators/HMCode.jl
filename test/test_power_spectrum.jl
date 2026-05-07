using Test
using NPZ
using Interpolations
include("../src/HMcode.jl")

using .HMcode

@testset "Full Power Spectrum" begin
    data = npzread(joinpath(@__DIR__, "../../test_data.npz"))
    k_arr = data["k"]
    zs_arr = data["zs"]
    Pk_hm_ref = data["Pk_hm"]
    Pk_lin_arr = data["Pk_lin"] # shape (4, 50)
    R_grid = data["R_grid"]
    sigma_grid = data["sigma_grid"] # shape (4, 100)
    
    # Create interpolants
    function Pk_lin(k_val, z_val)
        iz = findfirst(x -> x == z_val, zs_arr)
        itp = interpolate((log.(k_arr),), log.(Pk_lin_arr[iz, :]), Gridded(Linear()))
        extrap = extrapolate(itp, Line())
        return exp(extrap(log(k_val)))
    end
    
    function sigma_R(R_val, z_val)
        iz = findfirst(x -> x == z_val, zs_arr)
        itp = interpolate((log.(R_grid),), log.(sigma_grid[iz, :]), Gridded(Linear()))
        extrap = extrapolate(itp, Line())
        return exp(extrap(log(R_val)))
    end
    
    cosmo = HMcode.HMcodeCosmology(
        0.30, # Omega_m (0.25 + 0.05)
        0.05, # Omega_b
        0.70, # h
        0.96, # n_s
        0.8,  # sigma_8 (doesn't matter since sigma_R interpolant overrides)
        -1.0, # w0
        0.0,  # wa
        0.0,  # Omega_nu
        0.0   # Omega_k
    )
    
    Pk_HMcode = HMcode.hmcode_power(
        k_arr, zs_arr, Pk_lin, sigma_R, cosmo,
        T_AGN=10^7.8, nM=64, Mmin=1e0, Mmax=1e18
    )
    
    # Convert Pk_HMcode to match Pk_hm_ref shape if necessary.
    # Julia output: [nk, nz]. Python output: [nz, nk]
    Pk_HMcode_transposed = transpose(Pk_HMcode)
    
    @test isapprox(Pk_HMcode_transposed, Pk_hm_ref, rtol=5e-3)
end

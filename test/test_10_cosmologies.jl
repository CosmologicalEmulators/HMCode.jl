using Test
using NPZ
using Interpolations
include("../src/HMcode.jl")
using .HMcode

@testset "10 Cosmologies Validation" begin
    data = npzread(joinpath(@__DIR__, "../../cosmo_test_data.npz"))
    k_arr = data["k"]
    k_grid_lin = data["k_grid_lin"]
    zs_arr = data["zs"]
    R_grid = data["R_grid"]

    for i in 0:9
        Pk_hm_ref = data["cfg_$(i)_Pk_hm"]
        Pk_lin_arr = data["cfg_$(i)_Pk_lin_grid"]
        sigma_grid = data["cfg_$(i)_sigma_grid"]
        params = data["cfg_$(i)_params"]
        
        Om_m, Om_b, h, n_s, w0, wa, Om_nu = params
        
        function Pk_lin(k_val, z_val)
            iz = findfirst(x -> isapprox(x, z_val, atol=1e-3), zs_arr)
            itp = interpolate((log.(k_grid_lin),), log.(Pk_lin_arr[iz, :]), Gridded(Linear()))
            extrap = extrapolate(itp, Line())
            return exp(extrap(log(k_val)))
        end
        
        function sigma_R(R_val, z_val)
            iz = findfirst(x -> isapprox(x, z_val, atol=1e-3), zs_arr)
            itp = interpolate((log.(R_grid),), log.(sigma_grid[iz, :]), Gridded(Linear()))
            extrap = extrapolate(itp, Line())
            return exp(extrap(log(R_val)))
        end
        
        cosmo = HMcode.HMcodeCosmology(Om_m, Om_b, h, n_s, 0.8, w0, wa, Om_nu, 0.0)
        
        Pk_HMcode = HMcode.hmcode_power(
            k_arr, zs_arr, Pk_lin, sigma_R, cosmo,
            T_AGN=10^7.8, nM=64, Mmin=1e0, Mmax=1e18
        )
        
        Pk_HMcode_transposed = transpose(Pk_HMcode)
        
        rel_err = abs.(Pk_HMcode_transposed .- Pk_hm_ref) ./ Pk_hm_ref
        mean_err = sum(rel_err) / length(rel_err)
        max_err = maximum(rel_err)
        
        println("Config $i: Max Error = $(round(max_err*100, digits=3))%, Mean Error = $(round(mean_err*100, digits=3))%")
        
        @test mean_err < 0.005 # Less than 0.5% average error
    end
end

using Test
include("../src/cosmology.jl")
include("../src/linear.jl")
include("../src/halomodel.jl")

@testset "HaloModel Tinker" begin
    z = 0.0
    Om_m = 0.3
    Dv = 200.0
    dc = 1.686
    
    hmod = HaloModel(z, Om_m, Dv, dc)
    
    @test isapprox(hmod.tinker.alpha, 0.368, rtol=1e-5)
    @test isapprox(hmod.tinker.beta, 0.589, rtol=1e-5)
    @test isapprox(hmod.tinker.gamma, 0.864, rtol=1e-5)
    @test isapprox(hmod.tinker.phi, -0.729, rtol=1e-5)
    @test isapprox(hmod.tinker.eta, -0.243, rtol=1e-5)
    
    @test isapprox(hmod.tinker.A, 1.0000597439, rtol=1e-5)
    @test isapprox(hmod.tinker.a, 0.132453198, rtol=1e-5)
    @test isapprox(hmod.tinker.C, 0.265230764, rtol=1e-5)
    
    nu = [0.5, 1.0, 2.0]
    g_nu = mass_function_nu(hmod, nu)
    b_nu = linear_bias_nu(hmod, nu)
    
    g_ref = [0.54047579, 0.34933235, 0.10594267]
    b_ref = [0.65508733, 0.96549205, 2.4118132]
    
    @test isapprox(g_nu, g_ref, rtol=1e-5)
    @test isapprox(b_nu, b_ref, rtol=1e-5)
    
    M = 10.0 .^ range(10, 15, length=10)
    sigmaM = 10.0 ./ log10.(M)
    mf = mass_function(hmod, M, sigmaM)
    
    mf_ref = [1.01252366e-11, 6.92694571e-13, 4.69320678e-14, 3.15766029e-15,
              2.10948095e-16, 1.39912250e-17, 9.21230805e-19, 6.02119035e-20,
              3.90634882e-21, 2.51177382e-22]
              
    @test isapprox(mf, mf_ref, rtol=2e-5)
    
    avg = average(hmod, M, sigmaM, M)
    # Python returned average: 6611264264.314587
    @test isapprox(avg, 6611264264.314587, rtol=2e-5)
end

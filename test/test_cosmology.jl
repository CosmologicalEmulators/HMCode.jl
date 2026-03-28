using Test
include("../src/cosmology.jl")

@testset "Cosmology formulas" begin
    # Reference cosmology parameters
    Om_m = 0.3
    f_nu = 0.0
    g = 0.8
    G = 0.8
    a = 1.0

    dc = dc_Mead(a, Om_m, f_nu, g, G)
    Dv = Dv_Mead(a, Om_m, f_nu, g, G)

    @test isapprox(dc, 1.6920741332; rtol=1e-5)
    @test isapprox(Dv, 103.9611823032; rtol=1e-5)
end
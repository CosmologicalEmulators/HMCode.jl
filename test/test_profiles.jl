using Test
include("../src/cosmology.jl")
include("../src/profiles.jl")

@testset "Halo Profiles" begin
    k = [0.1, 1.0, 10.0]
    rv = 1.5
    c = 4.0
    W = win_NFW(k, rv, c)
    
    W_ref = [0.99883434, 0.89053616, 0.07072137]
    @test isapprox(W, W_ref, rtol=1e-5)

    M = 1e12
    Mb = 1e14
    fstar = 0.03
    Om_m = 0.3
    Om_c = 0.25
    Om_b = 0.05
    W_b = win_NFW_baryons(k, rv, c, M, Mb, fstar, Om_m, Om_c, Om_b)
    
    W_b_ref = [0.8623756, 0.77212564, 0.08893544]
    @test isapprox(W_b, W_b_ref, rtol=1e-5)
end

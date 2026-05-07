using Test
include("../src/linear.jl")

@testset "Linear Perturbations" begin
    k = 10.0 .^ range(-3, 1, length=5)
    h = 0.7
    wm = 0.14
    wb = 0.022
    Tk = Tk_EH_nowiggle(k, h, wm, wb)
    
    Tk_ref = [0.991371374, 0.775381287, 0.126412368, 0.00432618329, 8.20030937e-05]
    @test isapprox(Tk, Tk_ref, rtol=1e-5)

    xs = 10.0 .^ range(-1, 1, length=20)
    fs = xs .^ 3
    deriv = derivative_from_samples(0.5, xs, fs)
    @test isapprox(deriv, 0.7584092040209818, rtol=1e-5)

    ratio = sin.(log.(k) .* 10.0)
    sm = gaussian_filter1d(ratio, 1.5)
    sm_ref = [-0.14854814, -0.10097003, -0.08286888, -0.18359124, -0.32479379]
    @test isapprox(sm, sm_ref, rtol=1e-5)
end

using Test
include("../src/HMcode.jl")
using .HMcode

@testset "Feedback" begin
    feedback_params = HMcode.get_feedback_parameters(10.0^7.7)
    Omega_b = 0.051
    Omega_m = 0.312
    baryon_fraction = Omega_b / Omega_m

    uncapped_low_z = feedback_params[:f0] * 10.0^(1.0 * feedback_params[:fz])
    @test uncapped_low_z < baryon_fraction
    @test HMcode.feedback_stellar_fraction(feedback_params, 1.0, Omega_b, Omega_m) ≈ uncapped_low_z

    uncapped_high_z = feedback_params[:f0] * 10.0^(4.0 * feedback_params[:fz])
    @test uncapped_high_z > baryon_fraction
    @test HMcode.feedback_stellar_fraction(feedback_params, 4.0, Omega_b, Omega_m) == baryon_fraction
end

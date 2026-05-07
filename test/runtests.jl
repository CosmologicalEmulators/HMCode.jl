using Test

@testset "HMcode" begin
    include("test_cosmology.jl")
    include("test_linear.jl")
    include("test_profiles.jl")
    include("test_halomodel.jl")
    include("test_power_spectrum.jl")
    include("test_10_cosmologies.jl")
end

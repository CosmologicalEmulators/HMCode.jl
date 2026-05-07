module HMcode

using SpecialFunctions
export hmcode_power, HMcodeCosmology

include("cosmology.jl")
include("linear.jl")
include("profiles.jl")
include("halomodel.jl")
include("feedback.jl")
include("power_spectrum.jl")

end # module

module HMcode

using SpecialFunctions
export hmcode_power, hmcode_power!, HMcodeCosmology, HMcodeWorkspace, wnfw_xc, wnfw_fast, Fast2DInterp

include("cosmology.jl")
include("linear.jl")
include("profiles.jl")
include("halomodel.jl")
include("feedback.jl")
include("interpolants.jl")
include("power_spectrum.jl")

end # module

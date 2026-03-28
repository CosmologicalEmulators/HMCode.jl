# feedback.jl

function get_feedback_parameters(T_AGN::Float64)
    theta = log10(T_AGN / 10.0^7.8)
    params = Dict{Symbol, Float64}(
        :B0 => 3.44 - 0.496 * theta,
        :Bz => -0.0671 - 0.0371 * theta,
        :Mb0 => 10.0^(13.87 + 1.81 * theta),
        :Mbz => -0.108 + 0.195 * theta,
        :f0 => (2.01 - 0.3 * theta) * 1e-2,
        :fz => 0.409 + 0.0224 * theta
    )
    return params
end

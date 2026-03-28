# profiles.jl
using SpecialFunctions
using Roots

"""
Container for halo profiles in Fourier space.

Fields follow pyhalomodel semantics:
- `Uk` stores dimensionless profile transforms
- `Wk` stores dimensionful profile transforms after amplitude/normalisation
"""
struct HaloProfile
    k::Vector{Float64}
    M::Vector{Float64}
    Uk::Matrix{Float64}           # [nk, nM]
    Wk::Matrix{Float64}           # [nk, nM]
    amplitude::Vector{Float64}
    normalisation::Float64
    variance::Union{Nothing, Vector{Float64}}
    mass_tracer::Bool
    discrete_tracer::Bool
end

"""
Create a Fourier-space halo profile equivalent to `pyhalomodel.profile.Fourier`.

If `amplitude === nothing`, `Uk` is treated as dimensionful and amplitudes are inferred
from low-k values (first k-bin), then converted to dimensionless internally.
"""
function Fourier(
    k::AbstractVector,
    M::AbstractVector,
    Uk_in::AbstractMatrix;
    amplitude::Union{Nothing, AbstractVector} = nothing,
    normalisation::Real = 1.0,
    variance::Union{Nothing, AbstractVector} = nothing,
    mass_tracer::Bool = false,
    discrete_tracer::Bool = false,
)
    nk, nM = length(k), length(M)
    size(Uk_in) == (nk, nM) || throw(ArgumentError("Uk shape must be (length(k), length(M))"))

    Uk = Array{Float64}(Uk_in)
    Wk = similar(Uk)

    amp = if amplitude === nothing
        # Infer dimensionful amplitudes from k->0 row
        vec(Uk[1, :])
    else
        length(amplitude) == nM || throw(ArgumentError("amplitude must have length(M) entries"))
        collect(Float64.(amplitude))
    end

    if amplitude === nothing
        # Input is dimensionful; normalize internally to dimensionless Uk and keep Wk as given
        Wk .= Uk
        @inbounds for j in 1:nM
            Uk[:, j] ./= amp[j]
        end
    else
        # Input is already dimensionless Uk; build dimensionful Wk
        @inbounds for j in 1:nM
            Wk[:, j] .= (amp[j] .* Uk[:, j]) ./ float(normalisation)
        end
    end

    var = variance === nothing ? nothing : collect(Float64.(variance))
    return HaloProfile(collect(Float64.(k)), collect(Float64.(M)), Uk, Wk, amp, float(normalisation), var, mass_tracer, discrete_tracer)
end

# ------------------------------------------------------------
# NFW window functions
# ------------------------------------------------------------

"""Normalised Fourier transform for an NFW profile (scalar rv, c)."""
function win_NFW(k::AbstractVector, rv::Real, c::Real)
    rs = rv / c
    kv = k .* rv
    ks = k .* rs

    Sisv = sinint.(ks .+ kv)
    Cisv = cosint.(ks .+ kv)
    Sis = sinint.(ks)
    Cis = cosint.(ks)

    f1 = @. cos(ks) * (Cisv - Cis)
    f2 = @. sin(ks) * (Sisv - Sis)
    f3 = @. sin(kv) / (ks + kv)
    f4 = log(1.0 + c) - c / (1.0 + c)

    return @. (f1 + f2 - f3) / f4
end

"""
Normalised Fourier transform for NFW profile including baryonic corrections
(Equation 25 in Mead et al. 2021).
"""
function win_NFW_baryons(
    k::AbstractVector,
    rv::Real,
    c::Real,
    M::Real,
    Mb::Real,
    fstar::Real,
    Om_m::Real,
    Om_c::Real,
    Om_b::Real,
)
    Wk = win_NFW(k, rv, c)
    fg = (Om_b / Om_m - fstar) * (M / Mb)^2 / (1.0 + (M / Mb)^2)
    return @. (Om_c / Om_m + fg) * Wk + fstar
end


"""
Bullock et al. (2001)-style halo collapse redshifts used by HMcode concentration model.
"""
function get_halo_collapse_redshifts(
    M::AbstractVector,
    z::Real,
    dc::Real,
    Om_m::Real,
    growth,
    sigmaR_func,
)
    gamma = 0.01
    a = scalefactor_from_redshift(z)

    zf = similar(collect(Float64.(M)))
    @inbounds for (i, m) in enumerate(M)
        Mc = gamma * m
        Rc = Lagrangian_radius(Mc, Om_m)
        sigma = sigmaR_func(Rc)
        fac = growth(a) * dc / sigma

        af = if fac >= growth(a)
            a
        else
            find_zero(af -> growth(af) - fac, (1e-3, 1.0), Bisection())
        end
        zf[i] = redshift_from_scalefactor(af)
    end

    return zf
end

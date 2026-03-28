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
        vec(Uk[1, :])
    else
        length(amplitude) == nM || throw(ArgumentError("amplitude must have length(M) entries"))
        collect(Float64.(amplitude))
    end

    if amplitude === nothing
        Wk .= Uk
        @inbounds for j in 1:nM
            Uk[:, j] ./= amp[j]
        end
    else
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

"""Scalar NFW Fourier window from x ≡ k*rv and concentration c."""
@inline function wnfw_xc(x::Float64, c::Float64)
    ks = x / c
    Sisv = sinint(ks + x)
    Cisv = cosint(ks + x)
    Sis = sinint(ks)
    Cis = cosint(ks)

    f1 = cos(ks) * (Cisv - Cis)
    f2 = sin(ks) * (Sisv - Sis)
    f3 = sin(x) / (ks + x)
    f4 = log(1.0 + c) - c / (1.0 + c)
    return (f1 + f2 - f3) / f4
end

"""
In-place NFW Fourier window evaluation.
`rv_eff` allows folding the HMcode bloating factor into radius (rv * ν^η).
"""
function win_NFW!(out::AbstractVector{Float64}, k::AbstractVector, rv_eff::Float64, c::Float64)
    @inbounds for i in eachindex(k)
        out[i] = wnfw_xc(k[i] * rv_eff, c)
    end
    return out
end

"""Normalised Fourier transform for an NFW profile (scalar rv, c)."""
function win_NFW(k::AbstractVector, rv::Real, c::Real)
    out = Vector{Float64}(undef, length(k))
    return win_NFW!(out, k, float(rv), float(c))
end

"""
In-place NFW+baryons profile.
"""
function win_NFW_baryons!(
    out::AbstractVector{Float64},
    k::AbstractVector,
    rv_eff::Float64,
    c::Float64,
    M::Float64,
    Mb::Float64,
    fstar::Float64,
    Om_m::Float64,
    Om_c::Float64,
    Om_b::Float64,
)
    win_NFW!(out, k, rv_eff, c)
    fg = (Om_b / Om_m - fstar) * (M / Mb)^2 / (1.0 + (M / Mb)^2)
    coeff = Om_c / Om_m + fg
    @inbounds for i in eachindex(out)
        out[i] = coeff * out[i] + fstar
    end
    return out
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
    out = Vector{Float64}(undef, length(k))
    return win_NFW_baryons!(
        out,
        k,
        float(rv),
        float(c),
        float(M),
        float(Mb),
        float(fstar),
        float(Om_m),
        float(Om_c),
        float(Om_b),
    )
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

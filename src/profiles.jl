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

const SI_SMALL = (1.0, -1.0/18.0, 1.0/600.0, -1.0/35280.0, 1.0/3265920.0,
                  -1.0/402361344.0, 1.0/62270208000.0,
                  -1.0/11564863119360.0, 1.0/2504902400942080.0)

@inline si_small(x) = x * evalpoly(x*x, SI_SMALL)

@inline function si_large(x)
    u = 1.0/x
    u2 = u*u
    f = u  * evalpoly(u2, (1.0, -2.0, 24.0, -720.0, 40320.0))
    g = u2 * evalpoly(u2, (1.0, -6.0, 120.0, -5040.0, 362880.0))
    s, c = sin(x), cos(x)
    return pi/2.0 - f*c - g*s
end

@inline si_fast(x) = ifelse(x < 4.0, si_small(x), si_large(x))

const CI_INT_SMALL = (-0.25, 1.0/96.0, -1.0/4320.0, 1.0/322560.0,
                      -1.0/36288000.0, 1.0/6402373248000.0)

@inline ci_int_small(x) = x*x * evalpoly(x*x, CI_INT_SMALL)

const EULER_GAMMA = 0.5772156649015329

@inline function ci_int_large(x)
    u = 1.0/x
    u2 = u*u
    f = u  * evalpoly(u2, (1.0, -2.0, 24.0, -720.0, 40320.0))
    g = u2 * evalpoly(u2, (1.0, -6.0, 120.0, -5040.0, 362880.0))
    s, c = sin(x), cos(x)
    ci = f*s - g*c
    return ci - log(x) - EULER_GAMMA
end

@inline ci_int_fast(x) = ifelse(x < 4.0, ci_int_small(x), ci_int_large(x))

"""Fast scalar NFW window from x ≡ k*rv, c and precomputed ln(1+c)."""
@inline function wnfw_fast(x::Float64, c::Float64, ln1pc::Float64)::Float64
    x_plus  = x * (1.0 + 1.0/c)
    x_minus = x / c
    ΔSi = si_fast(x_plus) - si_fast(x_minus)
    ΔCi = ln1pc + ci_int_fast(x_plus) - ci_int_fast(x_minus)
    s, cv = sin(x_minus), cos(x_minus)
    sinc_xp = sin(x) / x_plus
    norm = ln1pc - c/(1.0 + c)
    return (ΔSi*s + ΔCi*cv - sinc_xp) / norm
end

using LoopVectorization

function win_NFW_fast!(out::AbstractVector{Float64}, k::AbstractVector, rv_eff::Float64, c::Float64, ln1pc::Float64)
    @turbo for i in eachindex(k)
        out[i] = wnfw_fast(k[i] * rv_eff, c, ln1pc)
    end
    return out
end

function win_NFW_fast!(W_buf::AbstractMatrix{Float64}, k::AbstractVector, rv_eff::AbstractVector, c_vec::AbstractVector, ln1pc::AbstractVector)
    nM, nk = size(W_buf)
    @turbo for ik in 1:nk, iM in 1:nM
        x = k[ik] * rv_eff[iM]
        W_buf[iM, ik] = wnfw_fast(x, c_vec[iM], ln1pc[iM])
    end
    return W_buf
end

function win_NFW_baryons!(
    out::AbstractVector{Float64},
    k::AbstractVector,
    rv_eff::Float64,
    c::Float64,
    ln1pc::Float64,
    M::Float64,
    Mb::Float64,
    fstar::Float64,
    Om_m::Float64,
    Om_c::Float64,
    Om_b::Float64,
)
    win_NFW_fast!(out, k, rv_eff, c, ln1pc)
    fg = (Om_b / Om_m - fstar) * (M / Mb)^2 / (1.0 + (M / Mb)^2)
    coeff = Om_c / Om_m + fg
    @inbounds for i in eachindex(out)
        out[i] = coeff * out[i] + fstar
    end
    return out
end

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

struct CollapseScaleRoot{G}
    growth_itp::G
    fac::Float64
end
@inline function (s::CollapseScaleRoot)(af::Float64)::Float64
    return s.growth_itp(af) - s.fac
end

function compute_collapse_redshifts_exact(
    M_grid::AbstractVector,
    z::Real,
    dc::Real,
    Om_m::Real,
    growth_itp,
    sigmaR_func
)
    gamma = 0.01
    a = scalefactor_from_redshift(z)
    g_a = growth_itp(a)

    zf = zeros(Float64, length(M_grid))
    @inbounds for iM in eachindex(M_grid)
        Mc = gamma * M_grid[iM]
        Rc = Lagrangian_radius(Mc, Om_m)
        sigma = sigmaR_func(Rc)
        fac = g_a * dc / sigma

        if fac >= g_a
            zf[iM] = z
        else
            f = CollapseScaleRoot(growth_itp, float(fac))
            af = find_zero(f, (1e-3, 1.0), Bisection())
            zf[iM] = redshift_from_scalefactor(af)
        end
    end

    return zf
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
    return compute_collapse_redshifts_exact(M, z, dc, Om_m, growth, sigmaR_func)
end

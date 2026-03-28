# power_spectrum.jl
using LinearAlgebra
using Base.Threads

const ND_HMCODE = 2.853

struct HMcodeParams
    R_nl::Vector{Float64}
    n_eff::Vector{Float64}
    C_curv::Vector{Float64}
    sigma_v::Vector{Float64}
    Delta_v::Vector{Float64}
    delta_c::Vector{Float64}
    eta::Vector{Float64}
    A::Vector{Float64}
    f_damp::Vector{Float64}
    k_star::Vector{Float64}
    B::Vector{Float64}
    k_damp::Vector{Float64}
end

mutable struct SliceBuffers
    g_nu::Vector{Float64}
    wt_nu::Vector{Float64}
    wvec::Vector{Float64}
    W2buf::Matrix{Float64}
    Pk_1h_raw::Vector{Float64}
    Pk_1h::Vector{Float64}
end

_new_slice_buffers(nk::Int, nM::Int) = SliceBuffers(
    zeros(Float64, nM),
    zeros(Float64, nM),
    zeros(Float64, nM),
    Matrix{Float64}(undef, nk, nM),
    zeros(Float64, nk),
    zeros(Float64, nk),
)

"""Precompute Σ(M,z) on an [nM,nz] grid via broadcasting."""
function compute_sigma_grid(R_grid::AbstractVector, zs::AbstractVector, sigma_R::Function)
    return sigma_R.(reshape(R_grid, :, 1), reshape(zs, 1, :))
end

"""Fill trapezoidal weights for integration over a non-uniform x-grid."""
function trapz_weights!(w::AbstractVector{Float64}, x::AbstractVector)
    n = length(x)
    @assert length(w) == n
    @assert n >= 2
    w[1] = 0.5 * (x[2] - x[1])
    @inbounds for i in 2:n-1
        w[i] = 0.5 * (x[i+1] - x[i-1])
    end
    w[n] = 0.5 * (x[n] - x[n-1])
    return w
end

"""In-place Sheth-Tormen mass-function g(nu)."""
function mass_function_nu_st!(out::AbstractVector{Float64}, nu::AbstractVector, A::Float64, p::Float64, q::Float64)
    @inbounds for i in eachindex(nu)
        ν2 = nu[i] * nu[i]
        out[i] = A * (1.0 + (q * ν2)^(-p)) * exp(-q * ν2 / 2.0)
    end
    return out
end

"""
Compute per-redshift HMcode scalar parameters used in the nonlinear model.
"""
function compute_hmcode_params(
    k::AbstractVector,
    zs::AbstractVector,
    Pk_lin::Function,
    sigma_R::Function,
    sigma_grid::AbstractMatrix,
    R_grid::AbstractVector,
    cosmo::HMcodeCosmology;
    tweaks::Bool = true,
    kmin_sigmaV::Float64 = 1e-5,
)
    nz = length(zs)
    Om_m = cosmo.Omega_m
    f_nu = cosmo.Omega_nu / Om_m

    R_nl   = zeros(Float64, nz)
    n_eff  = zeros(Float64, nz)
    C_curv = zeros(Float64, nz)  # currently unused in this implementation
    sigma_v = zeros(Float64, nz)
    Delta_v = zeros(Float64, nz)
    delta_c = zeros(Float64, nz)
    eta = zeros(Float64, nz)
    A = zeros(Float64, nz)
    f_damp = zeros(Float64, nz)
    k_star = zeros(Float64, nz)
    B = zeros(Float64, nz)
    k_damp = zeros(Float64, nz)

    growth = get_growth_interpolator(cosmo, LCDM=false)

    @inbounds for iz in eachindex(zs)
        z = zs[iz]
        a = scalefactor_from_redshift(z)

        Om_mz = _Omega_m_a(a, cosmo, LCDM=false)
        g = growth(a)
        G = get_accumulated_growth(a, growth)
        dc = dc_Mead(a, Om_mz, f_nu, g, G)
        Dv = Dv_Mead(a, Om_mz, f_nu, g, G)

        delta_c[iz] = dc
        Delta_v[iz] = Dv

        sigmaM = @view sigma_grid[:, iz]
        sigmaR_func_z = r -> sigma_R(r, z)
        Rnl = get_nonlinear_radius(R_grid[1], R_grid[end], dc, sigmaR_func_z)
        sigma8 = sigma_R(8.0, z)
        sigmaV_val = sigmaV(0.0, kk -> Pk_lin(kk, z); kmin=kmin_sigmaV)
        neff = get_effective_index(Rnl, R_grid, sigmaM)

        R_nl[iz] = Rnl
        sigma_v[iz] = sigmaV_val
        n_eff[iz] = neff

        ks = 0.05618 * sigma8^(-1.013)
        k_star[iz] = ks

        if tweaks
            k_damp[iz] = 0.05699 * sigma8^(-1.089)
            f_damp[iz] = 0.2696 * sigma8^(0.9403)
            eta[iz] = 0.1281 * sigma8^(-0.3644)
            B[iz] = 5.196
            A[iz] = 1.875 * (1.603)^neff
        else
            k_damp[iz] = 0.0
            f_damp[iz] = 0.0
            eta[iz] = 0.0
            B[iz] = 4.0
            A[iz] = 1.0
        end
    end

    return HMcodeParams(R_nl, n_eff, C_curv, sigma_v, Delta_v, delta_c, eta, A, f_damp, k_star, B, k_damp)
end

function _assemble_slice!(
    Pk_out::AbstractMatrix,
    iz::Int,
    k::AbstractVector,
    Pk_lin_mat::AbstractMatrix,
    W::Array{Float64,3},
    nu_mat::AbstractMatrix,
    hmpars::HMcodeParams,
    M::AbstractVector,
    amp_no::AbstractVector,
    amp_fb::AbstractVector,
    rhom::Float64,
    cosmo::HMcodeCosmology,
    tweaks::Bool,
    T_AGN::Union{Nothing,Float64},
    p_st::Float64,
    q_st::Float64,
    A_st::Float64,
    buf::SliceBuffers,
)
    nk = length(k)
    nM = length(M)

    nu = @view nu_mat[:, iz]
    mass_function_nu_st!(buf.g_nu, nu, A_st, p_st, q_st)
    trapz_weights!(buf.wt_nu, nu)

    if (T_AGN !== nothing) && (!tweaks)
        @inbounds for i in 1:nM
            a2 = amp_fb[i] * amp_fb[i]
            buf.wvec[i] = (buf.g_nu[i] / M[i]) * a2 * buf.wt_nu[i]
        end
    else
        @inbounds for i in 1:nM
            a2 = amp_no[i] * amp_no[i]
            buf.wvec[i] = (buf.g_nu[i] / M[i]) * a2 * buf.wt_nu[i]
        end
    end

    @inbounds for j in 1:nM, i in 1:nk
        wv = W[i, j, iz]
        buf.W2buf[i, j] = wv * wv
    end

    mul!(buf.Pk_1h_raw, buf.W2buf, buf.wvec)
    @. buf.Pk_1h_raw = buf.Pk_1h_raw * rhom

    ks = hmpars.k_star[iz]
    @inbounds for ik in 1:nk
        x = k[ik] / ks
        x4 = x * x
        x4 *= x4
        buf.Pk_1h[ik] = x4 / (1.0 + x4) * buf.Pk_1h_raw[ik]
    end

    if tweaks
        kd = hmpars.k_damp[iz]
        f = hmpars.f_damp[iz]
        alpha = hmpars.A[iz]

        omega_m = cosmo.Omega_m * cosmo.h^2
        omega_b = cosmo.Omega_b * cosmo.h^2
        Pk_lin_k = @view Pk_lin_mat[:, iz]

        Pk_wig = get_Pk_wiggle(k, Pk_lin_k, cosmo.h, omega_m, omega_b, cosmo.n_s)

        @inbounds for ik in 1:nk
            damp_lin = 1.0 - exp(-(k[ik] * hmpars.sigma_v[iz])^2)
            Pk_dwl = Pk_lin_k[ik] - damp_lin * Pk_wig[ik]

            y = (k[ik] / kd)^ND_HMCODE
            Pk_2h = Pk_dwl * (1.0 - f * y / (1.0 + y))

            Pk_out[ik, iz] = (Pk_2h^alpha + buf.Pk_1h[ik]^alpha)^(1.0 / alpha)
        end
    else
        @inbounds for ik in 1:nk
            Pk_out[ik, iz] = Pk_lin_mat[ik, iz] + buf.Pk_1h[ik]
        end
    end

    return nothing
end

"""
Internal single-pass HMcode computation.
Returns `Pk[nk, nz]`.
"""
function hmcode_power_single(
    k::AbstractVector,
    zs::AbstractVector,
    Pk_lin::Function,
    sigma_R::Function,
    cosmo::HMcodeCosmology;
    T_AGN::Union{Nothing, Float64} = nothing,
    Mmin::Float64 = 1e0,
    Mmax::Float64 = 1e18,
    nM::Int = 256,
    tweaks::Bool = true,
    kmin_sigmaV::Float64 = 1e-5,
    threaded::Bool = false,
)
    is_array_monotonic(-collect(Float64.(zs))) || throw(ArgumentError("Redshifts must be monotonically decreasing"))
    is_array_monotonic(collect(Float64.(k))) || throw(ArgumentError("k must be monotonically increasing"))

    nk = length(k)
    nz = length(zs)

    # Mass/radius grids
    logM_grid = range(log(Mmin), log(Mmax), length=nM)
    M = exp.(logM_grid)
    Om_m = cosmo.Omega_m
    R = Lagrangian_radius(M, Om_m)

    # Precompute sigma(M,z)
    Σ = compute_sigma_grid(R, zs, sigma_R)

    # Precompute per-z scalar parameters
    hmpars = compute_hmcode_params(
        k,
        zs,
        Pk_lin,
        sigma_R,
        Σ,
        R,
        cosmo;
        tweaks=tweaks,
        kmin_sigmaV=kmin_sigmaV,
    )

    zc = 10.0
    ac = scalefactor_from_redshift(zc)

    Om_b = cosmo.Omega_b
    Om_nu = cosmo.Omega_nu
    Om_c = Om_m - Om_b - Om_nu
    f_nu = Om_nu / Om_m
    rhom = comoving_matter_density(Om_m)

    growth = get_growth_interpolator(cosmo, LCDM=false)
    growth_LCDM = get_growth_interpolator(cosmo, LCDM=true)

    feedback_params = (T_AGN === nothing) ? Dict{Symbol, Float64}() : get_feedback_parameters(T_AGN)

    # Precompute ν(M,z)
    nu_mat = similar(Σ)
    @inbounds for iz in 1:nz
        @views nu_mat[:, iz] .= hmpars.delta_c[iz] ./ Σ[:, iz]
    end

    # Precompute linear power matrix on requested k-grid
    Pk_lin_mat = Matrix{Float64}(undef, nk, nz)
    @inbounds for iz in 1:nz
        z = zs[iz]
        for ik in 1:nk
            Pk_lin_mat[ik, iz] = Pk_lin(k[ik], z)
        end
    end

    # Precompute full NFW tensor W[nk, nM, nz]
    W = Array{Float64}(undef, nk, nM, nz)
    nfw_buf = Vector{Float64}(undef, nk)

    @inbounds for iz in 1:nz
        z = zs[iz]
        a = scalefactor_from_redshift(z)
        dc = hmpars.delta_c[iz]
        Dv = hmpars.Delta_v[iz]
        eta = hmpars.eta[iz]

        B = hmpars.B[iz]
        Mb, fstar = 0.0, 0.0
        if (T_AGN !== nothing) && (!tweaks)
            B = feedback_params[:B0] * 10.0^(z * feedback_params[:Bz])
            Mb = feedback_params[:Mb0] * 10.0^(z * feedback_params[:Mbz])
            fstar = feedback_params[:f0] * 10.0^(z * feedback_params[:fz])
        end

        sigmaR_func_z = r -> sigma_R(r, z)
        zf = get_halo_collapse_redshifts(M, z, dc, Om_m, growth, sigmaR_func_z)

        dolag = (growth(ac) / growth_LCDM(ac)) * (growth_LCDM(a) / growth(a))
        rv_scale = cbrt(Dv)

        for iM in 1:nM
            rv = R[iM] / rv_scale
            c = B * (1.0 + zf[iM]) / (1.0 + z) * dolag
            rv_eff = rv * (nu_mat[iM, iz]^eta)

            if (T_AGN !== nothing) && (!tweaks)
                win_NFW_baryons!(
                    nfw_buf,
                    k,
                    rv_eff,
                    c,
                    M[iM],
                    Mb,
                    fstar,
                    Om_m,
                    Om_c,
                    Om_b,
                )
            else
                win_NFW!(nfw_buf, k, rv_eff, c)
            end

            @views W[:, iM, iz] .= nfw_buf
        end
    end

    # ST mass function parameters (used by this HMcode path)
    p_st = 0.3
    q_st = 0.707
    A_st = sqrt(2.0 * q_st) / (sqrt(pi) + gamma(0.5 - p_st) / 2.0^p_st)

    amp_no = M .* (1.0 - f_nu) ./ rhom
    amp_fb = M ./ rhom

    Pk_out = zeros(Float64, nk, nz)

    nbuf = threaded ? Threads.maxthreadid() : 1
    bufs = [_new_slice_buffers(nk, nM) for _ in 1:nbuf]

    if threaded
        Threads.@threads for iz in 1:nz
            buf = bufs[Threads.threadid()]
            _assemble_slice!(Pk_out, iz, k, Pk_lin_mat, W, nu_mat, hmpars, M, amp_no, amp_fb, rhom, cosmo, tweaks, T_AGN, p_st, q_st, A_st, buf)
        end
    else
        buf = bufs[1]
        @inbounds for iz in 1:nz
            _assemble_slice!(Pk_out, iz, k, Pk_lin_mat, W, nu_mat, hmpars, M, amp_no, amp_fb, rhom, cosmo, tweaks, T_AGN, p_st, q_st, A_st, buf)
        end
    end

    return Pk_out
end

function _get_feedback_suppression(
    k::AbstractVector,
    zs::AbstractVector,
    Pk_lin::Function,
    sigma_R::Function,
    cosmo::HMcodeCosmology,
    T_AGN::Float64;
    Mmin::Float64 = 1e0,
    Mmax::Float64 = 1e18,
    nM::Int = 256,
    threaded::Bool = false,
)
    Pk_gravity = hmcode_power_single(
        k,
        zs,
        Pk_lin,
        sigma_R,
        cosmo;
        T_AGN=nothing,
        Mmin=Mmin,
        Mmax=Mmax,
        nM=nM,
        tweaks=false,
        threaded=threaded,
    )

    Pk_feedback = hmcode_power_single(
        k,
        zs,
        Pk_lin,
        sigma_R,
        cosmo;
        T_AGN=T_AGN,
        Mmin=Mmin,
        Mmax=Mmax,
        nM=nM,
        tweaks=false,
        threaded=threaded,
    )

    return Pk_feedback ./ Pk_gravity
end

"""
Top-level HMcode power spectrum.
Returns `Pk[nk, nz]`.
"""
function hmcode_power(
    k::AbstractVector,
    zs::AbstractVector,
    Pk_lin::Function,
    sigma_R::Function,
    cosmo::HMcodeCosmology;
    T_AGN::Union{Nothing, Float64} = 10.0^7.8,
    Mmin::Float64 = 1e0,
    Mmax::Float64 = 1e18,
    nM::Int = 256,
    threaded::Bool = false,
)
    Pk_hm = hmcode_power_single(
        k,
        zs,
        Pk_lin,
        sigma_R,
        cosmo;
        T_AGN=nothing,
        Mmin=Mmin,
        Mmax=Mmax,
        nM=nM,
        tweaks=true,
        threaded=threaded,
    )

    if T_AGN !== nothing
        suppression = _get_feedback_suppression(
            k,
            zs,
            Pk_lin,
            sigma_R,
            cosmo,
            T_AGN;
            Mmin=Mmin,
            Mmax=Mmax,
            nM=nM,
            threaded=threaded,
        )
        Pk_hm .*= suppression
    end

    return Pk_hm
end

# power_spectrum.jl
using LinearAlgebra
using Base.Threads
using Polyester

struct SigmaREval{F}
    sigma_R::F
    z::Float64
end
@inline (s::SigmaREval)(R::Float64) = s.sigma_R(R, s.z)

struct PkLinEval{F}
    Pk_lin::F
    z::Float64
end
@inline (s::PkLinEval)(k::Float64) = s.Pk_lin(k, s.z)

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

mutable struct HMcodeWorkspace
    M::Vector{Float64}
    R::Vector{Float64}
    sigma_fast::Any
    Pk_fast::Any
    growth_itp::Any
    growth_LCDM_itp::Any
    params_tweaks::HMcodeParams
    params_notweaks::HMcodeParams
    Sigma::Matrix{Float64}
    nu_mat::Matrix{Float64}
    Pk_lin_mat::Matrix{Float64}
    Ptmp1::Matrix{Float64}
    Ptmp2::Matrix{Float64}
    W_buf::Vector{Matrix{Float64}}
    W2_buf::Vector{Matrix{Float64}}
    I1h_buf::Vector{Vector{Float64}}
    Pk1h_buf::Vector{Vector{Float64}}
    Pkwig_buf::Vector{Vector{Float64}}
    rv_eff_buf::Vector{Vector{Float64}}
end

function HMcodeWorkspace(
    k::AbstractVector,
    zs::AbstractVector,
    M_grid::AbstractVector,
    cosmo::HMcodeCosmology,
    sigma_R_interp,
    Pk_lin_interp;
    nthreads::Int=Threads.maxthreadid(),
    use_fast_interp::Bool=true,
    nR_fast::Int=500,
    nk_fast::Int=512,
    nz_fast::Int=200,
    logR_range::Tuple{Float64,Float64}=(log(1e-3), log(1e2)),
    logk_range::Tuple{Float64,Float64}=(log(1e-5), log(1e2)),
    z_max::Float64=4.0,
    z_interp::Symbol=:nearest,
)
    nk, nz, nM = length(k), length(zs), length(M_grid)
    M = collect(Float64.(M_grid))
    R = Lagrangian_radius(M, cosmo.Omega_m)
    sigma_fast = sigma_R_interp
    Pk_fast = Pk_lin_interp
    growth_itp = get_growth_interpolator(cosmo, LCDM=false)
    growth_LCDM_itp = get_growth_interpolator(cosmo, LCDM=true)
    Sigma = zeros(Float64, nM, nz)
    compute_sigma_grid!(Sigma, R, zs, sigma_fast)
    Pk_lin_mat = zeros(Float64, nk, nz)
    for iz in 1:nz, ik in 1:nk
        Pk_lin_mat[ik, iz] = Pk_fast(k[ik], zs[iz])
    end
    params_tweaks = compute_hmcode_params(k, zs, Pk_fast, sigma_fast, Sigma, R, cosmo, growth_itp; tweaks=true)
    params_notweaks = HMcodeParams(
        params_tweaks.R_nl, params_tweaks.n_eff, params_tweaks.C_curv, params_tweaks.sigma_v,
        params_tweaks.Delta_v, params_tweaks.delta_c,
        zeros(Float64, nz), ones(Float64, nz), zeros(Float64, nz),
        params_tweaks.k_star, fill(4.0, nz), zeros(Float64, nz)
    )
    nu_mat = zeros(Float64, nM, nz)
    for iz in 1:nz
        @views nu_mat[:, iz] .= params_tweaks.delta_c[iz] ./ Sigma[:, iz]
    end
    Ptmp1 = zeros(Float64, nk, nz)
    Ptmp2 = zeros(Float64, nk, nz)
    W_buf = [zeros(Float64, nM, nk) for _ in 1:nthreads]
    W2_buf = [zeros(Float64, nM, nk) for _ in 1:nthreads]
    I1h_buf = [zeros(Float64, nk) for _ in 1:nthreads]
    Pk1h_buf = [zeros(Float64, nk) for _ in 1:nthreads]
    Pkwig_buf = [zeros(Float64, nk) for _ in 1:nthreads]
    rv_eff_buf = [zeros(Float64, nM) for _ in 1:nthreads]
    return HMcodeWorkspace(M, R, sigma_fast, Pk_fast, growth_itp, growth_LCDM_itp, params_tweaks, params_notweaks, Sigma, nu_mat, Pk_lin_mat, Ptmp1, Ptmp2, W_buf, W2_buf, I1h_buf, Pk1h_buf, Pkwig_buf, rv_eff_buf)
end

function compute_sigma_grid!(Sigma::AbstractMatrix{Float64}, R_grid::AbstractVector, zs::AbstractVector, sigma_R)
    @inbounds for iz in eachindex(zs), iM in eachindex(R_grid)
        Sigma[iM, iz] = sigma_R(R_grid[iM], zs[iz])
    end
    return Sigma
end

function trapz_weights!(w::AbstractVector{Float64}, x::AbstractVector)
    n = length(x)
    w[1] = 0.5 * (x[2] - x[1])
    @inbounds for i in 2:n-1
        w[i] = 0.5 * (x[i+1] - x[i-1])
    end
    w[n] = 0.5 * (x[n] - x[n-1])
    return w
end

function mass_function_nu_st!(out::AbstractVector{Float64}, nu::AbstractVector, A::Float64, p::Float64, q::Float64)
    @inbounds for i in eachindex(nu)
        ν2 = nu[i] * nu[i]
        out[i] = A * (1.0 + (q * ν2)^(-p)) * exp(-q * ν2 / 2.0)
    end
    return out
end

function fill_weight_1h!(out::AbstractVector{Float64}, g_nu::AbstractVector, wt_nu::AbstractVector, M::AbstractVector, amp::AbstractVector)
    @inbounds for i in eachindex(out)
        a2 = amp[i] * amp[i]
        out[i] = (g_nu[i] / M[i]) * a2 * wt_nu[i]
    end
    return out
end

function compute_hmcode_params(
    k::AbstractVector,
    zs::AbstractVector,
    Pk_lin,
    sigma_R,
    sigma_grid::AbstractMatrix,
    R_grid::AbstractVector,
    cosmo::HMcodeCosmology,
    growth_itp;
    tweaks::Bool = true,
    kmin_sigmaV::Float64 = 1e-5,
)
    nz = length(zs)
    Om_m = cosmo.Omega_m
    f_nu = cosmo.Omega_nu / Om_m
    R_nl, n_eff, C_curv, sigma_v, Delta_v, delta_c = (zeros(Float64, nz) for _ in 1:6)
    eta, A, f_damp, k_star, B, k_damp = (zeros(Float64, nz) for _ in 1:6)
    @inbounds for iz in eachindex(zs)
        z = zs[iz]
        a = scalefactor_from_redshift(z)
        Om_mz = _Omega_m_a(a, cosmo, LCDM=false)
        g = growth_itp(a)
        G = get_accumulated_growth(a, growth_itp)
        dc = dc_Mead(a, Om_mz, f_nu, g, G)
        Dv = Dv_Mead(a, Om_mz, f_nu, g, G)
        delta_c[iz], Delta_v[iz] = dc, Dv
        sigmaM = @view sigma_grid[:, iz]
        Rnl = get_nonlinear_radius(R_grid[1], R_grid[end], dc, SigmaREval(sigma_R, z))
        sigma8 = sigma_R(8.0, z)
        sigmaV_val = sigmaV(0.0, PkLinEval(Pk_lin, z); kmin=kmin_sigmaV)
        neff = get_effective_index(Rnl, R_grid, sigmaM)
        R_nl[iz], sigma_v[iz], n_eff[iz] = Rnl, sigmaV_val, neff
        ks = 0.05618 * sigma8^(-1.013)
        k_star[iz] = ks
        if tweaks
            k_damp[iz] = 0.05699 * sigma8^(-1.089)
            f_damp[iz] = 0.2696 * sigma8^(0.9403)
            eta[iz] = 0.1281 * sigma8^(-0.3644)
            B[iz] = 5.196
            A[iz] = 1.875 * (1.603)^neff
        else
            B[iz], A[iz] = 4.0, 1.0
        end
    end
    return HMcodeParams(R_nl, n_eff, C_curv, sigma_v, Delta_v, delta_c, eta, A, f_damp, k_star, B, k_damp)
end

function fill_W_buf!(
    Wbuf::AbstractMatrix{Float64},
    k::AbstractVector,
    rv_eff_iz::AbstractVector,
    c_iz::AbstractVector,
    ln1pc_iz::AbstractVector;
    use_fast_specials::Bool = true,
)
    nk, nM = length(k), length(rv_eff_iz)
    if use_fast_specials
        win_NFW_fast!(Wbuf, k, rv_eff_iz, c_iz, ln1pc_iz)
    else
        for ik in 1:nk, iM in 1:nM
            Wbuf[iM, ik] = wnfw_xc(k[ik] * rv_eff_iz[iM], c_iz[iM])
        end
    end
    return Wbuf
end

function apply_baryonic_transform!(
    Wbuf::AbstractMatrix{Float64},
    M::AbstractVector,
    Mb::Float64,
    fstar::Float64,
    Om_m::Float64,
    Om_c::Float64,
    Om_b::Float64,
)
    nM, nk = size(Wbuf)
    @inbounds for iM in 1:nM
        fg = (Om_b / Om_m - fstar) * (M[iM] / Mb)^2 / (1.0 + (M[iM] / Mb)^2)
        coeff = Om_c / Om_m + fg
        for ik in 1:nk
            Wbuf[iM, ik] = coeff * Wbuf[iM, ik] + fstar
        end
    end
    return Wbuf
end

function _assemble_slice!(
    Pk_out::AbstractMatrix,
    iz::Int,
    k::AbstractVector,
    Pk_lin_mat::AbstractMatrix,
    hmpars::HMcodeParams,
    M::AbstractVector,
    rhom::Float64,
    cosmo::HMcodeCosmology,
    tweaks::Bool,
    T_AGN::Union{Nothing,Float64},
    nu_iz::AbstractVector,
    w1h_iz::AbstractVector,
    rv_iz::AbstractVector,
    rv_eff_tmp::AbstractVector,
    c_iz::AbstractVector,
    ln1pc_iz::AbstractVector,
    Mb::Float64,
    fstar::Float64,
    Om_m::Float64,
    Om_c::Float64,
    Om_b::Float64,
    Wbuf::AbstractMatrix{Float64},
    W2buf::AbstractMatrix{Float64},
    I1h::AbstractVector{Float64},
    Pk1h::AbstractVector{Float64},
    Pkwig::AbstractVector{Float64};
    use_fast_specials::Bool = true,
)
    nk, nM = length(k), length(M)
    η = hmpars.eta[iz]
    @inbounds for iM in 1:nM
        rv_eff_tmp[iM] = rv_iz[iM] * (nu_iz[iM]^η)
    end
    fill_W_buf!(Wbuf, k, rv_eff_tmp, c_iz, ln1pc_iz; use_fast_specials=use_fast_specials)
    if (T_AGN !== nothing) && (!tweaks)
        apply_baryonic_transform!(Wbuf, M, Mb, fstar, Om_m, Om_c, Om_b)
    end
    @inbounds for ik in 1:nk, iM in 1:nM
        wv = Wbuf[iM, ik]
        W2buf[iM, ik] = wv * wv
    end
    mul!(I1h, transpose(W2buf), w1h_iz)
    @. I1h = I1h * rhom
    ks = hmpars.k_star[iz]
    @inbounds for ik in 1:nk
        x = k[ik] / ks
        x4 = x * x; x4 *= x4
        Pk1h[ik] = x4 / (1.0 + x4) * I1h[ik]
    end
    if tweaks
        kd, f, alpha = hmpars.k_damp[iz], hmpars.f_damp[iz], hmpars.A[iz]
        omega_m, omega_b = cosmo.Omega_m * cosmo.h^2, cosmo.Omega_b * cosmo.h^2
        Pk_lin_k = @view Pk_lin_mat[:, iz]
        Pk_wig = get_Pk_wiggle(k, Pk_lin_k, cosmo.h, omega_m, omega_b, cosmo.n_s)
        @inbounds for ik in 1:nk
            damp_lin = 1.0 - exp(-(k[ik] * hmpars.sigma_v[iz])^2)
            Pk_dwl = Pk_lin_k[ik] - damp_lin * Pk_wig[ik]
            y = (k[ik] / kd)^ND_HMCODE
            Pk_2h = Pk_dwl * (1.0 - f * y / (1.0 + y))
            Pk_out[ik, iz] = (Pk_2h^alpha + Pk1h[ik]^alpha)^(1.0 / alpha)
        end
    else
        @inbounds for ik in 1:nk
            Pk_out[ik, iz] = Pk_lin_mat[ik, iz] + Pk1h[ik]
        end
    end
    return nothing
end

function hmcode_power_single!(
    Pk_out::AbstractMatrix{Float64},
    k::AbstractVector,
    zs::AbstractVector,
    cosmo::HMcodeCosmology,
    ws::HMcodeWorkspace;
    T_AGN::Union{Nothing, Float64} = nothing,
    tweaks::Bool = true,
    threaded::Bool = false,
    use_fast_specials::Bool = true,
    hmpars::HMcodeParams = tweaks ? ws.params_tweaks : ws.params_notweaks
)
    nk, nz, nM = length(k), length(zs), length(ws.M)
    Om_m, Om_b, Om_nu = cosmo.Omega_m, cosmo.Omega_b, cosmo.Omega_nu
    Om_c, f_nu = Om_m - Om_b - Om_nu, Om_nu / Om_m
    rhom = comoving_matter_density(Om_m)
    zc = 10.0; ac = scalefactor_from_redshift(zc)
    growth, growth_LCDM = ws.growth_itp, ws.growth_LCDM_itp
    feedback_params = (T_AGN === nothing) ? Dict{Symbol, Float64}() : get_feedback_parameters(T_AGN)
    rv, cc, ln1pc = Matrix{Float64}(undef, nM, nz), Matrix{Float64}(undef, nM, nz), Matrix{Float64}(undef, nM, nz)
    for iz in 1:nz
        z = zs[iz]; a = scalefactor_from_redshift(z)
        dc, Dv, B = hmpars.delta_c[iz], hmpars.Delta_v[iz], hmpars.B[iz]
        if (T_AGN !== nothing) && (!tweaks)
            B = feedback_params[:B0] * 10.0^(z * feedback_params[:Bz])
        end
        sigmaR_func_z = SigmaREval(ws.sigma_fast, z)
        zf = get_halo_collapse_redshifts(ws.M, z, dc, Om_m, growth, sigmaR_func_z)
        dolag = (growth(ac) / growth_LCDM(ac)) * (growth_LCDM(a) / growth(a))
        rv_scale = cbrt(Dv)
        for iM in 1:nM
            rv[iM, iz] = ws.R[iM] / rv_scale
            cc[iM, iz] = B * (1.0 + zf[iM]) / (1.0 + z) * dolag
            ln1pc[iM, iz] = log(1.0 + cc[iM, iz])
        end
    end
    p_st, q_st = 0.3, 0.707
    A_st = sqrt(2.0 * q_st) / (sqrt(pi) + gamma(0.5 - p_st) / 2.0^p_st)
    amp_no, amp_fb = ws.M .* (1.0 - f_nu) ./ rhom, ws.M ./ rhom
    w1h_mat = Matrix{Float64}(undef, nM, nz)
    for iz in 1:nz
        ncol = view(ws.nu_mat, :, iz)
        gcol = mass_function_nu_st!(zeros(nM), ncol, A_st, p_st, q_st)
        wtcol = trapz_weights!(zeros(nM), ncol)
        fill_weight_1h!(view(w1h_mat, :, iz), gcol, wtcol, ws.M, (T_AGN !== nothing && !tweaks) ? amp_fb : amp_no)
    end
    Mb_vec, fstar_vec = zeros(nz), zeros(nz)
    if (T_AGN !== nothing) && (!tweaks)
        for iz in 1:nz
            z = zs[iz]
            Mb_vec[iz] = feedback_params[:Mb0] * 10.0^(z * feedback_params[:Mbz])
            fstar_vec[iz] = feedback_params[:f0] * 10.0^(z * feedback_params[:fz])
        end
    end
    if threaded
        @batch for iz in 1:nz
            tid = threadid()
            _assemble_slice!(Pk_out, iz, k, ws.Pk_lin_mat, hmpars, ws.M, rhom, cosmo, tweaks, T_AGN, view(ws.nu_mat, :, iz), view(w1h_mat, :, iz), view(rv, :, iz), ws.rv_eff_buf[tid], view(cc, :, iz), view(ln1pc, :, iz), Mb_vec[iz], fstar_vec[iz], Om_m, Om_c, Om_b, ws.W_buf[tid], ws.W2_buf[tid], ws.I1h_buf[tid], ws.Pk1h_buf[tid], ws.Pkwig_buf[tid]; use_fast_specials=use_fast_specials)
        end
    else
        for iz in 1:nz
            _assemble_slice!(Pk_out, iz, k, ws.Pk_lin_mat, hmpars, ws.M, rhom, cosmo, tweaks, T_AGN, view(ws.nu_mat, :, iz), view(w1h_mat, :, iz), view(rv, :, iz), ws.rv_eff_buf[1], view(cc, :, iz), view(ln1pc, :, iz), Mb_vec[iz], fstar_vec[iz], Om_m, Om_c, Om_b, ws.W_buf[1], ws.W2_buf[1], ws.I1h_buf[1], ws.Pk1h_buf[1], ws.Pkwig_buf[1]; use_fast_specials=use_fast_specials)
        end
    end
    return Pk_out
end

function hmcode_power!(Pk_out, k, zs, Pk_lin, sigma_R, cosmo, ws; T_AGN=10^7.8, threaded=true, use_fast_specials=true)
    hmcode_power_single!(Pk_out, k, zs, cosmo, ws; T_AGN=nothing, tweaks=true, threaded=threaded, use_fast_specials=use_fast_specials)
    if T_AGN !== nothing
        hmcode_power_single!(ws.Ptmp1, k, zs, cosmo, ws; T_AGN=nothing, tweaks=false, threaded=threaded, use_fast_specials=use_fast_specials)
        hmcode_power_single!(ws.Ptmp2, k, zs, cosmo, ws; T_AGN=T_AGN, tweaks=false, threaded=threaded, use_fast_specials=use_fast_specials)
        @. Pk_out = Pk_out * (ws.Ptmp2 / ws.Ptmp1)
    end
    return Pk_out
end

function hmcode_power(k, zs, Pk_lin, sigma_R, cosmo; T_AGN=10^7.8, Mmin=1e0, Mmax=1e18, nM=256, threaded=false, use_fast_specials=false)
    M = exp.(range(log(Mmin), log(Mmax), length=nM))
    ws = HMcodeWorkspace(k, zs, M, cosmo, sigma_R, Pk_lin; nthreads=Threads.maxthreadid())
    Pk_out = zeros(length(k), length(zs))
    hmcode_power!(Pk_out, k, zs, Pk_lin, sigma_R, cosmo, ws; T_AGN=T_AGN, threaded=threaded, use_fast_specials=use_fast_specials)
    return Pk_out
end

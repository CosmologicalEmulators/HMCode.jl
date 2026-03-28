# power_spectrum.jl

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
)
    is_array_monotonic(-collect(Float64.(zs))) || throw(ArgumentError("Redshifts must be monotonically decreasing"))
    is_array_monotonic(collect(Float64.(k))) || throw(ArgumentError("k must be monotonically increasing"))

    M = 10.0 .^ range(log10(Mmin), log10(Mmax), length=nM)

    zc = 10.0
    ac = scalefactor_from_redshift(zc)

    Om_m = cosmo.Omega_m
    Om_b = cosmo.Omega_b
    Om_nu = cosmo.Omega_nu
    Om_c = Om_m - Om_b - Om_nu
    f_nu = Om_nu / Om_m

    growth = get_growth_interpolator(cosmo, LCDM=false)
    growth_LCDM = get_growth_interpolator(cosmo, LCDM=true)

    feedback_params = (T_AGN === nothing) ? Dict{Symbol, Float64}() : get_feedback_parameters(T_AGN)

    nk = length(k)
    nz = length(zs)
    Pk_out = zeros(Float64, nk, nz)

    for (iz, z) in enumerate(zs)
        a = scalefactor_from_redshift(z)

        # linear inputs at this redshift
        Pk_lin_k = [Pk_lin(kk, z) for kk in k]

        # Spherical-collapse quantities
        Om_mz = _Omega_m_a(a, cosmo, LCDM=false)
        g = growth(a)
        G = get_accumulated_growth(a, growth)
        dc = dc_Mead(a, Om_mz, f_nu, g, G)
        Dv = Dv_Mead(a, Om_mz, f_nu, g, G)

        # Halo model initialisation
        hmod = HaloModel(z, Om_m; name="Sheth & Tormen (1999)", Dv=Dv, dc=dc)
        R = Lagrangian_radius(hmod, M)
        sigmaM = [sigma_R(r, z) for r in R]
        nu = _peak_height(hmod, M, sigmaM)

        # Linear-spectrum-derived HMcode quantities
        sigmaR_func_z = r -> sigma_R(r, z)
        Rnl = get_nonlinear_radius(R[1], R[end], dc, sigmaR_func_z)
        sigma8 = sigma_R(8.0, z)
        sigmaV_val = sigmaV(0.0, kk -> Pk_lin(kk, z); kmin=kmin_sigmaV)
        neff = get_effective_index(Rnl, R, sigmaM)

        # HMcode-2020 parameters (Table 2 Mead et al. 2021)
        ks = 0.05618 * sigma8^(-1.013)
        if tweaks
            kd = 0.05699 * sigma8^(-1.089)
            f = 0.2696 * sigma8^(0.9403)
            nd = 2.853
            eta = 0.1281 * sigma8^(-0.3644)
            B = 5.196
            alpha = 1.875 * (1.603)^neff
        else
            eta = 0.0
            B = 4.0
            kd, f, nd, alpha = 0.0, 0.0, 0.0, 1.0
        end

        Mb, fstar = 0.0, 0.0
        if (T_AGN !== nothing) && (!tweaks)
            B = feedback_params[:B0] * 10.0^(z * feedback_params[:Bz])
            Mb = feedback_params[:Mb0] * 10.0^(z * feedback_params[:Mbz])
            fstar = feedback_params[:f0] * 10.0^(z * feedback_params[:fz])
        end

        # Concentration relation + Dolag correction
        zf = get_halo_collapse_redshifts(M, z, dc, Om_m, growth, sigmaR_func_z)
        c = @. B * (1.0 + zf) / (1.0 + z)
        c .*= (growth(ac) / growth_LCDM(ac)) * (growth_LCDM(a) / growth(a))

        # Build matter profile
        rv = virial_radius(hmod, M)
        Uk = zeros(Float64, nk, nM)
        @inbounds for iM in eachindex(M)
            k_eff = k .* (nu[iM]^eta)
            if (T_AGN !== nothing) && (!tweaks)
                Uk[:, iM] .= win_NFW_baryons(k_eff, rv[iM], c[iM], M[iM], Mb, fstar, Om_m, Om_c, Om_b)
            else
                Uk[:, iM] .= win_NFW(k_eff, rv[iM], c[iM])
            end
        end

        if (T_AGN !== nothing) && (!tweaks)
            profile_m = Fourier(k, M, Uk; amplitude=M ./ hmod.rhom, mass_tracer=true)
        else
            profile_m = Fourier(k, M, Uk; amplitude=M .* (1.0 - f_nu) ./ hmod.rhom, mass_tracer=true)
        end

        _, Pk_1h_dict, _ = power_spectrum(
            hmod,
            k,
            Pk_lin_k,
            M,
            sigmaM,
            Dict("m" => profile_m);
            simple_twohalo=true,
        )
        Pk_1h_raw = Pk_1h_dict["m-m"]

        # One-halo damping, Eq. (17)
        Pk_1h = @. (k / ks)^4 / (1.0 + (k / ks)^4) * Pk_1h_raw

        # Two-halo + transition
        if tweaks
            omega_m = cosmo.Omega_m * cosmo.h^2
            omega_b = cosmo.Omega_b * cosmo.h^2
            Pk_wig = get_Pk_wiggle(k, Pk_lin_k, cosmo.h, omega_m, omega_b, cosmo.n_s)
            Pk_dwl = @. Pk_lin_k - (1.0 - exp(-(k * sigmaV_val)^2)) * Pk_wig
            Pk_2h = @. Pk_dwl * (1.0 - f * (k / kd)^nd / (1.0 + (k / kd)^nd))
            Pk_hm = @. (Pk_2h^alpha + Pk_1h^alpha)^(1.0 / alpha)
        else
            Pk_hm = @. Pk_lin_k + Pk_1h
        end

        Pk_out[:, iz] .= Pk_hm
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
        )
        Pk_hm .*= suppression
    end

    return Pk_hm
end

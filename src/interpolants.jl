# interpolants.jl
"""
Fast 2D interpolation utilities built on DataInterpolations.jl.
"""

import DataInterpolations
const DI = DataInterpolations

struct Fast2DInterp{I}
    slices::Vector{I}
    z_knots::Vector{Float64}
    inv_dz::Float64
    nz::Int
    z_interp::Symbol   # :nearest (default) or :linear
    log_output::Bool
end

function Fast2DInterp(
    f_original,
    logx_grid::AbstractVector{Float64},
    z_grid::AbstractVector{Float64};
    z_interp::Symbol = :nearest,
    log_output::Bool = true,
)
    nx = length(logx_grid)
    nz = length(z_grid)
    x_grid = exp.(logx_grid)

    f_vals = Matrix{Float64}(undef, nx, nz)
    @inbounds for iz in 1:nz
        z = z_grid[iz]
        for ix in 1:nx
            v = f_original(x_grid[ix], z)
            f_vals[ix, iz] = log_output ? log(max(v, 1e-300)) : v
        end
    end

    first_itp = DI.LinearInterpolation(
        collect(@view(f_vals[:, 1])),
        collect(logx_grid);
        extrapolation=DI.ExtrapolationType.Linear,
    )
    I = typeof(first_itp)
    slices = Vector{I}(undef, nz)
    slices[1] = first_itp
    @inbounds for iz in 2:nz
        slices[iz] = DI.LinearInterpolation(
            collect(@view(f_vals[:, iz])),
            collect(logx_grid);
            extrapolation=DI.ExtrapolationType.Linear,
        )
    end

    dz = nz > 1 ? z_grid[2] - z_grid[1] : 1.0
    return Fast2DInterp{I}(slices, collect(z_grid), 1.0 / dz, nz, z_interp, log_output)
end

@inline function _nearest_z_index(itp::Fast2DInterp, z::Float64)
    j = Int(round((z - itp.z_knots[1]) * itp.inv_dz)) + 1
    return clamp(j, 1, itp.nz)
end

@inline function _linear_z_bracket(itp::Fast2DInterp, z::Float64)
    zc = clamp(z, itp.z_knots[1], itp.z_knots[end])
    fj = (zc - itp.z_knots[1]) * itp.inv_dz
    j = clamp(Int(floor(fj)), 1, itp.nz - 1)
    α = fj - floor(fj)
    return j, α
end

@inline function (itp::Fast2DInterp)(x::Float64, z::Float64)::Float64
    logx = log(x)
    if itp.z_interp === :nearest || itp.nz == 1
        j = _nearest_z_index(itp, z)
        v = itp.slices[j](logx)
        return itp.log_output ? exp(v) : v
    else
        j, α = _linear_z_bracket(itp, z)
        v0 = itp.slices[j](logx)
        v1 = itp.slices[j + 1](logx)
        v = v0 + α * (v1 - v0)
        return itp.log_output ? exp(v) : v
    end
end

@inline (itp::Fast2DInterp)(x::Real, z::Real) = itp(Float64(x), Float64(z))

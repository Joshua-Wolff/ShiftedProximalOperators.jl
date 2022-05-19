export ShiftedNormL0BInf

mutable struct ShiftedNormL0BInf{
  R <: Real,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3 <: Union{R, AbstractVector{R}},
  V4 <: Union{R, AbstractVector{R}}
} <: ShiftedProximableFunction
  h::NormL0{R}
  xk::V0
  sj::V1
  sol::V2
  l::V3
  u::V4
  shifted_twice::Bool

  function ShiftedNormL0BInf(
    h::NormL0{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    l::Union{R, AbstractVector{R}},
    u::Union{R, AbstractVector{R}},
    shifted_twice::Bool
  ) where {R <: Real}
    sol = similar(xk)
    if sum(l .> u) > 0
      error("Error on the trust region bounds, at least one lower bound is greater than the upper bound.")
    else
      new{R, typeof(xk), typeof(sj), typeof(sol), typeof(l), typeof(u)}(h, xk, sj, sol, l, u, shifted_twice)
    end
  end

end

# We cannot use this function anymore with [l,u] trust region 
#=
function (ψ::ShiftedNormL0BInf)(y)
  return ψ.h(ψ.xk + ψ.sj + y) + IndGenTR(ψ.l, ψ.u)(ψ.sj + y)
end 
=#

shifted(h::NormL0{R}, xk::AbstractVector{R}, l::Union{R, AbstractVector{R}}, u::Union{R, AbstractVector{R}}) where {R <: Real} =
  ShiftedNormL0BInf(h, xk, zero(xk), l, u, false)

shifted(
  ψ::ShiftedNormL0BInf{R, V0, V1, V2, V3, V4},
  sj::AbstractVector{R},
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3 <: Union{R, AbstractVector{R}}, V4 <: Union{R, AbstractVector{R}}} =
  ShiftedNormL0BInf(ψ.h, ψ.xk, sj, ψ.l, ψ.u, true)

fun_name(ψ::ShiftedNormL0BInf) = "shifted L0 pseudo-norm with generalized trust region indicator"
fun_expr(ψ::ShiftedNormL0BInf) = "t ↦ λ ‖xk + sj + t‖₀ + χ({sj + t .∈ [l,u]})"
fun_params(ψ::ShiftedNormL0BInf) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "lb = $(ψ.l)\n" * " "^14 * "ub = $(ψ.u)"


function prox!(
  y::AbstractVector{R},
  ψ::ShiftedNormL0BInf{R, V0, V1, V2, V3, V4},
  q::AbstractVector{R},
  σ::R,
) where {R <: Real, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3 <: Union{R, AbstractVector{R}}, V4 <: Union{R, AbstractVector{R}}}
  
  c2 = 2 * ψ.λ * σ

  for i ∈ eachindex(q)

    if isa(ψ.l, Real) & isa(ψ.u, Real)
      li = ψ.l
      ui = ψ.u
    elseif isa(ψ.l, Real)
      li = ψ.l
      ui = ψ.u[i]
    elseif isa(ψ.u, Real) 
      li = ψ.l[i]
      ui = ψ.u
    else
      li = ψ.l[i]
      ui = ψ.u[i]
    end 

    qi = q[i]
    xs = ψ.xk[i] + ψ.sj[i]
  
    if ui < qi
      if li <= -xs <= ui
        if (xs + qi)^2 < (ui - qi)^2 + c2
          y[i] = -xs
        else 
          y[i] = ui
        end
      else
        y[i] = ui
      end
  
    elseif li > qi
      if li <= -xs <= ui
        if (xs + qi)^2 < (li - qi)^2 + c2
          y[i] = -xs
        else 
          y[i] = li
        end
      else
        y[i] = li
      end
  
    else 
      if li <= -xs <= ui
        if (xs + qi)^2 < c2
          y[i] = -xs
        else 
          y[i] = qi
        end
      else
        y[i] = qi
      end
    end
  end
  return y
end


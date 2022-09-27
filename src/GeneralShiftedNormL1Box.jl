export GeneralShiftedNormL1Box

mutable struct GeneralShiftedNormL1Box{
  R <: Real,
  T <: Integer,
  V0 <: AbstractVector{R},
  V1 <: AbstractVector{R},
  V2 <: AbstractVector{R},
  V3,
  V4
} <: ShiftedProximableFunction
  h::NormL1{R}
  xk::V0
  sj::V1
  sol::V2
  l::V3
  u::V4
  Δ::R
  shifted_twice::Bool
  selected::UnitRange{T}

  function GeneralShiftedNormL1Box(
    h::NormL1{R},
    xk::AbstractVector{R},
    sj::AbstractVector{R},
    l,
    u,
    Δ::R,
    shifted_twice::Bool,
    selected::UnitRange{T}
  ) where {R <: Real, T <: Integer}
    sol = similar(xk)
    if any(l .> u)
      error("Error: at least one lower bound is greater than the upper bound.")
    end
    new{R, T, typeof(xk), typeof(sj), typeof(sol), typeof(l), typeof(u)}(h, xk, sj, sol, l, u, Δ, shifted_twice, selected)
  end

end

shifted(h::NormL1{R}, xk::AbstractVector{R}, l, u, Δ::R) where {R <: Real} =
  GeneralShiftedNormL1Box(h, xk, zero(xk), l, u, Δ, false, 1:length(xk))
shifted(h::NormL1{R}, xk::AbstractVector{R}, l, u, Δ::R, selected::UnitRange{T}) where {R <: Real, T <: Integer} =
  GeneralShiftedNormL1Box(h, xk, zero(xk), l, u, Δ, false, selected)
shifted(
  ψ::GeneralShiftedNormL1Box{R, T, V0, V1, V2, V3, V4},
  sj::AbstractVector{R},
) where {R <: Real, T <: Integer, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3, V4} =
  GeneralShiftedNormL1Box(ψ.h, ψ.xk, sj, ψ.l, ψ.u, ψ.Δ, true, ψ.selected)

(ψ::GeneralShiftedNormL1Box)(y) = ψ.h((ψ.xk + ψ.sj + y)[ψ.selected]) + (all(ψ.l .<= ψ.sj + y .<= ψ.u .|| isapprox.(ψ.l, ψ.sj + y, atol = eps()) .|| isapprox.(ψ.u, ψ.sj + y, atol = eps())) ? 0 : Inf)

fun_name(ψ::GeneralShiftedNormL1Box) = "shifted L0 pseudo-norm with box indicator"
fun_expr(ψ::GeneralShiftedNormL1Box) = "t ↦ λ ‖xk + sj + t‖₀ + χ({sj + t .∈ [l,u]})"
fun_params(ψ::GeneralShiftedNormL1Box) =
  "xk = $(ψ.xk)\n" * " "^14 * "sj = $(ψ.sj)\n" * " "^14 * "lb = $(ψ.l)\n" * " "^14 * "ub = $(ψ.u)"

function prox!(
  y::AbstractVector{R},
  ψ::GeneralShiftedNormL1Box{R, T, V0, V1, V2, V3, V4},
  q::AbstractVector{R},
  D::LO,
) where {R <: Real, T <: Integer, V0 <: AbstractVector{R}, V1 <: AbstractVector{R}, V2 <: AbstractVector{R}, V3, V4, LO}
  
  selected = ψ.selected  

  for i ∈ eachindex(q)

    li = isa(ψ.l, Real) ? ψ.l : ψ.l[i]
    ui = isa(ψ.u, Real) ? ψ.u : ψ.u[i]
    qi = q[i]
    xi = ψ.xk[i]
    si = ψ.sj[i]
    δ =  isa(D.d, Number) ? D.d : D.d[i]

    if i ∈ selected
      set = [-xi-si, li-si, ui-si]
      if δ > 0
        if qi+ψ.λ/δ < -xi-si
          append!(set,qi+ψ.λ/δ)
        end
        if qi-ψ.λ/δ > -xi-si
          append!(set,qi-ψ.λ/δ)
        end
      end 
      set = set[li-si .<= set .<= ui-si .|| map(t -> isapprox(li-si, t, atol=eps(typeof(li))), set) .||  map(t -> isapprox(ui-si, t, atol=eps(typeof(ui))), set)]
      obj = map(t -> 1/2 * δ * (t - qi)^2 + ψ.h(xi + si + t), set)
      y[i] = set[argmin(obj)]
    else
      set = [qi, li-si, ui-si]
      set = set[li-si .<= set .<= ui-si .|| map(t -> isapprox(li-si, t, atol=eps(typeof(li))), set) .||  map(t -> isapprox(ui-si, t, atol=eps(typeof(ui))), set)]
      obj = map(t -> 1/2 * δ * (t - qi)^2, set)
      y[i] = set[argmin(obj)]
    end 

  end
  return y
end

using LinearAlgebra
using ConicQKD
using JuMP
using Ket
import MosekTools
import Hypatia
import Hypatia.Cones
import Dualization
import Optim

function gkraus()
    return [kron(Matrix(I, 2, 3), Matrix(I, 2, 3))]
end

function zkraus()
    return [kron(proj(1, 2), I(2)), kron(proj(2, 2), I(2))]
end

coherent(α::Number, Nc::Integer) = exp(-abs2(α) / 2) * [α^n / sqrt(factorial(n)) for n ∈ 0:Nc]

#p_z should 2/3
function composite_state(p_z::T, η::T, α::Union{T,Complex{T}}) where {T<:Real}
    Nc = 10
    ψAC =
        √(p_z / 2) * (kron(ket(1, 3), coherent(α, Nc)) + kron(ket(2, 3), coherent(-α, Nc))) +
        √(1 - p_z) * kron(ket(3, 3), ket(1, Nc + 1))
    ψACBC = kron(ψAC, ψAC)
    ψABCC = permute_systems(ψACBC, [1, 3, 2, 4], [3, Nc + 1, 3, Nc + 1])
    return ψAC
end

function coherent_inner_product(α::Number, β::Number)
    return exp(-abs2(α)/2 - abs2(β)/2 + conj(α)*β)
end

function reduced_state(α, δ_SPF = 0)
    SPF_phase_factor = exp(im * (δ_SPF))
    β = δ_SPF == 0 ? -α : -α * SPF_phase_factor
    ρA = fill(one(β) / 3, 3, 3)
    ρA[1, 2] *= coherent_inner_product(β, α)
    ρA[1, 3] *= exp(-abs2(α) / 2)
    ρA[2, 3] *= exp(-abs2(α) / 2)
    return Hermitian(ρA)
end

function probability_charlie(αA, αB, η, pd, constructive::Bool)
    s = constructive ? 1 : -1
    return (1 - exp(-(η / 2) * abs2(αA + s * αB)) * (1 - pd)) * exp(-(η / 2) * abs2(αA - s * αB)) * (1 - pd)
end 

function target_probabilities(α, η, pd, δ_mis, δ_SPF)
    SPF_phase_factor = exp(im * (δ_SPF))
    settingsA = (α, -α * SPF_phase_factor , zero(α))
    rotated = α * exp(im * (δ_mis))
    settingsB = (rotated, -rotated * SPF_phase_factor, zero(rotated))
    constructive = [probability_charlie(αA, αB, η, pd, true) for αA ∈ settingsA, αB ∈ settingsB]
    destructive = [probability_charlie(αA, αB, η, pd, false) for αA ∈ settingsA, αB ∈ settingsB]
    return [constructive, destructive]
end

function total_rate_mdi_optimized(ϵ::T, δ_SPF::T, D::AbstractVector, α0 = T(0.43); eph = false) where {T<:Real}
    α0 = [α0]
    sol_vector = NaN*ones(T, length(D))
    for i in eachindex(D)
        f(α) = -total_rate_mdi(α[1], ϵ, δ_SPF, D[i]; eph)
        sol = Optim.optimize(f, α0)
        α0 = sol.minimizer
        display(α0[1])
        sol_vector[i] = -sol.minimum
    end
    return sol_vector
end

function total_rate_mdi(α::T, ϵ::T, δ_SPF::T, D; eph = false) where {T<:Real}
    attenuation = T(2) / 10
    η = 10^(-attenuation * D / 20)
    f = T(116) / 100
    pd = inv(T(10^6))
    δ_mis = T(0)
    if eph
        K = hae_mdi_eph(α, η, ϵ, pd, δ_mis, δ_SPF) - f * hab_mdi(α, η, pd, δ_mis, δ_SPF)
    else
        correction = T(9) / 4 #1/p_z^2 for p_z = 2/3
        K = correction * hae_mdi(α, η, ϵ, pd, δ_mis, δ_SPF) - f * hab_mdi(α, η, pd, δ_mis, δ_SPF)
    end
    return K
end

function hab_mdi(α, η, pd)
    bit_error_rate = pd / (2pd + (1 - pd) * (1 - exp(-2η * abs2(α))))
    detection_probability = (1-pd)*(2pd + (1 - pd) * (1 - exp(-2η * abs2(α))))
    return detection_probability * binary_entropy(bit_error_rate)
end

function hab_mdi(α, η, pd, δ_mis, δ_SPF)
    Y = target_probabilities(α, η, pd, δ_mis, δ_SPF)
    # Y[1] is constructive, Y[2] is destructive
    # For Z basis: indices (1,1), (1,2), (2,1), (2,2) correspond to 0Z0Z, 0Z1Z, 1Z0Z, 1Z1Z
    Y_Z = sum(sum(Y[γ][1:2, 1:2]) for γ ∈ 1:2)/4

    Y_Z_err = (Y[1][1, 2] + Y[1][2, 1] + Y[2][1, 1] + Y[2][2, 2])/4

    e_Z = Y_Z_err / Y_Z
    hab = Y_Z * binary_entropy(e_Z)
    return hab
end

function hae_mdi(α::T, η::T, ϵ::T, pd::T, δ_mis::T, δ_SPF::T) where {T<:Real}

    is_complex = δ_SPF != 0
    is_complex ? R = Complex{T} : R = T
    psd_cone, wrapper, hermitian_space = Ket._sdp_parameters(is_complex)

    model = GenericModel{T}(() -> Hypatia.Optimizer{T}(; verbose = true))

    dim_ρAB = 9
    ρA_target = reduced_state(α, δ_SPF)
    ρAB_target = wrapper(kron(ρA_target, ρA_target))
    fidelities = 9conj(ρAB_target)
    if ϵ == 0
        ρvec = [@variable(model, [1:dim_ρAB, 1:dim_ρAB] ∈ psd_cone) for _ ∈ 1:2]
        ρvec3 = ρAB_target - sum(ρvec)
        @constraint(model, ρvec3 ∈ psd_cone)
    else
        ρvec = [@variable(model, [1:dim_ρAB, 1:dim_ρAB] ∈ psd_cone) for _ ∈ 1:2]
        d_gram = 18
        @variable(model, gram[1:d_gram, 1:d_gram] ∈ psd_cone)
        @constraint(model, diag(gram) .== 1)
        @constraint(model, diag(gram, 9) .== 0)
        for j ∈ 1:9, i ∈ 1:j-1
            @constraint(model, gram[i, j] == fidelities[i, j])
        end
        ρAB = Matrix{typeof(1 * ρvec[1][1])}(undef, dim_ρAB, dim_ρAB)
        for i ∈ 1:dim_ρAB
            ρAB[i, i] = T(1) / 9
        end
        bigϵ = 2ϵ - ϵ^2
        for j ∈ 1:9, i ∈ 1:j-1
            ρAB[j, i] = 
                ((1 - bigϵ) / 9) * fidelities[i, j] +
                (√(bigϵ * (1 - bigϵ)) / 9) * (gram[i, j+9] + gram[i+9, j]) +
                (bigϵ / 9) * gram[i+9, j+9]
        end
        ρvec3 = wrapper(ρAB, :L) - sum(ρvec)
        @constraint(model, ρvec3 ∈ psd_cone)
    end

    Y = target_probabilities(α, η, pd, δ_mis, δ_SPF)
    #the factor 9 is to transform joint probabilities in conditional ones,
    #assuming each state is prepared with probability 1/3
    for i ∈ 1:3, j ∈ 1:3, γ ∈ 1:2
        ind = 3*(i-1) + j
        @constraint(model, 9*ρvec[γ][ind, ind] == Y[γ][i, j])
    end

    Ghat = gkraus()
    Z = zkraus()
    Zhat = vec([Zi * Gi for Gi ∈ Ghat, Zi ∈ Z])

    vec_dim = Cones.svec_length(R, dim_ρAB)
    @variable(model, h1)
    @variable(model, h2)
    @objective(model, Min, (h1 + h2) / log(T(2)))
    @constraint(model, [h1; svec(ρvec[1])] in EpiQKDTriCone{T,R}(Ghat, Zhat, 1 + vec_dim))
    @constraint(model, [h2; svec(ρvec[2])] in EpiQKDTriCone{T,R}(Ghat, Zhat, 1 + vec_dim))

    optimize!(model)
    return objective_value(model)
end

function hae_mdi_eph(α::T, η::T, ϵ::T = T(0), pd::T = T(0), δ_mis::T = T(0), δ_SPF::T = T(0)) where {T<:Real}
    # This function computes an bound on hae by first computing 
    # a bound on the phase error rate eph using SDP

    # First, compute the phase error rate upper bound using SDP
    Y_Z_ph = phase_error_rate_sdp_gram(α, η, 2 * ϵ - ϵ^2, pd, δ_mis, δ_SPF)

    # Compute Y_Z (the total detection probability for Z basis states)
    Y = target_probabilities(α, η, pd, δ_mis, δ_SPF)
    # Y[1] is constructive, Y[2] is destructive
    # For Z basis: indices (1,1), (1,2), (2,1), (2,2) correspond to 0Z0Z, 0Z1Z, 1Z0Z, 1Z1Z
    Y_Z = (Y[1][1, 1] + Y[2][1, 1] + Y[1][1, 2] + Y[2][1, 2] + Y[1][2, 1] + Y[2][2, 1] + Y[1][2, 2] + Y[2][2, 2]) / 4

    # Compute the phase error rate
    ephA = Y_Z_ph / Y_Z
    # Compute hae using binary entropy
    if ephA > 0.5
        hae = T(0)
    else
        hae = 1 - binary_entropy(ephA)
    end

    return Y_Z * hae
end

function phase_error_rate_sdp_gram(α::T, η::T, ξ::T = T(0), pd::T = T(0), δ_mis::T = T(0), δ_SPF::T = T(0)) where {T<:Real}
    
    is_complex = δ_SPF != 0
    is_complex ? R = Complex{T} : R = T
    psd_cone, wrapper, hermitian_space = Ket._sdp_parameters(is_complex)

    model = GenericModel{T}()
    if T <: Float64
        solver = MosekTools.Mosek.Optimizer
        dualize = false
    else
        solver = Hypatia.Optimizer{T}
        dualize = true
    end

    if dualize
        set_optimizer(model, Dualization.dual_optimizer(solver; coefficient_type = T))
    else
        set_optimizer(model, solver)
    end

    # We have 9 joint states for MDI (3 Alice states × 3 Bob states)
    n = 9

    # Create Gram matrix G with appropriate dimensions
    if ξ == 0
        dim_G = 3 * n  # M_Dc, M_Dd and M_fail components
    else
        dim_G = 6 * n  # Double the dimension to account for perpendicular states
    end

    @variable(model, G[1:dim_G, 1:dim_G] in psd_cone)

    # Get inner products from reduced_state
    ρA = reduced_state(α, δ_SPF)
    # Extract inner products from ρA
    inn_prod_mat = n * kron(ρA, ρA)

    # Get observed yields
    Y = target_probabilities(α, η, pd, δ_mis, δ_SPF)

    # Define Gprod function for ξ ≠ 0 case
    function Gprod(i::Int, j::Int)
        if ξ == 0
            return G[i, j]
        else
            return (1 - ξ) * G[i, j] + sqrt((1 - ξ) * ξ) * (G[i, j+n] + G[i+n, j]) + ξ * G[i+n, j+n]
        end
    end

    # Constraints: First type - observed yields
    if ξ == 0
        # Direct constraints for ξ = 0
        idx = 0
        for γ ∈ 1:2, i ∈ 1:3, j ∈ 1:3
            idx += 1
            @constraint(model, G[idx, idx] == Y[γ][i, j])
        end
    else
        # Constraints using Gprod for ξ ≠ 0
        # For M_Dc (indices 1 to n)
        idx = 0
        for i ∈ 1:3, j ∈ 1:3
            idx += 1
            @constraint(model, Gprod(idx, idx) == Y[1][i, j])
        end

        # For M_Dd (indices 2n+1 to 3n)
        idx = 0
        for i ∈ 1:3, j ∈ 1:3
            idx += 1
            @constraint(model, Gprod(2 * n + idx, 2 * n + idx) == Y[2][i, j])
        end
    end

    # Constraints: Second type - completeness relation
    if ξ == 0
        for i ∈ 1:n, j ∈ 1:i
            # Sum over M_Dc, M_Dd and M_fail
            @constraint(model, G[i, j] + G[i+n, j+n] + G[i+2*n, j+2*n] == inn_prod_mat[j, i])
        end
    else
        for i ∈ 1:n, j ∈ 1:i
            # Sum over M_Dc, M_Dd and M_fail
            @constraint(model, G[i, j] + G[i+2*n, j+2*n] + G[i+4*n, j+4*n] == inn_prod_mat[j, i])
        end

        # Additional constraints for ξ ≠ 0
        # Third type - orthogonality
        for i ∈ 1:n
            @constraint(model, G[i, i+n] + G[i+2*n, i+3*n] + G[i+4*n, i+5*n] == 0)
        end

        # Fourth type - normalization of perpendicular components
        for i ∈ 1:n
            @constraint(model, G[i+n, i+n] + G[i+3*n, i+3*n] + G[i+5*n, i+5*n] == 1)
        end
    end

    # Define indices for states
    # Mapping: (i,j) -> (i-1)*3 + j
    idx_0Z0Z = 1  # (1,1)
    idx_0Z1Z = 2  # (1,2)
    idx_1Z0Z = 4  # (2,1)
    idx_1Z1Z = 5  # (2,2)

    # Objective function
    if ξ == 0
        # Direct objective for ξ = 0
        obj_Dc = (
            G[idx_0Z0Z, idx_0Z0Z] +
            G[idx_0Z0Z, idx_1Z1Z] +
            G[idx_0Z1Z, idx_0Z1Z] +
            G[idx_0Z1Z, idx_1Z0Z] +
            G[idx_1Z0Z, idx_0Z1Z] +
            G[idx_1Z0Z, idx_1Z0Z] +
            G[idx_1Z1Z, idx_0Z0Z] +
            G[idx_1Z1Z, idx_1Z1Z]
        )

        obj_Dd = (
            G[idx_0Z0Z+n, idx_0Z0Z+n] +
            G[idx_0Z0Z+n, idx_1Z1Z+n] +
            G[idx_0Z1Z+n, idx_0Z1Z+n] +
            G[idx_0Z1Z+n, idx_1Z0Z+n] +
            G[idx_1Z0Z+n, idx_0Z1Z+n] +
            G[idx_1Z0Z+n, idx_1Z0Z+n] +
            G[idx_1Z1Z+n, idx_0Z0Z+n] +
            G[idx_1Z1Z+n, idx_1Z1Z+n]
        )
    else
        # Objective using Gprod for ξ ≠ 0
        obj_Dc = (
            Gprod(idx_0Z0Z, idx_0Z0Z) +
            Gprod(idx_0Z0Z, idx_1Z1Z) +
            Gprod(idx_0Z1Z, idx_0Z1Z) +
            Gprod(idx_0Z1Z, idx_1Z0Z) +
            Gprod(idx_1Z0Z, idx_0Z1Z) +
            Gprod(idx_1Z0Z, idx_1Z0Z) +
            Gprod(idx_1Z1Z, idx_0Z0Z) +
            Gprod(idx_1Z1Z, idx_1Z1Z)
        )

        obj_Dd = (
            Gprod(idx_0Z0Z + 2 * n, idx_0Z0Z + 2 * n) +
            Gprod(idx_0Z0Z + 2 * n, idx_1Z1Z + 2 * n) +
            Gprod(idx_0Z1Z + 2 * n, idx_0Z1Z + 2 * n) +
            Gprod(idx_0Z1Z + 2 * n, idx_1Z0Z + 2 * n) +
            Gprod(idx_1Z0Z + 2 * n, idx_0Z1Z + 2 * n) +
            Gprod(idx_1Z0Z + 2 * n, idx_1Z0Z + 2 * n) +
            Gprod(idx_1Z1Z + 2 * n, idx_0Z0Z + 2 * n) +
            Gprod(idx_1Z1Z + 2 * n, idx_1Z1Z + 2 * n)
        )
    end

    obj =real((obj_Dc + obj_Dd) / 8)

    @objective(model, Max, obj)

    optimize!(model)
    return value(obj)
end

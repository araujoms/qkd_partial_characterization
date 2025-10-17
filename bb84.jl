import Hypatia
using JuMP
using LinearAlgebra
using Ket
using ConicQKD
using MosekTools

function zkraus(p_z)
    if p_z == 1 || p_z == 0
        Π = [kron(proj(1, 2), I(2)), kron(proj(2, 2), I(2))]
    else
        Π0 = proj(1, 4) + proj(3, 4)
        Π1 = proj(2, 4) + proj(4, 4)
        Π = [kron(Π0, I(2)), kron(Π1, I(2))]
    end
    return Π
end

function gkraus(p_z)
    if p_z == 1
        return [kron(Matrix(I, 2, 4), Matrix(I, 2, 3))]
    elseif p_z == 0
        return [kron([0 0 1 0; 0 0 0 1], Matrix(I, 2, 3))]
    else
        EZ = kron(proj(1, 4) + proj(2, 4), Matrix(I, 2, 3))
        EX = kron(proj(3, 4) + proj(4, 4), Matrix(I, 2, 3))
        return [√p_z * EZ, √(1 - p_z) * EX]
    end
end

function corr(ρ::AbstractMatrix)
    ZA = [proj(1, 4), proj(2, 4)]
    XA = [proj(3, 4), proj(4, 4)]
    Z = [proj(1, 2), proj(2, 2)]
    X = [0.5 * [1 1; 1 1], 0.5 * [1 -1; -1 1]]
    ZB = [zeros(3, 3) for i ∈ 1:2]
    XB = [zeros(3, 3) for i ∈ 1:2]
    ZB[1][1:2, 1:2] = Z[1]
    ZB[2][1:2, 1:2] = Z[2]
    XB[1][1:2, 1:2] = X[1]
    XB[2][1:2, 1:2] = X[2]
    basis_A = [ZA; XA]
    basis_B = [ZB; XB]
    global_basis = [kron(A, B) for A ∈ basis_A, B ∈ basis_B]
    return real(dot.(Ref(ρ), global_basis))
end

function target_probabilities(η::T, δ::T, pd::T) where {T}
    κ = 1 + δ / π
    ϕ = (T(0), κ * π, κ * π / 2, (3 * κ * π) / 2)
    probAB = fill((1 - η) * pd * (1 - pd / 2), 4, 4)
    for i ∈ 1:4
        p0 = (1 + cos(ϕ[i])) / 2
        p1 = 1 - p0
        pp = (1 + sin(ϕ[i])) / 2
        pm = 1 - pp
        probAB[i, 1] += η * (p0 * (1 - pd) + pd / 2)
        probAB[i, 2] += η * (p1 * (1 - pd) + pd / 2)
        probAB[i, 3] += η * (pp * (1 - pd) + pd / 2)
        probAB[i, 4] += η * (pm * (1 - pd) + pd / 2)
    end
    return probAB
end

function states(δ::T) where {T}
    k = 1 + δ / π
    zero = ket(1, 2)
    one = cos(k * π / 2) * ket(1, 2) + sin(k * π / 2) * ket(2, 2)
    plus = cos(k * π / 4) * ket(1, 2) + sin(k * π / 4) * ket(2, 2)
    minus = cos(k * 3 * π / 4) * ket(1, 2) + sin(k * 3 * π / 4) * ket(2, 2)
    return [zero one plus minus]
end

function combined_state(p_z::T, δ::T, η::T) where {T<:Real}
    k = 1 + δ / π
    zero = ket(1, 3)
    one = cos(k * π / 2) * ket(1, 3) + sin(k * π / 2) * ket(2, 3)
    plus = cos(k * π / 4) * ket(1, 3) + sin(k * π / 4) * ket(2, 3)
    minus = cos(k * 3 * π / 4) * ket(1, 3) + sin(k * 3 * π / 4) * ket(2, 3)
    ψ = sqrt(p_z / 2) * (kron(ket(1, 4), zero) + kron(ket(2, 4), one))
    ψ .+= sqrt((1 - p_z) / 2) * (kron(ket(3, 4), plus) + kron(ket(4, 4), minus))
    ψmatrix = ketbra(ψ)
    ρ = η * ψmatrix + (1 - η) * kron(partial_trace(ψmatrix, 2, [4, 3]), proj(3, 3))
    return ρ
end

function hab_bb84(p_z, η, δ, pd)
    corr_target = target_probabilities(η, δ, pd)
    p01 = corr_target[1:2, 1:2]
    p01 ./= 2
    ppm = corr_target[3:4, 3:4]
    ppm ./= 2
    return p_z^2 * conditional_entropy(p01) + (1 - p_z)^2 * conditional_entropy(ppm)
end

function total_rate_bb84(ϵ::T, δ::T, D; eph = false) where {T<:Real}
    α = T(2) / 10
    ηbin = T(1) / 2
    ηdetector = T(73) / 100
    η = 10^(-α * D / 10)
    f = T(116) / 100
    pd = inv(T(10^6))
    p_z = T(1)
    if eph
        K = hae_bb84_eph(ϵ, η, δ, pd) - f * hab_bb84(p_z, η, δ, pd)    
    else
        K = hae_bb84(p_z, ϵ, η, δ, pd) - f * hab_bb84(p_z, η, δ, pd)
    end
    return K
end

function hae_bb84(p_z::T, ϵ::T, η::T, δ::T, pd::T) where {T<:Real}
    is_complex = false
    is_complex ? R = Complex{T} : R = T
    psd_cone, wrapper, hermitian_space = Ket._sdp_parameters(is_complex)

    dim_ρ = 12
    model = GenericModel{T}(() -> Hypatia.Optimizer{T}(; verbose = true))

    bad_states = states(δ)
    fidelities = bad_states' * bad_states
    #fidelities ordered as ⟨ϕ1|ϕ0⟩ ⟨ϕ+|ϕ0⟩ ⟨ϕ-|ϕ0⟩ ⟨ϕ+|ϕ1⟩ ⟨ϕ-|ϕ1⟩ ⟨ϕ-|ϕ+⟩
    
    correction = zeros(T, 4, 4)
    if p_z == 1 || p_z == 0
        diag_target = fill(T(0.5), 4)
        correction[1, 2] = 2
        correction[1:2, 3:4] .= 2
        correction[3, 4] = 2
    else
        diag_target = 0.5 * [p_z, p_z, 1 - p_z, 1 - p_z]
        correction[1, 2] = 2 / p_z
        correction[1:2, 3:4] .= 2 / sqrt(p_z * (1 - p_z))
        correction[3, 4] = 2 / (1 - p_z)
    end
    correction += correction' + Diagonal(inv.(diag_target))

    if ϵ == 0
        @variable(model, ρcore[1:6, 1:6] ∈ hermitian_space)
        if p_z == 1 || p_z == 0
            u, s, v = svd(bad_states)
        else
            u, s, v = svd(bad_states * Diagonal(sqrt.(diag_target)))
        end
        V = kron(v, I(3))
        ρ = wrapper(V * ρcore * V')
        ρA = partial_trace(ρ, 2, [4, 3])
        for j ∈ 2:4, i ∈ 1:j-1
            @constraint(model, ρA[j, i] * correction[j, i] == fidelities[i, j])
        end
    else
        @variable(model, ρ[1:dim_ρ, 1:dim_ρ] ∈ hermitian_space)
        ρA = partial_trace(ρ, 2, [4, 3])

        @variable(model, Gcore[1:6, 1:6] ∈ psd_cone)
        @constraint(model, diag(Gcore) .== 1)
        @constraint(model, Gcore[1, 2] == 0)
        overlaps = Matrix{typeof(1 * Gcore[1])}(undef, 4, 4)
        for j ∈ 1:4, i ∈ 1:4
            @views overlaps[i, j] = dot(bad_states[:, i], Gcore[1:2, 2+j])
        end
        @constraint(model, diag(overlaps) .== 0) #constraints ⟨ϕ_i|ϕ_i^⟂⟩ == 0
        G = wrapper([fidelities overlaps; overlaps' Gcore[3:6, 3:6]])

        for j ∈ 2:4, i ∈ 1:j-1
            @constraint(
                model,
                ρA[j, i] * correction[j, i] ==
                (1 - ϵ) * fidelities[i, j] + √(ϵ * (1 - ϵ)) * (G[i, j+4] + G[i+4, j]) + ϵ * G[i+4, j+4]
            )
        end
    end

    corr_ρ = Diagonal(inv.(diag_target)) * corr(ρ)
    corr_target = target_probabilities(η, δ, pd)
    @constraint(model, corr_ρ[:, [1, 2, 3]] .== corr_target[:, [1, 2, 3]])
    @constraint(model, diag(ρA) .== diag_target)

    Ghat = gkraus(p_z)
    Z = zkraus(p_z)
    Zhat = vec([Zi * Gi for Gi ∈ Ghat, Zi ∈ Z])

    ρ_vec = svec(ρ)
    @variable(model, h)
    @objective(model, Min, h / log(T(2)))
    @constraint(model, [h; ρ_vec] in EpiQKDTriCone{T,R}(Ghat, Zhat, 1 + length(ρ_vec)))

    optimize!(model)
    return objective_value(model)
end

hae_bb84_analytic(p_z, η) = η * ((1 - p_z)^2 + p_z^2)

function phase_error_rate_sdp_gram(ϵ::T, η::T, δ::T, pd::T) where {T<:Real}
    is_complex = false
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

    # BB84 parameters
    n_alice = 4  # number of states: |0⟩, |1⟩, |+⟩, |−⟩
    n_bob = 3    # number of outcomes: 0X, 1X, fail
    n_basis = 2  # |0⟩ and |1⟩ basis states
    m = n_basis + n_alice  # 6

    # Get target probabilities
    Y = target_probabilities(η, δ, pd)
    # Y is a 4×4 matrix with:
    # - Rows: Alice's states [|0⟩, |1⟩, |+⟩, |−⟩]
    # - Columns: Bob's outcomes [0z, 1z, 0x, 1x]
    # For BB84 phase error rate, we need Bob's X basis measurements:
    # Y[i,3] = probability Bob gets 0 in X basis when Alice sends state i
    # Y[i,4] = probability Bob gets 1 in X basis when Alice sends state i

    # coef_states is a 2×4 matrix where each column is a state vector
    # expressed in the computational basis
    coef_states = states(δ)

    # Get state inner products (fidelities)
    fidelities = coef_states' * coef_states

    # Create smaller Gram matrix Gnew
    dim_Gnew = m * n_bob  # 18 dimensions
    @variable(model, Gnew[1:dim_Gnew, 1:dim_Gnew] in psd_cone)

    # Construct G from Gnew
    dim_G = n_alice * n_bob * 2  # 24 dimensions
    G = zeros(typeof(1 * Gnew[1, 1]), dim_G, dim_G)

    # fill!(G,NaN) # (uncomment to make sure we are not using any element of G that is not filled below)

    # Fill all elements of G using the relationship between G and Gnew
    for l ∈ 0:n_bob-1
        # G entries for good-good states
        for i ∈ 1:n_alice, j ∈ 1:i
            sum_val = zero(typeof(1 * Gnew[1, 1]))
            sum_conj = zero(typeof(1 * Gnew[1, 1]))
            for k ∈ 1:n_basis, kp ∈ 1:n_basis
                sum_val += coef_states[kp, j] * coef_states[k, i] * Gnew[k+l*m, kp+l*m]
                sum_conj += coef_states[kp, j] * coef_states[k, i] * Gnew[kp+l*m, k+l*m]
            end
            G[i+l*2*n_alice, j+l*2*n_alice] = sum_val
            G[j+l*2*n_alice, i+l*2*n_alice] = sum_conj
        end

        # G entries for good-perpendicular states
        for i ∈ 1:n_alice, j ∈ 1:n_alice
            sum_val = zero(typeof(1 * Gnew[1, 1]))
            sum_conj = zero(typeof(1 * Gnew[1, 1]))
            for k ∈ 1:n_basis
                sum_val += coef_states[k, j] * Gnew[i+n_basis+l*m, k+l*m]
                sum_conj += coef_states[k, j] * Gnew[k+l*m, i+n_basis+l*m]
            end
            G[i+n_alice+l*2*n_alice, j+l*2*n_alice] = sum_val
            G[j+l*2*n_alice, i+n_alice+l*2*n_alice] = sum_conj
        end

        # G entries for perpendicular-perpendicular
        for i ∈ 1:n_alice, j ∈ 1:i
            G[i+n_alice+l*2*n_alice, j+n_alice+l*2*n_alice] = Gnew[i+n_basis+l*m, j+n_basis+l*m]
            G[j+n_alice+l*2*n_alice, i+n_alice+l*2*n_alice] = Gnew[j+n_basis+l*m, i+n_basis+l*m]
        end
    end

    # Define Gprod function for ϵ ≠ 0 case
    function Gprod(i::Int, j::Int)
        if ϵ == 0
            return G[i, j]
        else
            return (1 - ϵ) * G[i, j] +
                   sqrt((1 - ϵ) * ϵ) * (G[i, j+n_alice] + G[i+n_alice, j]) +
                   ϵ * G[i+n_alice, j+n_alice]
        end
    end

    # Constraints: observed yields

    # For 0X outcome
    for i ∈ 1:n_alice
        @constraint(model, Gprod(i, i) == Y[i, 3])
    end
    # For 1X outcome
    for i ∈ 1:n_alice
        @constraint(model, Gprod(i + 2 * n_alice, i + 2 * n_alice) == Y[i, 4])
    end

    # Constraints: completeness relation (sum over all outcomes equals fidelity)
    for j ∈ 2:n_alice, i ∈ 1:j-1
        @constraint(model, G[i, j] + G[i+2*n_alice, j+2*n_alice] + G[i+4*n_alice, j+4*n_alice] == fidelities[i, j])
    end

    # Normalization constraints
    for i ∈ 1:2*n_alice
        @constraint(model, G[i, i] + G[i+2*n_alice, i+2*n_alice] + G[i+4*n_alice, i+4*n_alice] == 1)
    end

    # Orthogonality constraints for perpendicular states
    for i ∈ 1:n_alice
        @constraint(model, G[i, i+n_alice] + G[i+2*n_alice, i+3*n_alice] + G[i+4*n_alice, i+5*n_alice] == 0)
    end

    # Objective function: phase error rate for Z basis
    # States 1 and 2 are Z basis states (|0⟩ and |1⟩)

    obj =
        (1 / 4) * (
            Gprod(2 * n_alice + 1, 2 * n_alice + 1) +
            Gprod(2 * n_alice + 1, 2 * n_alice + 2) +
            Gprod(2 * n_alice + 2, 2 * n_alice + 1) +
            Gprod(2 * n_alice + 2, 2 * n_alice + 2) +
            Gprod(1, 1) - Gprod(1, 2) - Gprod(2, 1) + Gprod(2, 2)
        )

    @objective(model, Max, obj)

    optimize!(model)
    return value(obj)
end

function hae_bb84_eph(ϵ::T, η::T, δ::T, pd::T) where {T<:Real}
    # This function computes a bound on hae by first computing 
    # a bound on the phase error rate eph using SDP

    # First, compute the phase error probability upper bound using SDP
    Y_Z_ph = phase_error_rate_sdp_gram(ϵ, η, δ, pd)

    # Compute Y_Z (the total detection probability for Z basis states)
    Y = target_probabilities(η, δ, pd)
    # Y is a 4×4 matrix with rows: [|0⟩, |1⟩, |+⟩, |−⟩] and columns: [0z, 1z, 0x, 1x]
    # Z basis yields are in the first two rows and first two columns
    Y_Z = (Y[1, 1] + Y[1, 2] + Y[2, 1] + Y[2, 2]) / 2

    # Compute the phase error rate
    ephA = Y_Z_ph / Y_Z

    # Compute hae using binary entropy
    if ephA > 0.5
        hae = T(0)  # Key rate goes to zero if phase error rate > 0.5
    else
        hae = 1 - binary_entropy(ephA)
    end

    return Y_Z * hae
end

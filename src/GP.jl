module GP

using Distances, LinearAlgebra, Optim


# structs
struct hyperparameters
    σ::AbstractVector
    ℓ::AbstractVector
    σ_n::Float64
    ℓx::Vector{Int64}
end

struct hyperparameters_multi
    σ::AbstractVector
    ℓ::AbstractVector
    σ_n::Float64
    ℓx::Vector{Int64}
end

struct GP_obj
    X::AbstractArray
    Lxx::AbstractArray
    α::AbstractArray
    Θ::hyperparameters
end

struct GP_obj_multi
    X::AbstractArray
    Lxx::AbstractArray
    α::AbstractArray
    Θ::hyperparameters_multi
end

struct SS
    A::AbstractArray
    B::AbstractArray
    C::AbstractArray
    D::AbstractArray
end

# squared exponential kernel
function k_SE(R, σ)
    K = (σ^2) * exp.(-(1/2) * R)
    return K
end

# calculate distance matrix 
function r(x₁, x₂) 
    r = pairwise(SqEuclidean(), x₁, x₂, dims=2)
    return r
end



# multiply kernels
function r_SE_mul(X₁, X₂, ℓ)
    R = zeros(size(X₁,1), size(X₂,1))
    for i in 1:size(X₁,2)
        R += r(reshape(X₁[:, i], 1, size(X₁,1)), reshape(X₂[:, i], 1, size(X₂,1))) ./ (ℓ[i]^2) 
    end
    return R
end

# analytical derivative of posterior mean for dimension d
function δμδxsf(GP_obj,Xs,Σs,k,d)
    return -((GP_obj.Θ.ℓ[GP_obj.Θ.ℓx[d]].^(-2))*(GP_obj.X[:,d] .- Xs[:,d]') .* k(GP_obj.X,Xs,GP_obj.Θ,d))'*GP_obj.α
end


function εf(GP_obj,Xs,Σs,k)
    δμδxs = zeros(size(Xs))
    for d in 1:size(Xs,2) 
        δμδxs[:,d] = δμδxsf(GP_obj,Xs,Σs,k,d) 
    end
    return [(δμδxs[i,:]' * + diagm(Σs[i,:]) * δμδxs[i,:]) for i in 1:size(Xs,1)]
end

function μf(X, xs, k, Θ, α)
    return (k(X,xs,Θ)'*α)[]
end

# calculate posterior distribution
function posterior(GP_obj, Xs, Σs, k, calc_σ)
    Kxs = k(GP_obj.X,Xs,GP_obj.Θ)
    Kss = k(Xs,Xs,GP_obj.Θ)
    μ = (Kxs'*GP_obj.α)
    # uncertainty propagation
    if calc_σ==true
        σ = sqrt.(diag(Kss) - sum((GP_obj.Lxx \ Kxs).^2, dims=1)' + εf(GP_obj,Xs,Σs,k))  
    else
        σ = zeros(size(Xs,1))
    end
    return μ, σ
end

# training kernel cholesky factorization
function calcLxx(X, y, k, Θ::hyperparameters)   
    # kernel
    Kxx = k(X,X,Θ) + Θ.σ_n*I 
    Cxx = cholesky(Kxx)
    Lxx = Cxx.L
    Lxx_y = (Lxx \ y) 
    α = Lxx' \ Lxx_y
    return Lxx, α
end


# negative log marginal likelihood (for optimization of hyperparameters)
function nlml(X, y, k, Θₒₚₜ, Θ₀::hyperparameters)
    nσ, nℓ = size(Θ₀.σ,1), size(Θ₀.ℓ,1)
    Θ = hyperparameters(Θₒₚₜ[1:nσ], Θₒₚₜ[nσ+1:nσ+nℓ], Θ₀.σ_n, Θ₀.ℓx)
    # cholesky factorization of k(X,X')
    Lxx, α = calcLxx(X, y, k, Θ)
    # log marginal likelihood
    n = size(X,1)
    logml = -(1/2) * y' * α  - sum(log.(diag(Lxx))) - (n/2) * log(2*pi)
    print(".")
    return -logml[1]
end 

# greedy learning of sparse approximation
function sparse_subset(X, y, ix, k, Θ, m)
    # dimensions of input
    d = size(X,2)
    # induced points
    Xm = X[1,:]'
    ym = y[1] 
    Θm = Θ
    ixm = ix[1]
    # not-yet-chosen points
    Xn = X[2:end, :]
    yn = y[2:end]
    
    print("[")
    for j = 1:(m-1)
        Hmax=-1e5; xmax=zeros(size(X,2)); ymax=0; imax=0; xs=zeros(size(X,2)); ys=0
        print(".")
        for i in 1:size(yn,1)
            # new test point
            xs = Xn[i, :]'
            ys = yn[i]
            Xms = vcat(Xm, xs)
            yms = vcat(ym, ys)
            # calculate the entropy
            Lxx, α = calcLxx(Xms, yms, k, Θ)
            Hs = (1/2)*sum(log.(((2*pi*exp(1))^d)*(diag(Lxx))))
            #Kxx = k(Xms,Xms,Θ) + 0.001*I 
            #Hs = (1/2)*sum(log.(((2*pi*exp(1))^d)*(det(Kxx))))
            # use point that maximizes differential entropy
            if Hs>Hmax
                #println("$Hs > $Hmax")
                Hmax = Hs
                xmax = xs
                ymax = ys
                imax = i
            end
        end
        if imax!=0
            Xm = vcat(Xm, xmax)
            ym = vcat(ym, ymax)
            ixm = vcat(ixm,ix[imax])
            Xn = Xn[[i for i in 1:size(Xn,1) if i!=imax],:]
            yn = yn[[i for i in 1:size(Xn,1) if i!=imax]]
        end
    end
    println("]")
    return Xm, ym, ixm
end


# optimize hyperparameters 
function optimize_hyperparameters(X, y, k, Θ₀::hyperparameters)
    Θ₀_opt = vcat(Θ₀.σ, Θ₀.ℓ)
    nσ, nℓ = length(Θ₀.σ), length(Θ₀.ℓ)
    lower = vcat(zeros(nσ), zeros(nℓ))
    upper = [Inf for i in 1:(nσ+nℓ)]

    print("[")
    results = optimize(Θ -> nlml(X, y, k, Θ, Θ₀), lower, upper, Θ₀_opt, Fminbox(LBFGS())) #LBFGS() 
    println("]")
    print(results)

    Θ₁ = results.minimizer
    Θp = hyperparameters(Θ₁[1:nσ], Θ₁[nσ+1:nσ+nℓ], Θ₀.σ_n, Θ₀.ℓx)

    return Θp
end

end
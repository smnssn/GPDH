module ModelicaPipe_GPSSM

using GP, Plots, CSV, DataFrames

function read_csv(file_suffix="1", x_i=4:6, y_i=2)
    # read csv and create datasets
    data_df = CSV.read("data/gp_data_$file_suffix.csv", DataFrame)   
    input_df = CSV.read("data/input_data_$file_suffix.csv", DataFrame)
    validation_df = CSV.read("data/validation_data_$file_suffix.csv", DataFrame)
    
    # read csv and create datasets
    data_df_p = CSV.read("data/gp_data_$file_suffix.csv", DataFrame)
    y = data_df_p[:,y_i]
    X = convert(Array, data_df_p[:,x_i])

    # indicies
    #ix = [(data_df.t_samples[i], data_df.x_samples[i]) for i in 1:size(data_df,1)]
    ix = [(data_df.x_samples[i], data_df.t_samples[i]) for i in 1:size(data_df,1)]

    return X, y, ix, data_df, input_df, validation_df
end




function update_inputs(T_sim_t, V_sim_t, T_in, m_flow_in, Σs_in, nx)
    # initialize
    Xs_t = zeros(nx, 3)
    Σs_t = zeros(nx, 3)

    # update inputs for temperature
    Xs_t[1, 1] = T_in
    Xs_t[2:end, 1] = T_sim_t[2:end]
    # update inputs for Δ temperature
    Xs_t[1, 2] = T_sim_t[1] - T_in
    Xs_t[2:end, 2] = T_sim_t[2:end] - T_sim_t[1:end-1] 
    # update inputs for m_flow
    Xs_t[:, 3] = m_flow_in*ones(size(Xs_t,1))
    
    # propagate forward Σs, set nx=1 from input
    Σs_t[1, :] = Σs_in
    Σs_t[2:end, 1] = V_sim_t[2:end]
    Σs_t[2:end, 2] = (V_sim_t[1:end-1] + V_sim_t[2:end])
    Σs_t[2:end, 3] = Σs_in[3]*ones(size(Σs_t,1)-1)
   
    return Xs_t, Σs_t
end

function update_states!(T_sim_t, V_sim_t, T_in, Δ_sim, σ_sim)
    # update T and V
    T_sim_t[1] = T_in
    T_sim_t[2:end] = T_sim_t[2:end] + Δ_sim[2:end] 
    V_sim_t[1:end] = σ_sim
end


function forward_simulation(t_range, input_df, GP_pipe::GP.GP_obj, k, nx, calc_σ, nmc=0)
    # time series of inputs
    Σs_in = [5.0,5.0,1]
    #Σs_in = [0,0,0]
    
    T_in = Array(input_df[1:length(t_range)+1,2])
    T_in = hcat(T_in, T_in .+ Σs_in[2]*randn(length(t_range)+1,nmc))
    m_flow_in = Array(input_df[1:length(t_range)+1,3])
    m_flow_in = hcat(m_flow_in, m_flow_in .+ Σs_in[3]*randn(length(t_range)+1,nmc))
    
    # inputs at time t
    #Xs_t = zeros(Float64, nx, 3)
    #Σs_t = zeros(Float64, nx, 3)


    # new values
    Δ_sim = zeros(nx)
    σ_sim = zeros(nx)

    print("Simulating pipe [")
    # initialize
    T_sim_mc, V_sim_mc = zeros((nmc+1), nx, length(t_range)), zeros((nmc+1), nx, length(t_range))
    
    for n in 1:(nmc+1)
        print("[n = $n]")
        T_sim, V_sim = (273.15+40)*ones(nx), zeros(nx)
        T_sim_t, V_sim_t = (273.15+40.0)*ones(nx), zeros(nx)
        # run in t_range
        for t in t_range
            print(".")
            simulation_step!(T_sim_t, V_sim_t, GP_pipe, T_in[t,n], m_flow_in[t,n], Σs_in, nx)
            T_sim, V_sim = hcat(T_sim, T_sim_t), hcat(V_sim, V_sim_t)
        end
        # store results
        T_sim, V_sim = T_sim[:, 2:end] .- 273.15, V_sim[:, 2:end]
        T_sim_mc[n, :, :], V_sim_mc[n, :, :] = T_sim, V_sim
        # re-initialize
        T_sim, V_sim = (273.15+40)*ones(nx), zeros(nx)
        T_sim_t, V_sim_t = (273.15+40.0)*ones(nx), zeros(nx)
    end
    
    print("]")

    return T_sim_mc, V_sim_mc
end


function simulation_step!(T_sim_t, V_sim_t, GP_pipe, T_in, m_flow_in, Σs_in, nx)
    Xs_t, Σs_t = update_inputs(T_sim_t, V_sim_t, T_in, m_flow_in, Σs_in, nx)
    Δ_sim, σ_sim = GP.posterior(GP_pipe, Xs_t, Σs_t, k_pipe, true)   
    update_states!(T_sim_t, V_sim_t, T_in, Δ_sim, σ_sim)
end



function plot_results_t(T_sim_mc, V_sim_mc, T_val, Δt, x, t_range, xlabel, ylabel, filename, save_file, plot_samples)
    t_plot = 1:Δt:Δt*(length(t_range))

    p = plot(xlabel=xlabel, ylabel=ylabel, legend=:topright)

    if plot_samples
        for i in 2:size(T_sim_mc, 1)
            T_sim = T_sim_mc[i, :, :]
            V_sim = V_sim_mc[i, :, :]
            μs_x = T_sim[x, t_range]; 
            σs_x = V_sim[x, t_range]; 
            ref = T_val[x, t_range];

            # plot mean of GP
            plot!(p, t_plot, μs_x, linewidth=0.1, linecolor=:blue, alpha=0.3, label=""); 
        end
    end

    T_sim = T_sim_mc[1, :, :]
    V_sim = V_sim_mc[1, :, :]
    μs_x = T_sim[x, t_range]; 
    σs_x = V_sim[x, t_range]; 
    ref = T_val[x, t_range];

    # plot mean of GP
    plot!(p, t_plot, μs_x, label="f(x)", linecolor=:green); 

    # plot validation measurement
    plot!(p, t_plot, ref, label="y", linecolor=:red, linestyle=:dash); 

    # plot posterior marginal std. dev.
    plot!(p, t_plot, [μs_x μs_x];
        linewidth=0.0,
        fillrange=[μs_x .- 3  .* σs_x μs_x .+ 3 .* σs_x],
        fillalpha=0.1,
        fillcolor=:blue,
        label="");

    if save_file
        savefig(p, filename)
    else
        display(p)
    end
end

function plot_results_x(T_sim_mc, V_sim_mc, T_val, Δx, x_range, t, xlabel, ylabel, filename, save_file)
    T_sim = T_sim_mc[1, :, :]
    V_sim = V_sim_mc[1, :, :]

    x_plot = 1:Δx:Δx*(length(x_range))
    μs_x = T_sim[x_range, t]; 
    σs_x = V_sim[x_range, t]; 
    ref = T_val[x_range, t];

    p = plot(xlabel=xlabel, ylabel=ylabel, legend=:topright)

    # plot mean of GP
    plot!(p, x_plot, μs_x, label="f(x)", linecolor=:green); 

    # plot validation measurement
    plot!(p, x_plot, ref, label="y", linecolor=:red, linestyle=:dash); 

    # plot posterior marginal std. dev.
    plot!(p, x_plot, [μs_x μs_x];
        linewidth=0.0,
        fillrange=[μs_x .- 3  .* σs_x μs_x .+ 3 .* σs_x],
        fillalpha=0.1,
        fillcolor=:blue,
        label="");

    if save_file
        savefig(p, filename)
    else
        display(p)
    end
end

# pipe specific kernel
function k_pipe(X₁, X₂, Θ::GP.hyperparameters, d=1:3)
    # kernels to multiply
    mul_1 = [2,3]; mul_2 = [1]
    
    # distance matrixes
    r_1 = GP.r_SE_mul(X₁[:, mul_1], X₂[:, mul_1], Θ.ℓ[Θ.ℓx[mul_1]])
    r_2 = GP.r_SE_mul(X₁[:, mul_2], X₂[:, mul_2], Θ.ℓ[Θ.ℓx[mul_2]])
    # squared exponential kernel
    k_1 =  GP.k_SE(r_1, Θ.σ[1])
    k_2 = GP.k_SE(r_2, Θ.σ[2])
    if d in mul_1
        return k_1
    elseif d in mul_2
        return k_2
    else
        return k_1 + k_2
    end
end

function POD(X,y,ix)
    return X, X, 0
end




function generate_GP(n_pre, n_large, m_sparse, Θ₀)

    # read csv files
    Xp, yp, ixp = read_csv("pre", 4:6, 2)
    Xl, yl, ixl = read_csv("large", 4:6, 2)
    
    
    # pre-optimize hyperparameters on the n dataset
    println("Pre-optimizing hyperparameters...")
    Θp = GP.optimize_hyperparameters(Xp, yp, k_pipe, Θ₀)


    # find a sparse approximation with m_sparse points
    println("Finding sparse approximation...")
    Xm, ym, ixm = GP.sparse_subset(Xl, yl, ixl, k_pipe, Θp, m_sparse)


    # re-optimize hyperparameters on the sparse dataset
    println("Post-optimizing hyperparameters...")
    Θm = GP.optimize_hyperparameters(Xm, ym, k_pipe, Θp)

    # cholesky factorization and inverse of K
    Lxxm, αm = GP.calcLxx(Xm, ym, k_pipe, Θm)

    return GP.GP_obj(Xm, Lxxm, αm, Θm), ixl, ixm
end

end
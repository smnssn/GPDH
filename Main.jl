using Plots, Serialization, LaTeXStrings, Printf

push!(LOAD_PATH,joinpath(pwd(),"src"));

using GP
using ModelicaPipe_GPSSM


###########################
# Main loop
###########################
# parameters [50-100-1000-100 funkar bra]
nx = 50 # number of segments of simulated pipe
n_pre = 100 # number of points for initial dataset
n_large = 1000 # number of points for large dataset
m_sparse = 50 # number of points for sparse dataset
Θ₀ = GP.hyperparameters([1.0, 1.0], [1.0, 1.0, 1.0], 0.01, [3,1,2]) 


# generate the GP
GP_pipe, ixl, ixm = ModelicaPipe_GPSSM.generate_GP(n_pre, n_large, m_sparse, Θ₀)
k_pipe = ModelicaPipe_GPSSM.k_pipe

# get validation data
Xv, yv, ixv, data_df, input_df, validation_df = ModelicaPipe_GPSSM.read_csv("val", 4:6, 2)
T_val = Array(validation_df[:,2:51])' .- 273.15

# create ranges for simulation 
t_range = 1:1000 #size(input_df,1)-1


# number of simulations for monte carlo type simulation 
nmc=1

# simulation with sparse GP
T_sim_mc, V_sim_mc = @time ModelicaPipe_GPSSM.forward_simulation(t_range, input_df, GP_pipe, k_pipe, nx, true, nmc)

# plots
gr(); save_results=false

# results at pipe outlet
Δt = 5; Δx = 10; x=50; 
ModelicaPipe_GPSSM.plot_results_t(T_sim_mc, V_sim_mc, T_val, Δt, x, t_range, "Time [s]", "Temperature [C]", "plots/posterior_plot_t.pdf", save_results, false);

# monte carlo simulation results
ModelicaPipe_GPSSM.plot_results_t(T_sim_mc, V_sim_mc, T_val, Δt, x, t_range, "Time [s]", "Temperature [C]", "plots/posterior_plot_t_mc.pdf", save_results, true);

# spatial distribution
x_range=1:50; t=20;
ModelicaPipe_GPSSM.plot_results_x(T_sim_mc, V_sim_mc, T_val, Δx, x_range, t, "Position [m]", "Temperature [C]", "plots/posterior_plot_x.pdf", save_results);


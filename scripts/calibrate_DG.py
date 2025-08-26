import os
import sys
sys.path.append('.')
import numpy as np
import xarray as xr
from slurm_DG import *

## Model configuration
ZB_smooth_params = PARAMETERS.add(USE_ZB2020='True', STRESS_SMOOTH_PASS=4, DAYMAX=3650.0)
base_path = '/scratch/pp2681/mom6/Apr2023/R4-sensitivity'
optimization_idx = 'Vanilla-EKI-ens-size-20'
N_iterations = 5

from julia import Main

Main.eval("""
    using EnsembleKalmanProcesses, EnsembleKalmanProcesses.ParameterDistributions, Random, LinearAlgebra
    prior_1 = constrained_gaussian("Smagorinsky", 0.045, 0.045/2., 0.0, 0.09)
    prior_2 = constrained_gaussian("ZB-coefficient", 1.5, 0.75, 0.0, 3.0)
    prior = combine_distributions([prior_1, prior_2])
            
    Random.seed!(1)   # Fix random numbers globally
            
    # Noise variance of observation + noise variance of the forward model
    # In the 5-year mean ssh on 1-degree grid. Variance holds most
    # of the support of the variance spatial map.
    Γ = 0.01*I
    # Note, for simplicity we incorporate observation vector to the
    # forward model. Thus,
    y = zeros(440)  # 440 is the size of the ssh vector
            
    ########### Initialize the EnsembleKalmanProcess ##########
    N_ensemble = 20
    initial_ensemble = construct_initial_ensemble(prior, N_ensemble)
          
    vanilla_eki = EnsembleKalmanProcess(
    initial_ensemble, y, Γ, Inversion(),
    scheduler = DefaultScheduler(1),
    accelerator = DefaultAccelerator(),
    localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
    verbose=true)
    """)

os.makedirs(f'{base_path}/{optimization_idx}', exist_ok=True)

for iteration in range(N_iterations):
    print(f'################ iteration {iteration} ####################')
    # Return the parameters in constrained space
    params = Main.eval("get_ϕ_final(prior, vanilla_eki)")

    iteration_path = f'{base_path}/{optimization_idx}/iteration-{iteration:02d}'
    params_file = f'{iteration_path}-params.txt'

    print('Params mean/std', params.mean(-1), '/', params.std(-1))

    if not(os.path.exists(params_file)):
        print('Saving parameters to file', params_file)
        np.savetxt(params_file, params)
    else:
        params_old = np.loadtxt(params_file)
        if not(np.allclose(params, params_old)):
            print('Parameters changed! Check the optimization algorithm.')
            sys.exit(1)   # terminate immediately with error code
        else:
            print('Parameters are the same. Keep going...')

    if os.path.exists(iteration_path):
        print('Folder with experiments exists')
        ssh = lambda x: x.ea.isel(zi=0).sel(Time=slice(1825,3650)).mean('Time')
        ssh_fine = ssh(xr.open_mfdataset('/scratch/pp2681/mom6/Feb2022/bare/R64/output/ave_*.nc', decode_times=False)).coarsen({'xh': 64, 'yh': 64}).mean().compute()

        # 440 is the size of ssh vector
        g_ens = np.zeros(440, params.shape[1])
        for ens_member, param in enumerate(params.T):
            try:
                ssh_coarse = ssh(xr.open_mfdataset(f'{iteration_path}/ens-member-{ens_member:02d}/output/ave_*.nc', decode_times=False)).coarsen({'xh': 4, 'yh': 4}).mean().compute()
                # Error vector of size 440
                g_ens[:,ens_member] = (ssh_coarse - ssh_fine).values.ravel()
                print(f'Ensemble member {ens_member} succesfully ingested')
            except:
                # Experiment is not ready or exploded or runtime error
                g_ens[:,ens_member] = np.nan
                print(f'Ensemble member {ens_member} failed. Filled with NaNs')

        Main.g_ens = g_ens
        Main.eval("update_ensemble!(vanilla_eki, g_ens, deterministic_forward_map=false)")
        print('Forward model evaluations are passed to the EKI. Going to the next iterations...')
    else:
        print('Run experiments in folder ', iteration_path)
        for ens_member, param in enumerate(params.T):
            MOM6_parameters = ZB_smooth_params.add(
                SMAG_BI_CONST=param[0],
                ZB_SCALING=param[1]
            )
            hpc = HPC.add(mem=2, ntasks=4, time=2, executable='/home/pp2681/MOM6-examples/build/compiled_executables/MOM6-ZB-2023')
            run_experiment(f'{iteration_path}/ens-member-{ens_member:02d}', hpc, MOM6_parameters)

        print('Experiments are scheduled')
        print('Putting in a queue resubmission script')
        os.system('cd /home/pp2681/calibration/scripts; sbatch --mem=16GB --dependency=singleton --job-name=mom6 --wrap="python-jl calibrate_DG.py"')
        print('Exiting the script')
        sys.exit(0)   # terminate immediately without error code
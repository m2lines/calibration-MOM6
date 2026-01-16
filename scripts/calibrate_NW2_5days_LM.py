import os
import sys
sys.path.append('.')
import numpy as np
import xarray as xr
from slurm_DG import *
from loss_DG import *
import argparse

############## To get started ################
#  python-jl calibrate_NW2_5days_LM.py --echo

## Global paths
TAG = 'LM'
base_path = '/scratch/pp2681/mom6/CM26_Double_Gyre/calibration/NW2'
optimization_folder = 'base_5days_LM'
ANN_default_path = '/scratch/pp2681/mom6/CM26_ML_models/ocean3d/subfilter/FGR3/equivariant/learning_rate/N8-forcing-fluxes/0.05/model/'
this_file = os.path.abspath(__file__)  # full path of current script
script_name = os.path.basename(this_file)  # just the filename
commandline = f'cd /home/pp2681/calibration/scripts; sbatch --time=01:00:00 --cpus-per-task=1 --mem=16GB --dependency=singleton --export=NONE --job-name={TAG} -o {base_path}/{optimization_folder}/slurm-%j.out -e {base_path}/{optimization_folder}/slurm-%j.err --wrap="python-jl {script_name}"'
# Here, we attenuate the spread of the initial ensemble as if not doing so, experiments explode

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e", "--echo", action="store_true",
    help="Print the command to start LM optimization"
)
args = parser.parse_args()
if args.echo:
    print('To start the LM optimization, run the following command')
    print(commandline)
    sys.exit(0)

# Model configuration
exp_params = PARAMETERS_NW2.add(DAYMAX=5.0, MAXTRUNC=0).add(USE_ZB2020='True', ZB_SCALING = 1.0, ZB_KLOWER_R_DISS = 1.0, ZB_KLOWER_SHEAR = 1,  ZB2020_USE_ANN='True', ZB2020_ANN_FILE_TALL='INPUT/Tall.nc', USE_CIRCULATION_IN_HORVISC='True')
# Read necessary NETCDF files
ANN_netcdf_default = xr.open_dataset(f'{ANN_default_path}/eANN.nc').load()
# Read observation vector
observation = xr.open_dataset('/scratch/pp2681/mom6/Neverworld2/simulations/R32/R2_FGR3/EKI_dataset.nc').isel(time=0).squeeze()

N_iterations = 10

# Observation vector for LM algorithm
# 70400 values of interface mean
y = observation.e.values.ravel().astype('float64')

# Scalar product for LM algorithm
Gamma = observation.e_var.values.ravel().astype('float64')

J_initial = xr.open_dataset('/scratch/pp2681/mom6/CM26_Double_Gyre/calibration/NW2/base_5days/Jacobian.nc').Jacobian

os.makedirs(f'{base_path}/{optimization_folder}', exist_ok=True)

metrics = xr.Dataset()
nzl = 15
ny = 280
nx = 120
metrics['param'] = xr.DataArray(np.nan * np.zeros([N_iterations, 89]), dims=['iter', 'pdim'])
metrics['WMSE'] = xr.DataArray(np.nan * np.zeros([N_iterations]), dims=['iter'])
metrics['J'] = xr.DataArray(np.nan * np.zeros([N_iterations, 504000, 89]), dims=['iter', 'pobs', 'pdim'])
metrics['g'] = xr.DataArray(np.nan * np.zeros([N_iterations, 504000]), dims=['iter', 'pobs'])
metrics['latest_iteration'] = 0

initial_params = []
for parameter_key in ['weights1', 'biases1', 'weights2', 'biases2']:
    initial_params.append(ANN_netcdf_default[parameter_key].values)
initial_params = np.concatenate(initial_params)

metrics['param'][0] = initial_params

if os.path.exists(f'{base_path}/{optimization_folder}/metrics.nc'):
    print('Metrics file already exists. Loading previous metrics...')
    metrics_read = xr.open_dataset(
        f'{base_path}/{optimization_folder}/metrics.nc'
    )

    latest_iteration = int(metrics_read['latest_iteration'].item())

    for var in metrics.data_vars:
        if var == 'latest_iteration':
            continue
        metrics[var].isel(iter=slice(0, len(metrics_read.iter)))[:] = \
            metrics_read[var].isel(iter=slice(0, len(metrics_read.iter))).copy()

    metrics['latest_iteration'] = latest_iteration
    
latest_iteration = int(metrics['latest_iteration'].item())
for iteration in range(latest_iteration, N_iterations):
    print(f'################ iteration {iteration} ####################')
    # Return the parameters in unconstrained space

    iteration_path = f'{base_path}/{optimization_folder}/iteration-{iteration:02d}'
    params_file = f'{iteration_path}-params.txt'
    params = metrics['param'][iteration].values

    if not(os.path.exists(params_file)):
        print('Saving parameters to file', params_file)
        np.savetxt(params_file, params)
    else:
        params_old = np.loadtxt(params_file)
        if not(np.allclose(params, params_old)):
            print('Parameters changed! Check the optimization algorithm.')
            print('Params from txt', params_old)
            print('Params from netcdf', params)
        else:
            print('Parameters are the same. Keep going...')
    #import pdb
    #pdb.set_trace()
    
    if os.path.exists(iteration_path):
        print('Folder with experiments exists')
        try:
            data = xr.open_mfdataset(f'{iteration_path}/output/snapshots_*.nc', decode_times=False).sortby('time').fillna(0.).isel(zi=slice(0,-1)).isel(time=0).squeeze()
            metrics['g'][iteration] = data.e.values.ravel().astype('float64')
        except:
            pass
        
        metrics['WMSE'][iteration] = ((y - metrics['g'][iteration].values)**2 / Gamma).mean()

        if iteration > 0:
            w = metrics['g'][iteration].values - metrics['g'][iteration-1].values
            h = metrics['param'][iteration].values - metrics['param'][iteration-1].values
            # Broyden update using new incoming information
            J = metrics['J'][iteration-1].values
            metrics['J'][iteration] = J + np.outer(w-J@h, h) / np.dot(h,h)
        else:
            # See initialization
            metrics['J'][iteration] = J_initial

        g = metrics['g'][iteration].values
        params = metrics['param'][iteration].values
        J = metrics['J'][iteration].values        

        JJ = (J.T / Gamma)@J

        metrics['param'][iteration+1] = params + np.linalg.inv(JJ + np.trace(JJ)/89 * np.eye(89)) @ (J.T / Gamma) @ (y - g)
        metrics['latest_iteration'] = metrics['latest_iteration'] + 1
        
        os.system(f'rm -f {base_path}/{optimization_folder}/metrics.nc')
        metrics.astype('float32').to_netcdf(f'{base_path}/{optimization_folder}/metrics.nc')

        print('Forward model evaluations are passed to the LM. Going to the next iterations...')
    else:
        print('Run experiments in folder ', iteration_path)
        experiment_folder = f'{iteration_path}'
        param = metrics['param'][iteration].values
        weights_netcdf = ANN_netcdf_default.copy()
        weights_netcdf['weights1'] = xr.DataArray(param[:72], dims='pdim1')
        weights_netcdf['biases1'] = xr.DataArray(param[72:76], dims='pdim3')
        weights_netcdf['weights2'] = xr.DataArray(param[76:88], dims='pdim2')
        weights_netcdf['biases2'] = xr.DataArray(param[88:89], dims='pdim4')

        call_function = ('singularity exec --nv --overlay /scratch/$USER/python-container/python-overlay.ext3:ro '
                        '--bind /scratch/pp2681/python-container/escnn-cache:/ext3/miniconda3/lib/python3.11/site-packages/escnn/group/_cache/ '
                        ' /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif '
                        f' /bin/bash -c "source /ext3/env.sh; time python /home/pp2681/calibration/scripts/eANN_to_ANN.py --netcdf_ANN={ANN_default_path}/Tall.nc --netcdf_eANN={experiment_folder}/INPUT/eANN.nc --netcdf_output={experiment_folder}/INPUT/Tall.nc"')

        further_command = f'cp /scratch/pp2681/mom6/Neverworld2/simulations/R32/R2_FGR3/RESTART/MOM_0.res.nc {experiment_folder}/INPUT/MOM.res.nc'
        
        hpc = HPC.add(name=TAG, time=1, ntasks=16, mem=4, begin='1minute', executable='/scratch/pp2681/MOM6-examples/build/compiled_executables/MOM6-dev-m2lines-Aug18')
        
        run_experiment(experiment_folder, hpc, exp_params, 
            '/home/pp2681/MOM6-examples/build/configurations/NW2', call_function, further_command)
        # We save data after initializing experiment to do not interrupt workflow.
        os.makedirs(f'{experiment_folder}/INPUT', exist_ok=True)
        weights_netcdf.astype('float32').to_netcdf(f'{experiment_folder}/INPUT/eANN.nc')

        print('Experiments are scheduled')
        print('Putting in a queue resubmission script')
        os.system(commandline)
        print('Exiting the script')
        sys.exit(0)   # terminate immediately without error code

# Run final experiement
exp_params = exp_params.add(DAYMAX=2000., MAXTRUNC=100000, ZB2020_ANN_FILE_TALL=f'../iteration-{(N_iterations-1):02d}/ens-member-00/INPUT/Tall.nc')

hpc = HPC.add(name=TAG, time=6, ntasks=32, mem=16, begin='1minute', executable='/scratch/pp2681/MOM6-examples/build/compiled_executables/MOM6-dev-m2lines-Aug18')
experiment_folder = f'{base_path}/{optimization_folder}/final_experiment'
further_command = f'cp /scratch/pp2681/mom6/Neverworld2/simulations/R32/R2_FGR3/RESTART/MOM_0.res.nc {experiment_folder}/INPUT/MOM.res.nc'
run_experiment(experiment_folder, hpc, exp_params, 
                '/home/pp2681/MOM6-examples/build/configurations/NW2', '', further_command)
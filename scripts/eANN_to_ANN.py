import sys
sys.path.append('/home/pp2681/ANN-momentum-mesoscale/src/training-on-CM2.6')

from helpers.ann_tools import ANN_equivariant, equivariant_to_regular_ANN, export_ANN, import_ANN
import argparse
import xarray as xr
import torch

'''
Usage:
singularity exec --nv --overlay /scratch/$USER/python-container/python-overlay.ext3:ro --bind /scratch/pp2681/python-container/escnn-cache:/ext3/miniconda3/lib/python3.11/site-packages/escnn/group/_cache/ /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; time python -u eANN_to_ANN.py"
'''
if __name__ == '__main__':
    ########## Manual input of parameters ###############
    parser = argparse.ArgumentParser()
    parser.add_argument('--netcdf_ANN', type=str, default='/scratch/pp2681/mom6/CM26_ML_models/ocean3d/subfilter/FGR3/equivariant/16-N8-forcing-and-fluxes/model/Tall.nc')
    parser.add_argument('--netcdf_eANN', type=str, default='/scratch/pp2681/mom6/CM26_ML_models/ocean3d/subfilter/FGR3/equivariant/16-N8-forcing-and-fluxes/model/weights_and_perturbations.nc')
    parser.add_argument('--netcdf_output', type=str, default='test.nc')

    args = parser.parse_args()

    ann_regular = xr.open_dataset(args.netcdf_ANN).load()
    ann_equivariant = import_ANN(args.netcdf_ANN)
    
    weights_netcdf = xr.open_dataset(args.netcdf_eANN).astype('float32')
    ann_equivariant.model[0].weights.data = torch.tensor(weights_netcdf['weights1'].values)
    ann_equivariant.model[2].weights.data = torch.tensor(weights_netcdf['weights2'].values)
    ann_equivariant.model[0].bias.data = torch.tensor(weights_netcdf['biases1'].values)
    ann_equivariant.model[2].bias.data = torch.tensor(weights_netcdf['biases2'].values)

    export_ANN(ann_equivariant, torch.tensor(ann_regular.input_norms.values), torch.tensor(ann_regular.output_norms.values), args.netcdf_output)
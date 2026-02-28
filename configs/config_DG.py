tag: "gp2"

paths:
  base: "/scratch/pp2681/mom6/CM26_Double_Gyre/calibration/20-years"
  optimization_folder: "R2_FGR3_gprime_2"
  ann: "/scratch/pp2681/mom6/CM26_ML_models/ocean3d/subfilter/FGR3/equivariant/learning_rate/N8-forcing-fluxes/0.05/model/"
  observation: "/home/pp2681/calibration/scripts/R32/R2_FGR3_PCA_variability3.nc"
  executable: "/scratch/pp2681/MOM6-examples/build/compiled_executables/MOM6-dev-m2lines-Aug18"
  configuration: "/home/pp2681/MOM6-examples/build/configurations/double_gyre_long"
  #noise_model: "/scratch/pp2681/mom6/CM26_Double_Gyre/calibration/20-years/noise_model/metrics_00.nc"
  prior_cov: "None"
  noise_model: "None"

eki:
  n_iterations: 10
  ens_size: 100
  ens_spread: 0.25
  seed: 9
  seed_julia: 11
  outlier_scale: 10.
  diag_noise_trace_ratio: 1.
  svd_noise_trace_ratio: 0.
  sigma2: 1.0
  inversion: "TransformInversion()"
  scheduler: "DefaultScheduler(2.)"
  trainable_parameters: ["weights2", "biases2"]
  trainable_parameters_mom6:
    ZB_SCALING:
      mean: 1.0
      std: 0.25
  parameter_mask: "None"
    #  #weights2:
    #    [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0.]
  ave_start_day: 3650
  observation_vector: ["e_mean", "e_std"]
  gamma_vector: ["e_mean_var_PCA", "e_std_var_PCA"]
  observation_validation: ["e_mean", "e_std", "KE_mean", "KE_std", "APE_mean", "APE_std", "power_PCA_sqrt", "covariance_matrix_4"]
  gamma_validation: ["e_mean_var_PCA", "e_std_var_PCA", "KE_mean_var_loss", "KE_std_var_loss", "APE_mean_var_loss", "APE_std_var_loss", "power_PCA_sqrt_var", "covariance_matrix_4_var"]
  metrics_function: "return_climate_metrics"

mom6_namelist:
  DAYMAX: 7300.0
  NIGLOBAL: 44
  NJGLOBAL: 40
  DT: 2160.
  DT_FORCING: 2160.
  LAPLACIAN: False
  BIHARMONIC: True
  SMAGORINSKY_AH: True
  SMAG_BI_CONST: 0.06 
  USE_ZB2020: "True"
  ZB2020_USE_ANN: "True"
  ZB2020_ANN_FILE_TALL: "INPUT/Tall.nc"
  USE_CIRCULATION_IN_HORVISC: "True"
  ZB_SCALING: 1.0
  U_TRUNC_FILE: "U_velocity_truncations"
  V_TRUNC_FILE: "V_velocity_truncations"

slurm_eki: "sbatch --time=02:00:00 --cpus-per-task=4 --mem=64GB"

slurm_mom6:
  time: 1
  time_minutes: 0
  nodes: 1
  ntasks: 1
  mem: 0.5
  partition: "#SBATCH --partition=cs"

singularity_command: "singularity exec --nv --overlay /scratch/$USER/python-container/python-overlay.ext3:ro --bind /scratch/pp2681/python-container/escnn-cache:/ext3/miniconda3/lib/python3.11/site-packages/escnn/group/_cache/ /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif"

    #singularity_command: "singularity exec --nv --overlay /scratch/$USER/python-container/python-overlay.ext3:ro --bind /scratch/pp2681/python-container/escnn-cache:/ext3/miniconda3/lib/python3.11/site-packages/escnn/group/_cache/ /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif "

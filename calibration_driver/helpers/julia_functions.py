from julia import Main
import os
import numpy as np
import xarray as xr
from helpers.parameters import *

def initialize_eki(ANN_netcdf_default, observation_netcdf, config, optimization_folder_pwd):

    Main.eval(f"""
        using EnsembleKalmanProcesses, Random     
        using LinearAlgebra
        Random.seed!({config["eki"]["seed_julia"]})   # Fix random numbers globally
    """)

    ############### Create initial ensemble ################
    np.random.seed(config["eki"]["seed"])
    initial_ensemble, num_of_parameters = generate_ensemble(ANN_netcdf_default, 
                                            config["eki"]["trainable_parameters"],
                                            config["eki"]["trainable_parameters_mom6"],
                                            config["eki"]["ens_spread"],
                                            config["eki"]["ens_size"],
                                            config["paths"]["prior_cov"],
                                            config["eki"]["parameter_mask"])

    # Length of the observational vector
    len_obs = np.sum([observation_netcdf[metric].size for metric in config["eki"]["observation_vector"]])
    print("Length of the observational vector", len_obs)

    eki_state_file = f'{optimization_folder_pwd}/eki_state.jls'
    rng_state_file = f'{optimization_folder_pwd}/rng.jls'
    if os.path.exists(eki_state_file) and os.path.exists(rng_state_file):
        print('Reading EKI state from file')
        Main.eki_state_file = eki_state_file
        Main.rng_state_file = rng_state_file
        Main.eval("""
                using Serialization
                eki = deserialize(eki_state_file)
                copy!(Random.default_rng(), deserialize(rng_state_file))
                """)
    else:
        print('Initializing EKI from scratch')                                        
        ############ Prepare observational vector ##############
        observation_vector = []
        for key in config["eki"]["observation_vector"]:
            observation_vector.append(observation_netcdf[key].values.ravel())
        observation_vector = np.concatenate(observation_vector)

        ################ Prepare scalar product ################
        gamma_vector = []
        for key in config["eki"]["gamma_vector"]:
            gamma_vector.append(observation_netcdf[key].values.ravel())
        gamma_vector = np.concatenate(gamma_vector)

        # Construct the observational vector in rescaled space using
        # the defined scalar product
        Main.observation_vector = observation_vector / np.sqrt(gamma_vector)
        Main.initial_ensemble = initial_ensemble

        ############### Create noise model #####################
        if os.path.exists(config["paths"]["noise_model"]):
            print("Reading noise model from file")
            noise_model = xr.open_dataset(config["paths"]["noise_model"]).isel(iter=0).load()
            ens_size = len(noise_model.ens)

            # Create forward model evaluation matrix
            noise_ens = np.full(
                (len_obs, ens_size),
                np.nan,
                dtype="float64"
            )

            for ens_member in range(ens_size):
                noise_ens[:,ens_member] = np.concatenate([
                    noise_model[metric].isel(ens=ens_member).values.ravel() / np.sqrt(observation_netcdf[gamma].values.ravel()) 
                    for metric, gamma in zip(config["eki"]["observation_vector"], config["eki"]["gamma_vector"])])

            # Remove mean to make sure that we analyze fluctuations
            noise_ens = noise_ens - noise_ens.mean(1,keepdims=True)

            trace_noise_model = (noise_ens**2).sum(0).mean()
            diag_noise_trace_ratio = config["eki"]["diag_noise_trace_ratio"]
            svd_noise_trace_ratio = config["eki"]["svd_noise_trace_ratio"]
            print('Trace of the noise model before covariance inflation:', trace_noise_model)
            print('Trace of the noise model after covariance inflation:', trace_noise_model * (svd_noise_trace_ratio + diag_noise_trace_ratio))
            # Desired regularization parameter is found by equality of traces
            # On the right is the trace of identity operator with scaling coefficient alpha
            # trace_noise_model * noise_trace_ratio = alpha * len_obs
            alpha = trace_noise_model * diag_noise_trace_ratio / len_obs
            print("Alpha variance inflation parameter", alpha)

            Main.noise_ens = noise_ens
            Main.alpha = alpha
            Main.ones_vector = np.ones_like(observation_vector)

            # Here we compute the covariance matrix using SVD
            # with help of standard EnsembleKalmanProcesses.jl workflow
            # We also make sure to add the identity variance inflation with the
            # given trace
            # Main.eval("""
            # internal_cov = tsvd_cov_from_samples(noise_ens)
            # background_noise = ones_vector * alpha
            # covariance = SVDplusD(internal_cov, Diagonal(background_noise));
            # """)
            # We compute SVD using numpy as it is 10000 times faster
            if svd_noise_trace_ratio > 1.e-16:
                U, s, Vt = np.linalg.svd(noise_ens, full_matrices=False)

                # We reduce the number of degrees of freedom by one as it is done in 
                # EnsembleKalmanProcesses.jl
                eigvals = s**2 / (ens_size-1)
                Main.U = U[:,:-1]
                Main.eigvals = eigvals[:-1] * svd_noise_trace_ratio
                Main.eval("""
                internal_cov = SVD(U, eigvals, U')
                background_noise = ones_vector * alpha
                covariance = SVDplusD(internal_cov, Diagonal(background_noise));
                """)
                print('Computation of noise covariance matrix in SVD form is finished')
            else:
                Main.eval("""
                background_noise = ones_vector * alpha
                covariance = Diagonal(background_noise)
                """)
                print('Computation of noise covariance matrix in diagonal form is finished')
        else:
            Main.covariance = np.ones_like(observation_vector)
            Main.eval("covariance = Diagonal(covariance)")

        Main.eval(f"""
            observation = Observation(Dict(
            "samples" => observation_vector,
            "covariances" => covariance,
            "names" => "why_are_you_asking_my_name"
            ))  
            eki = EnsembleKalmanProcess(
            initial_ensemble, observation, {config["eki"]["inversion"]},
            scheduler = {config["eki"]["scheduler"]},
            accelerator = DefaultAccelerator(),
            localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
            verbose=true)
            """)
    
    return len_obs, num_of_parameters
    
def save_eki_on_disk(optimization_folder_pwd):
    eki_state_file = f'{optimization_folder_pwd}/eki_state.jls'
    rng_state_file = f'{optimization_folder_pwd}/rng.jls'
    Main.eki_state_file = eki_state_file
    Main.rng_state_file = rng_state_file

    os.system(f'rm -f {eki_state_file}')
    os.system(f'rm -f {rng_state_file}')
    Main.eval("""
            using Serialization
            serialize(eki_state_file, eki)
            serialize(rng_state_file, copy(Random.default_rng()))
        """)

def eki_get_params():
    return Main.eval("get_u_final(eki)")

def eki_update_ensemble(g_ens):
    Main.g_ens = g_ens
    return Main.eval("update_ensemble!(eki, g_ens)")

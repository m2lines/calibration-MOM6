from julia import Main
import os
def initialize_eki(observation_vector, gamma_vector, initial_ensemble, scheduler, inversion, seed, optimization_folder_pwd):
    Main.observation_vector = observation_vector
    Main.gamma_vector = gamma_vector
    Main.initial_ensemble = initial_ensemble

    Main.eval(f"""
        using EnsembleKalmanProcesses, Random     
        using LinearAlgebra
        Random.seed!({seed})   # Fix random numbers globally
    """)

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
        Main.eval(f"""
            eki = EnsembleKalmanProcess(
            initial_ensemble, observation_vector, Diagonal(gamma_vector), {inversion},
            scheduler = {scheduler},
            accelerator = DefaultAccelerator(),
            localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
            verbose=true)
            """)
    
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
    Main.eval("update_ensemble!(eki, g_ens)")
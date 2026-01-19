from julia import Main
def initialize_eki(observation_vector, gamma_vector, initial_ensemble, scheduler, inversion, seed):
    Main.observation_vector = observation_vector
    Main.gamma_vector = gamma_vector
    Main.initial_ensemble = initial_ensemble

    Main.eval(f"""
        using EnsembleKalmanProcesses, Random     
        using LinearAlgebra   
        Random.seed!({seed})   # Fix random numbers globally
    """)

    Main.eval(f"""
        eki = EnsembleKalmanProcess(
        initial_ensemble, observation_vector, Diagonal(gamma_vector), {inversion},
        scheduler = {scheduler},
        accelerator = DefaultAccelerator(),
        localization_method = EnsembleKalmanProcesses.Localizers.NoLocalization(),
        verbose=true)
        """)

def eki_get_params():
    return Main.eval("get_u_final(eki)")

def eki_update_ensemble(g_ens):
    Main.g_ens = g_ens
    Main.eval("update_ensemble!(eki, g_ens)")
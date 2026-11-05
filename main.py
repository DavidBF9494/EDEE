from config import config

config.config_lammps_routes()

from src.EDEE_MolecularDynamicSimulation import EDEE_MolecularDynamicSimulation
from src.EDEE_Parameters import EDEE_Parameters

from lammps.formats import LogFile
from pandas import DataFrame
from pathlib import Path
from pickle import load

if __name__ == "__main__":
    # log = LogFile("data/log.lammps")
    # log_run = log.runs
    # log_run_df = DataFrame(log_run[1])
    
    n_sim               = 5
    t_nvt               = 50000
    t_restart_nvt       = 2000
    t_nve               = 50000
    t_eq                = 50000
    t_neq               = 50000
    dt_nvt              = 100
    dt_nve              = 100
    dt_eq               = 1
    dt_neq              = 100
    temp                = 300.0
    omega               = 0.0085
    
    EDEE_MDS            = EDEE_MolecularDynamicSimulation(n_sim, t_nvt, t_restart_nvt, t_nve, t_eq, t_neq, dt_nvt, dt_nve, dt_eq, dt_neq, temp, omega)
    
    results_nvt         = EDEE_MDS.get('results_nvt')
    results_nve         = EDEE_MDS.get('results_nve')
    results_eq          = EDEE_MDS.get('results_eq')
    results_neq         = EDEE_MDS.get('results_neq')
    
    # EDEE_MDS.plot_df_data(results_neq[1]['df_logs'][0], 'Step', 'Temp', True)
    # EDEE_MDS.plot_df_data(results_neq[1]['df_logs'][0], 'Step', 'PotEng', True)
    # EDEE_MDS.plot_df_data(results_neq[1]['df_logs'][0], 'Step', 'KinEng', True)
    # EDEE_MDS.plot_df_data(results_neq[1]['df_logs'][0], 'Step', 'TotEng', True)
    # EDEE_MDS.plot_df_data(results_neq[1]['df_logs'][0], 'Step', 'Press', True)
    # EDEE_MDS.plot_df_data(results_neq[1]['df_logs'][0], 'Step', 'v_rCM', True)
    # EDEE_MDS.plot_df_data(results_neq[1]['df_logs'][0], 'Step', 'c_myRg', True)
    # EDEE_MDS.plot_df_data(results_neq[1]['df_custom'], 'Step', 'RotKinEng', False)
    # EDEE_MDS.plot_df_data(results_neq[1]['df_custom'], 'Step', 'DilKinEng', False)
    # EDEE_MDS.plot_df_data(results_neq[1]['df_custom'], 'Step', 'TransKinEng', False)
    
    # EDEE_MDS.plot_positions(
    #     results_neq[1]['masses'], 
    #     results_neq[1]['positions'], 
    #     results_neq[1]['gyration_tensor_eigenvectors'], 
    #     results_neq[1]['angular_momentum'], 
    #     results_neq[1]['angular_velocity']
    # )
    
    # directory_variables = Path("variables")
    # def load_variable(name):
    #     with open(directory_variables / f"{name}.pkl", "rb") as f:
    #         return load(f)
    # results_nvt         = load_variable("results_nvt")
    # results_nve         = load_variable("results_nve")
    # results_eq          = load_variable("results_eq")
    # results_neq         = load_variable("results_neq")
    
    EDEE_PRMS           = EDEE_Parameters(t_nvt, t_restart_nvt, t_nve, t_eq, t_neq, dt_nvt, dt_nve, dt_eq, dt_neq, temp, results_nvt, results_nve, results_eq, results_neq)
    
    parameters_nvt      = EDEE_PRMS.get('parameters_nvt')
    parameters_nve      = EDEE_PRMS.get('parameters_nve')
    parameters_eq       = EDEE_PRMS.get('parameters_eq')
    parameters_neq      = EDEE_PRMS.get('parameters_neq')
    
    # EDEE_PRMS.plot_df_data(parameters_eq['df_custom'], 'Step', '⟨M1(t) M1⟩', False)
    # EDEE_PRMS.plot_df_data(parameters_eq['df_custom'], 'Step', '⟨M2(t) M2⟩', False)
    # EDEE_PRMS.plot_df_data(parameters_eq['df_custom'], 'Step', '⟨M3(t) M3⟩', False)
    
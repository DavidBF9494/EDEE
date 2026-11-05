from matplotlib.pyplot import figure, grid, legend, xlabel, ylabel, title, plot, scatter, show
from numpy.fft import rfft, irfft, rfftfreq
from numpy import array, zeros, ndarray, linalg, diag, mean, sqrt, abs, exp, cos, arange, insert, argmax, stack, column_stack, savetxt
from pandas import DataFrame
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from .EDEE_Auxiliary import EDEE_Auxiliary

class EDEE_Parameters(EDEE_Auxiliary):
    
    def __init__(self, t_nvt: int, t_restart_nvt: int, t_nve: int, t_eq: int, t_neq: int, dt_nvt: int, dt_nve: int, dt_eq: int, dt_neq: int, temp: float, results_nvt: dict, results_nve: dict, results_eq: dict, results_neq: dict) -> None:
        super().__init__()
        
        self.__t_nvt = t_nvt
        self.__t_restart_nvt = t_restart_nvt
        self.__t_nve = t_nve
        self.__t_eq = t_eq
        self.__t_neq = t_neq
        self.__dt_nvt = dt_nvt
        self.__dt_nve = dt_nve
        self.__dt_eq = dt_eq
        self.__dt_neq = dt_neq
        self.__temp = temp
        self.__results_nvt = results_nvt
        self.__results_nve = results_nve
        self.__results_eq = results_eq
        self.__results_neq = results_neq
        
        self.__kB = 1.9872e-3
        self.__beta = 1 / (self.__temp * self.__kB)
        
        self.__calculate_EDEE_parameters__()
        return None
        
    def __calculate_EDEE_parameters__(self) -> None:
        print("\n<-- EDEE_Parameters - Cálculo Parámetros EDEE -->")
        print("\n<- Resultados NVT ->\n")
        print("Calculando parámetros...")
        self.__parameters_nvt = {}
        self.__parameters_nvt["df_custom"] = DataFrame({'Step': range(int(self.__t_restart_nvt / self.__dt_nvt + 1))})
        average_L_nvt = self.__calculate_average_L(self.__results_nvt, self.__t_restart_nvt, self.__dt_nvt, self.__parameters_nvt["df_custom"], key = "nvt")
        average_w_nvt = self.__calculate_average_w(self.__results_nvt, self.__t_restart_nvt, self.__dt_nvt, self.__parameters_nvt["df_custom"], key = "nvt")
        average_w_principal_axis_nvt = self.__calculate_average_w_principal_axis(self.__results_nvt, self.__t_restart_nvt, self.__dt_nvt, self.__parameters_nvt["df_custom"], key = "nvt")
        average_M_nvt = self.__calculate_average_M(self.__results_nvt, self.__t_restart_nvt, self.__dt_nvt, self.__parameters_nvt["df_custom"], key = "nvt")
        # average_M_rest_nvt = self.__calculate_average_M_rest(self.__results_nvt, self.__t_restart_nvt, self.__dt_nvt)
        # elasticity_matrix_SIGMA_nvt = self.__calculate_elasticity_matrix_SIGMA(self.__results_nvt, average_M_rest_nvt, self.__t_restart_nvt, self.__dt_nvt)
        # time_correlation_function_M_nvt = self.__calculate_time_correlation_function_M(self.__results_nvt, average_M_rest_nvt, self.__t_restart_nvt, self.__dt_nvt, self.__parameters_nvt["df_custom"], key = "nvt")
        # self.__calculate_trajectory_time_correlation_function_M(self.__results_nvt, average_M_rest_nvt, self.__t_restart_nvt, self.__dt_nvt, self.__parameters_nvt["df_custom"], key = "nvt")
        # time_correlation_matrix_eigenvectors_E_nvt = self.__calculate_time_correlation_matrix_eigenvectors_E(self.__results_nvt, self.__t_nvt, self.__dt_nvt, self.__parameters_nvt["df_custom"], key = "nvt")
        # self.__calculate_trajectory_time_correlation_matrix_eigenvectors_E(self.__results_nvt, self.__t_restart_nvt_, self.__dt_nvt, self.__parameters_nvt["df_custom"], key = "nvt")
        # dissipative_matrix_GAMMA_nvt, oscillation_frecuencies_nvt = self.__calculate_fitting_time_correlation_function_M_Ornstein_Ulhenbeck(average_M_rest_nvt, elasticity_matrix_SIGMA_nvt, time_correlation_function_M_nvt, self.__dt_nvt, t_int = 20000, key = "nvt")
        # orientational_diffusion_matrix_D0_nvt = self.__calculate_fitting_time_correlation_matrix_eigenvectors_E(time_correlation_matrix_eigenvectors_E_nvt, self.__dt_nvt, t_int = 10000, key = "nvt")
        average_rotational_kinetic_energy_nvt = self.__calculate_average_kinetic_energy(self.__results_nvt, self.__t_restart_nvt, self.__dt_nvt, self.__parameters_nvt["df_custom"], key = "nvt:rotational_kinetic_energy")
        average_dilational_kinetic_energy_nvt = self.__calculate_average_kinetic_energy(self.__results_nvt, self.__t_restart_nvt, self.__dt_nvt, self.__parameters_nvt["df_custom"], key = "nvt:dilational_kinetic_energy")
        average_translational_kinetic_energy_nvt = self.__calculate_average_kinetic_energy(self.__results_nvt, self.__t_restart_nvt, self.__dt_nvt, self.__parameters_nvt["df_custom"], key = "nvt:translational_kinetic_energy")
        self.__parameters_nvt["average_L"] = average_L_nvt
        self.__parameters_nvt["average_w"] = average_w_nvt
        self.__parameters_nvt["average_w_principal_axis"] = average_w_principal_axis_nvt
        self.__parameters_nvt["average_M"] = average_M_nvt
        # self.__parameters_nvt["average_M_rest"] = average_M_rest_nvt
        # self.__parameters_nvt["elasticity_matrix_SIGMA"] = elasticity_matrix_SIGMA_nvt
        # self.__parameters_nvt["time_correlation_function_M"] = time_correlation_function_M_nvt
        # self.__parameters_nvt["time_correlation_matrix_eigenvectors_E"] = time_correlation_matrix_eigenvectors_E_nvt
        # self.__parameters_nvt["dissipative_matrix_GAMMA"] = dissipative_matrix_GAMMA_nvt
        # self.__parameters_nvt["oscillation_frecuencies"] = oscillation_frecuencies_nvt
        # self.__parameters_nvt["orientational_diffusion_matrix_D0"] = orientational_diffusion_matrix_D0_nvt
        self.__parameters_nvt["average_rotational_kinetic_energy"] = average_rotational_kinetic_energy_nvt
        self.__parameters_nvt["average_dilational_kinetic_energy"] = average_dilational_kinetic_energy_nvt
        self.__parameters_nvt["average_translational_kinetic_energy"] = average_translational_kinetic_energy_nvt
        
        print("\n<- Resultados NVE ->\n")
        print("Calculando parámetros...")
        self.__parameters_nve = {}
        self.__parameters_nve["df_custom"] = DataFrame({'Step': range(int(self.__t_nve / self.__dt_nve + 1))})
        average_L_nve = self.__calculate_average_L(self.__results_nve, self.__t_nve, self.__dt_nve, self.__parameters_nve["df_custom"], key = "nve")
        average_w_nve = self.__calculate_average_w(self.__results_nve, self.__t_nve, self.__dt_nve, self.__parameters_nve["df_custom"], key = "nve")
        average_w_principal_axis_nve = self.__calculate_average_w_principal_axis(self.__results_nve, self.__t_nve, self.__dt_nve, self.__parameters_nve["df_custom"], key = "nve")
        average_M_nve = self.__calculate_average_M(self.__results_nve, self.__t_nve, self.__dt_nve, self.__parameters_nve["df_custom"], key = "nve")
        # average_M_rest_nve = self.__calculate_average_M_rest(self.__results_nve, self.__t_nve, self.__dt_nve)
        # elasticity_matrix_SIGMA_nve = self.__calculate_elasticity_matrix_SIGMA(self.__results_nve, average_M_rest_nve, self.__t_nve, self.__dt_nve)
        # time_correlation_function_M_nve = self.__calculate_time_correlation_function_M(self.__results_nve, average_M_rest_nve, self.__t_nve, self.__dt_nve, self.__parameters_nve["df_custom"], key = "nve")
        # self.__calculate_trajectory_time_correlation_function_M(self.__results_nve, average_M_rest_nve, self.__t_nve, self.__dt_nve, self.__parameters_nve["df_custom"], key = "nve")
        # time_correlation_matrix_eigenvectors_E_nve = self.__calculate_time_correlation_matrix_eigenvectors_E(self.__results_nve, self.__t_nve, self.__dt_nve, self.__parameters_nve["df_custom"], key = "nve")
        # self.__calculate_trajectory_time_correlation_matrix_eigenvectors_E(self.__results_nve, self.__t_nve, self.__dt_nve, self.__parameters_nve["df_custom"], key = "nve")
        # dissipative_matrix_GAMMA_nve, oscillation_frecuencies_nve = self.__calculate_fitting_time_correlation_function_M_Ornstein_Ulhenbeck(average_M_rest_nve, elasticity_matrix_SIGMA_nve, time_correlation_function_M_nve, self.__dt_nve, t_int = 20000, key = "nve")
        # orientational_diffusion_matrix_D0_nve = self.__calculate_fitting_time_correlation_matrix_eigenvectors_E(time_correlation_matrix_eigenvectors_E_nve, self.__dt_nve, t_int = 10000, key = "nve")
        average_rotational_kinetic_energy_nve = self.__calculate_average_kinetic_energy(self.__results_nve, self.__t_nve, self.__dt_nve, self.__parameters_nve["df_custom"], key = "nve:rotational_kinetic_energy")
        average_dilational_kinetic_energy_nve = self.__calculate_average_kinetic_energy(self.__results_nve, self.__t_nve, self.__dt_nve, self.__parameters_nve["df_custom"], key = "nve:dilational_kinetic_energy")
        average_translational_kinetic_energy_nve = self.__calculate_average_kinetic_energy(self.__results_nve, self.__t_nve, self.__dt_nve, self.__parameters_nve["df_custom"], key = "nve:translational_kinetic_energy")
        self.__parameters_nve["average_L"] = average_L_nve
        self.__parameters_nve["average_w"] = average_w_nve
        self.__parameters_nve["average_w_principal_axis"] = average_w_principal_axis_nve
        self.__parameters_nve["average_M"] = average_M_nve
        # self.__parameters_nve["average_M_rest"] = average_M_rest_nve
        # self.__parameters_nve["elasticity_matrix_SIGMA"] = elasticity_matrix_SIGMA_nve
        # self.__parameters_nve["time_correlation_function_M"] = time_correlation_function_M_nve
        # self.__parameters_nve["time_correlation_matrix_eigenvectors_E"] = time_correlation_matrix_eigenvectors_E_nve
        # self.__parameters_nve["dissipative_matrix_GAMMA"] = dissipative_matrix_GAMMA_nve
        # self.__parameters_nve["oscillation_frecuencies"] = oscillation_frecuencies_nve
        # self.__parameters_nve["orientational_diffusion_matrix_D0"] = orientational_diffusion_matrix_D0_nve
        self.__parameters_nve["average_rotational_kinetic_energy"] = average_rotational_kinetic_energy_nve
        self.__parameters_nve["average_dilational_kinetic_energy"] = average_dilational_kinetic_energy_nve
        self.__parameters_nve["average_translational_kinetic_energy"] = average_translational_kinetic_energy_nve
        
        print("\n<- Resultados Equilibrio ->\n")
        print("Calculando parámetros...")
        self.__parameters_eq = {}
        self.__parameters_eq["df_custom"] = DataFrame({'Step': range(int(self.__t_eq / self.__dt_eq + 1))})
        average_L_eq = self.__calculate_average_L(self.__results_eq, self.__t_eq, self.__dt_eq, self.__parameters_eq["df_custom"], key = "eq")
        average_w_eq = self.__calculate_average_w(self.__results_eq, self.__t_eq, self.__dt_eq, self.__parameters_eq["df_custom"], key = "eq")
        average_w_principal_axis_eq = self.__calculate_average_w_principal_axis(self.__results_eq, self.__t_eq, self.__dt_eq, self.__parameters_eq["df_custom"], key = "eq")
        average_M_eq = self.__calculate_average_M(self.__results_eq, self.__t_eq, self.__dt_eq, self.__parameters_eq["df_custom"], key = "eq")
        # average_M_rest_eq = self.__calculate_average_M_rest(self.__results_eq, self.__t_eq, self.__dt_eq)
        # elasticity_matrix_SIGMA_eq = self.__calculate_elasticity_matrix_SIGMA(self.__results_eq, average_M_rest_eq, self.__t_eq, self.__dt_eq)
        # time_correlation_function_M_eq = self.__calculate_time_correlation_function_M(self.__results_eq, average_M_rest_eq, self.__t_eq, self.__dt_eq, self.__parameters_eq["df_custom"], key = "eq")
        # self.__calculate_trajectory_time_correlation_function_M(self.__results_eq, average_M_rest_eq, self.__t_eq, self.__dt_eq, self.__parameters_eq["df_custom"], key = "eq")
        # time_correlation_matrix_eigenvectors_E_eq = self.__calculate_time_correlation_matrix_eigenvectors_E(self.__results_eq, self.__t_eq, self.__dt_eq, self.__parameters_eq["df_custom"], key = "eq")
        # self.__calculate_trajectory_time_correlation_matrix_eigenvectors_E(self.__results_eq, self.__t_eq, self.__dt_eq, self.__parameters_eq["df_custom"], key = "eq")
        # dissipative_matrix_GAMMA_eq, oscillation_frecuencies_eq = self.__calculate_fitting_time_correlation_function_M_Ornstein_Ulhenbeck(average_M_rest_eq, elasticity_matrix_SIGMA_eq, time_correlation_function_M_eq, self.__dt_eq, t_int = 10000, key = "eq")
        # orientational_diffusion_matrix_D0_eq = self.__calculate_fitting_time_correlation_matrix_eigenvectors_E(time_correlation_matrix_eigenvectors_E_eq, self.__dt_eq, t_int = 150000, key = "eq")
        average_rotational_kinetic_energy_eq = self.__calculate_average_kinetic_energy(self.__results_eq, self.__t_eq, self.__dt_eq, self.__parameters_eq["df_custom"], key = "eq:rotational_kinetic_energy")
        average_dilational_kinetic_energy_eq = self.__calculate_average_kinetic_energy(self.__results_eq, self.__t_eq, self.__dt_eq, self.__parameters_eq["df_custom"], key = "eq:dilational_kinetic_energy")
        average_translational_kinetic_energy_eq = self.__calculate_average_kinetic_energy(self.__results_eq, self.__t_eq, self.__dt_eq, self.__parameters_eq["df_custom"], key = "eq:translational_kinetic_energy")
        self.__parameters_eq["average_L"] = average_L_eq
        self.__parameters_eq["average_w"] = average_w_eq
        self.__parameters_eq["average_w_principal_axis"] = average_w_principal_axis_eq
        self.__parameters_eq["average_M"] = average_M_eq
        # self.__parameters_eq["average_M_rest"] = average_M_rest_eq
        # self.__parameters_eq["elasticity_matrix_SIGMA"] = elasticity_matrix_SIGMA_eq
        # self.__parameters_eq["time_correlation_function_M"] = time_correlation_function_M_eq
        # self.__parameters_eq["time_correlation_matrix_eigenvectors_E"] = time_correlation_matrix_eigenvectors_E_eq
        # self.__parameters_eq["dissipative_matrix_GAMMA"] = dissipative_matrix_GAMMA_eq
        # self.__parameters_eq["oscillation_frecuencies"] = oscillation_frecuencies_eq
        # self.__parameters_eq["orientational_diffusion_matrix_D0"] = orientational_diffusion_matrix_D0_eq
        self.__parameters_eq["average_rotational_kinetic_energy"] = average_rotational_kinetic_energy_eq
        self.__parameters_eq["average_dilational_kinetic_energy"] = average_dilational_kinetic_energy_eq
        self.__parameters_eq["average_translational_kinetic_energy"] = average_translational_kinetic_energy_eq
        
        print("\n<- Resultados No Equilibrio ->\n")
        print("Calculando parámetros...")
        self.__parameters_neq = {}
        self.__parameters_neq["df_custom"] = DataFrame({'Step': range(int(self.__t_neq / self.__dt_neq + 1))})
        average_L_neq = self.__calculate_average_L(self.__results_neq, self.__t_neq, self.__dt_neq, self.__parameters_neq["df_custom"], key = "neq")
        average_w_neq = self.__calculate_average_w(self.__results_neq, self.__t_neq, self.__dt_neq, self.__parameters_neq["df_custom"], key = "neq")
        average_w_principal_axis_neq = self.__calculate_average_w_principal_axis(self.__results_neq, self.__t_neq, self.__dt_neq, self.__parameters_neq["df_custom"], key = "neq")
        average_M_neq = self.__calculate_average_M(self.__results_neq, self.__t_neq, self.__dt_neq, self.__parameters_neq["df_custom"], key = "neq")
        # average_M_rest_neq = self.__calculate_average_M_rest(self.__results_neq, self.__t_neq, self.__dt_neq)
        # elasticity_matrix_SIGMA_neq = self.__calculate_elasticity_matrix_SIGMA(self.__results_neq, average_M_rest_neq, self.__t_neq, self.__dt_neq)
        # time_correlation_function_M_neq = self.__calculate_time_correlation_function_M(self.__results_neq, average_M_rest_neq, self.__t_neq, self.__dt_neq, self.__parameters_neq["df_custom"], key = "neq")
        # self.__calculate_trajectory_time_correlation_function_M(self.__results_neq, average_M_rest_neq, self.__t_neq, self.__dt_neq, self.__parameters_neq["df_custom"], key = "neq")
        # time_correlation_matrix_eigenvectors_E_neq = self.__calculate_time_correlation_matrix_eigenvectors_E(self.__results_neq, self.__t_neq, self.__dt_neq, self.__parameters_neq["df_custom"], key = "neq")
        # self.__calculate_trajectory_time_correlation_matrix_eigenvectors_E(self.__results_neq, self.__t_neq, self.__dt_neq, self.__parameters_neq["df_custom"], key = "neq")
        # dissipative_matrix_GAMMA_neq, oscillation_frecuencies_neq = self.__calculate_fitting_time_correlation_function_M_Ornstein_Ulhenbeck(average_M_rest_neq, elasticity_matrix_SIGMA_neq, time_correlation_function_M_neq, self.__dt_neq, t_int = 20000, key = "neq")
        # orientational_diffusion_matrix_D0_neq = self.__calculate_fitting_time_correlation_matrix_eigenvectors_E(time_correlation_matrix_eigenvectors_E_neq, self.__dt_neq, t_int = 10000, key = "neq")
        average_rotational_kinetic_energy_neq = self.__calculate_average_kinetic_energy(self.__results_neq, self.__t_neq, self.__dt_neq, self.__parameters_neq["df_custom"], key = "neq:rotational_kinetic_energy")
        average_dilational_kinetic_energy_neq = self.__calculate_average_kinetic_energy(self.__results_neq, self.__t_neq, self.__dt_neq, self.__parameters_neq["df_custom"], key = "neq:dilational_kinetic_energy")
        average_translational_kinetic_energy_neq = self.__calculate_average_kinetic_energy(self.__results_neq, self.__t_neq, self.__dt_neq, self.__parameters_neq["df_custom"], key = "neq:translational_kinetic_energy")
        self.__parameters_neq["average_L"] = average_L_neq
        self.__parameters_neq["average_w"] = average_w_neq
        self.__parameters_neq["average_w_principal_axis"] = average_w_principal_axis_neq
        self.__parameters_neq["average_M"] = average_M_neq
        # self.__parameters_neq["average_M_rest"] = average_M_rest_neq
        # self.__parameters_neq["elasticity_matrix_SIGMA"] = elasticity_matrix_SIGMA_neq
        # self.__parameters_neq["time_correlation_function_M"] = time_correlation_function_M_neq
        # self.__parameters_neq["time_correlation_matrix_eigenvectors_E"] = time_correlation_matrix_eigenvectors_E_neq
        # self.__parameters_neq["dissipative_matrix_GAMMA"] = dissipative_matrix_GAMMA_neq
        # self.__parameters_neq["oscillation_frecuencies"] = oscillation_frecuencies_neq
        # self.__parameters_neq["orientational_diffusion_matrix_D0"] = orientational_diffusion_matrix_D0_neq
        self.__parameters_neq["average_rotational_kinetic_energy"] = average_rotational_kinetic_energy_neq
        self.__parameters_neq["average_dilational_kinetic_energy"] = average_dilational_kinetic_energy_neq
        self.__parameters_neq["average_translational_kinetic_energy"] = average_translational_kinetic_energy_neq
        
    def __calculate_average_L(self, results: dict, t: int, dt: int, df_custom: DataFrame, key: str) -> ndarray:
        N_frames = int(t / dt + 1)
        L_arrays = []
        norm_L_arrays = []
        delta_L_arrays = []
        for _, value in results.items():
            L_k = value["angular_momentum"][:, 0, :]
            N_k = L_k.shape[0]
            if N_k != N_frames:
                continue
            L_arrays.append(L_k)
            norm_L_k = linalg.norm(L_k, axis = 1)
            norm_L_arrays.append(norm_L_k)
            delta_L_k = L_k - L_k[0]
            norm_delta_L_k = linalg.norm(delta_L_k, axis = 1)
            delta_L_arrays.append(norm_delta_L_k)
        average_L = mean(L_arrays, axis = 0)
        average_norm_L = mean(norm_L_arrays, axis = 0)
        average_delta_L = mean(delta_L_arrays, axis = 0)
        df_custom["⟨L1(t)⟩"] = average_L[:, 0]
        df_custom["⟨L2(t)⟩"] = average_L[:, 1]
        df_custom["⟨L3(t)⟩"] = average_L[:, 2]
        df_custom["⟨|L(t)|⟩"] = average_norm_L
        df_custom["⟨|L(t)-L(0)|⟩"] = average_delta_L
        data = column_stack((range(int(t / dt + 1)), average_L[:, 0], average_L[:, 1], average_L[:, 2], average_norm_L, average_delta_L))
        savetxt(f"{self._directory_txt}/{key}_average_L.dat", data, header = "Step average_L_1 average_L_2 average_L_3 average_norm_L average_delta_L")
        return average_L
    
    def __calculate_average_w(self, results: dict, t: int, dt: int, df_custom: DataFrame, key: str) -> ndarray:
        N_frames = int(t / dt + 1)
        w_arrays = []
        norm_w_arrays = []
        delta_w_arrays = []
        for _, value in results.items():
            w_k = value["angular_velocity"][:, 0, :]
            N_k = w_k.shape[0]
            if N_k != N_frames:
                continue
            w_arrays.append(w_k)
            norm_w_k = linalg.norm(w_k, axis = 1)
            norm_w_arrays.append(norm_w_k)
            delta_w_k = w_k - w_k[0]
            norm_delta_w_k = linalg.norm(delta_w_k, axis = 1)
            delta_w_arrays.append(norm_delta_w_k)
        average_w = mean(w_arrays, axis = 0)
        average_norm_w = mean(norm_w_arrays, axis = 0)
        average_delta_w = mean(delta_w_arrays, axis = 0)
        df_custom["⟨w1(t)⟩"] = average_w[:, 0]
        df_custom["⟨w2(t)⟩"] = average_w[:, 1]
        df_custom["⟨w3(t)⟩"] = average_w[:, 2]
        df_custom["⟨|w(t)|⟩"] = average_norm_w
        df_custom["⟨|w(t)-w(0)|⟩"] = average_delta_w
        data = column_stack((range(int(t / dt + 1)), average_w[:, 0], average_w[:, 1], average_w[:, 2], average_norm_w, average_delta_w))
        savetxt(f"{self._directory_txt}/{key}_average_w.dat", data, header = "Step average_w_1 average_w_2 average_w_3 average_norm_w average_delta_w")
        return average_w
    
    def __calculate_average_w_principal_axis(self, results: dict, t: int, dt: int, df_custom: DataFrame, key: str) -> ndarray:
        N_frames = int(t / dt + 1)
        w_arrays = []
        norm_w_arrays = []
        delta_w_arrays = []
        for _, value in results.items():
            w_k = value["angular_velocity_principal_axis"][:, 0, :]
            N_k = w_k.shape[0]
            if N_k != N_frames:
                continue
            w_arrays.append(w_k)
            norm_w_k = linalg.norm(w_k, axis = 1)
            norm_w_arrays.append(norm_w_k)
            delta_w_k = w_k - w_k[0]
            norm_delta_w_k = linalg.norm(delta_w_k, axis = 1)
            delta_w_arrays.append(norm_delta_w_k)
        average_w = mean(w_arrays, axis = 0)
        average_norm_w = mean(norm_w_arrays, axis = 0)
        average_delta_w = mean(delta_w_arrays, axis = 0)
        df_custom["⟨w1(t)⟩"] = average_w[:, 0]
        df_custom["⟨w2(t)⟩"] = average_w[:, 1]
        df_custom["⟨w3(t)⟩"] = average_w[:, 2]
        df_custom["⟨|w(t)|⟩"] = average_norm_w
        df_custom["⟨|w(t)-w(0)|⟩"] = average_delta_w
        data = column_stack((range(int(t / dt + 1)), average_w[:, 0], average_w[:, 1], average_w[:, 2], average_norm_w, average_delta_w))
        savetxt(f"{self._directory_txt}/{key}_average_w_principal_axis.dat", data, header = "Step average_w_1 average_w_2 average_w_3 average_norm_w average_delta_w")
        return average_w
        
    def __calculate_average_M(self, results: dict, t: int, dt: int, df_custom: DataFrame, key: str) -> ndarray:
        N_frames = int(t / dt + 1)
        M_arrays = []
        for _, value in results.items():
            M_k = value["gyration_tensor_eigenvalues"]
            N_k = M_k.shape[0]
            if N_k != N_frames:
                continue
            M_arrays.append(M_k)
        average_M = mean(M_arrays, axis = 0)
        df_custom["⟨M1(t)⟩"] = average_M[:, 0]
        df_custom["⟨M2(t)⟩"] = average_M[:, 1]
        df_custom["⟨M3(t)⟩"] = average_M[:, 2]
        data = column_stack((range(int(t / dt + 1)), average_M[:, 0], average_M[:, 1], average_M[:, 2]))
        savetxt(f"{self._directory_txt}/{key}_average_M.dat", data, header = "Step average_M_1 average_M_2 average_M_3")
        return average_M
        
    def __calculate_average_M_rest(self, results: dict, t: int, dt: int) -> ndarray:
        N_frames = int(t / dt + 1)
        M_rest = {}
        for key, value in results.items():
            M_k = value["gyration_tensor_eigenvalues"]
            N_k = M_k.shape[0]
            if N_k != N_frames:
                continue
            M_rest[key] = M_k.mean(axis = 0)
        if not M_rest:
            return array([])
        average_M_rest = sum(M_rest.values()) / len(M_rest)
        average_M_rest = array(average_M_rest)
        return average_M_rest
    
    def __calculate_elasticity_matrix_SIGMA(self, results: dict, average_M_rest: ndarray, t: int, dt: int) -> ndarray:
        N_frames = int(t / dt + 1)
        M_covariances = zeros((3, 3))
        total_frames = 0
        for key, value in results.items():
            M_k = value["gyration_tensor_eigenvalues"]
            N_k = M_k.shape[0]
            if N_k != N_frames:
                continue
            delta_M = M_k - average_M_rest
            M_covariances += delta_M.T @ delta_M
            total_frames += N_k
        if total_frames == 0:
            return array([])
        elasticity_matrix_SIGMA = self.__beta * (M_covariances / total_frames)
        return elasticity_matrix_SIGMA
    
    def __calculate_time_correlation_function_M(self, results: dict, average_M_rest: ndarray, t: int, dt: int, df_custom: DataFrame, key: str) -> ndarray:
        N_frames = int(t / dt + 1)
        deltas = []
        for value in results.values():
            M = value["gyration_tensor_eigenvalues"]
            if M.shape[0] != N_frames:
                continue
            deltas.append(M - average_M_rest)
        if len(deltas) == 0:
            time_correlation_function_M = zeros((N_frames, 3))
        else:
            deltas = stack(deltas)
            
        fft_length = 2 * N_frames
        spectrum = rfft(deltas, n = fft_length, axis = 1)
        correlation = irfft(spectrum * spectrum.conjugate(), n = fft_length, axis = 1)
        correlation = correlation[:, :N_frames, :]
        normalization = (N_frames - arange(N_frames)).reshape(1, N_frames, 1)
        time_correlation_function_M = (correlation / normalization).mean(axis = 0)
        
        df_custom["⟨M1(t) M1⟩"] = time_correlation_function_M[:, 0]
        df_custom["⟨M2(t) M2⟩"] = time_correlation_function_M[:, 1]
        df_custom["⟨M3(t) M3⟩"] = time_correlation_function_M[:, 2]
        data = column_stack((range(int(t / dt + 1)), time_correlation_function_M[:, 0], time_correlation_function_M[:, 1], time_correlation_function_M[:, 2]))
        savetxt(f"{self._directory_txt}/{key}_time_correlation_function_M.dat", data, header = "Step M_1_M_1 M_2_M_2 M_3_M_3")
        return time_correlation_function_M
    
    def __calculate_trajectory_time_correlation_function_M(self, results: dict, average_M_rest: ndarray, t: int, dt: int, df_custom: DataFrame, key: str) -> None:
        N_frames = int(t / dt + 1)
        fft_length = 2 * N_frames
        normalization = (N_frames - arange(N_frames)).reshape(N_frames, 1)
        data = [arange(N_frames)]
        header = ["Step"]
        for k, value in results.items():
            M_k = value["gyration_tensor_eigenvalues"]
            if M_k.shape[0] != N_frames:
                continue
            delta_M = M_k - average_M_rest
            spectrum = rfft(delta_M, n = fft_length, axis = 0)
            trajectory_time_correlation_function_M = irfft(spectrum * spectrum.conjugate(), n = fft_length, axis = 0)
            trajectory_time_correlation_function_M = (trajectory_time_correlation_function_M[:N_frames] / normalization)
    
            df_custom[f"⟨M1(t) M1⟩_{k}"] = trajectory_time_correlation_function_M[:, 0]
            df_custom[f"⟨M2(t) M2⟩_{k}"] = trajectory_time_correlation_function_M[:, 1]
            df_custom[f"⟨M3(t) M3⟩_{k}"] = trajectory_time_correlation_function_M[:, 2]
            data.extend([trajectory_time_correlation_function_M[:, 0], trajectory_time_correlation_function_M[:, 1], trajectory_time_correlation_function_M[:, 2],])
            header.extend([f"M_1_M_1_{k}", f"M_2_M_2_{k}", f"M_3_M_3_{k}",])
        data = column_stack(data)
        savetxt(f"{self._directory_txt}/{key}_trajectory_time_correlation_function_M.dat", data, header = " ".join(header))
            
    def __calculate_time_correlation_matrix_eigenvectors_E(self, results: dict, t: int, dt: int, df_custom: DataFrame, key: str) -> ndarray:
        N_frames = int(t / dt + 1)
        eigenvectors_components = []
        for value in results.values():
            eigenvectors = value["gyration_tensor_eigenvectors"]
            if eigenvectors.shape[0] != N_frames:
                continue
            eigenvectors_components.append(eigenvectors.transpose(0, 2, 1).reshape(N_frames, 9))
        eigenvectors_components = array(eigenvectors_components)
        
        n_sim = eigenvectors_components.shape[0]
        eig = eigenvectors_components.reshape(n_sim, N_frames, 3, 3)
        fft_length = 2 * N_frames
        normalization = (N_frames - arange(N_frames)).reshape(1, N_frames)
        correlation_matrix = zeros((N_frames, 3, 3))
        for i in range(3):
            for j in range(3):
                vi = eig[:, :, i, :]
                vj = eig[:, :, j, :]
                fft_i = rfft(vi, n = fft_length, axis = 1)
                fft_j = rfft(vj, n = fft_length, axis = 1)
                corr = irfft(fft_i * fft_j.conjugate(), n = fft_length, axis = 1)
                corr = corr[:, :N_frames, :]
                corr = corr.sum(axis = 2)
                corr /= normalization
                correlation_matrix[:, i, j] = corr.mean(axis = 0)
        correlation_matrix *= 1/1
        
        df_custom["⟨e1(t) e1⟩"] = correlation_matrix[:, 0, 0]
        df_custom["⟨e1(t) e2⟩"] = correlation_matrix[:, 0, 1]
        df_custom["⟨e1(t) e3⟩"] = correlation_matrix[:, 0, 2]
        df_custom["⟨e2(t) e1⟩"] = correlation_matrix[:, 1, 0]
        df_custom["⟨e2(t) e2⟩"] = correlation_matrix[:, 1, 1]
        df_custom["⟨e2(t) e3⟩"] = correlation_matrix[:, 1, 2]
        df_custom["⟨e3(t) e1⟩"] = correlation_matrix[:, 2, 0]
        df_custom["⟨e3(t) e2⟩"] = correlation_matrix[:, 2, 1]
        df_custom["⟨e3(t) e3⟩"] = correlation_matrix[:, 2, 2]
        data = column_stack((range(int(t / dt + 1)), correlation_matrix[:, 0, 0], correlation_matrix[:, 1, 1], correlation_matrix[:, 2, 2]))
        savetxt(f"{self._directory_txt}/{key}_time_correlation_matrix_eigenvectors_E.dat", data, header = "Step e_1_e_1 e_2_e_2 e_3_e_3")
        return correlation_matrix
    
    def __calculate_trajectory_time_correlation_matrix_eigenvectors_E(self, results: dict, t: int, dt: int, df_custom: DataFrame, key: str) -> ndarray:
        N_frames = int(t / dt + 1)
        fft_length = 2 * N_frames
        normalization = (N_frames - arange(N_frames))[:, None]
        first = True
        components = ["x", "y", "z"]
        data = [arange(N_frames)]
        header = ["Step"]
        for k, value in results.items():
            eigenvectors = value["gyration_tensor_eigenvectors"]
            if eigenvectors.shape[0] != N_frames:
                continue
            eig = eigenvectors.transpose(0, 2, 1)
            correlation_matrix = zeros((N_frames, 3, 3))
            for i in range(3):
                for j in range(3):
                    vi = eig[:, i, :]
                    vj = eig[:, j, :]
                    fft_i = rfft(vi, n = fft_length, axis = 0)
                    fft_j = rfft(vj, n = fft_length, axis = 0)
                    corr = irfft(fft_i * fft_j.conjugate(), n = fft_length, axis = 0)
                    corr = corr[:N_frames, :]
                    corr /= normalization
                    if first and i == j:
                        for c, component in enumerate(components):
                            df_custom[f"⟨e{i+1}(t),{component} e{j+1},{component}⟩_{k}"] = corr[:, c]
                            df_custom[f"⟨e{i+1}(t),{component} e{j+1},{component}⟩_{k}"] *= 1/3
                            data.append(df_custom[f"⟨e{i+1}(t),{component} e{j+1},{component}⟩_{k}"])
                            header.append(f"e_{i+1}_{component}_e_{i+1}_{component}_{k}")
                    corr = corr.sum(axis = 1)
                    correlation_matrix[:, i, j] = corr
                    correlation_matrix[:, i, j] *= 1/1
                    
            first = False
            df_custom[f"⟨e1(t) e1⟩_{k}"] = correlation_matrix[:, 0, 0]
            df_custom[f"⟨e2(t) e2⟩_{k}"] = correlation_matrix[:, 1, 1]
            df_custom[f"⟨e3(t) e3⟩_{k}"] = correlation_matrix[:, 2, 2]
            data.extend([correlation_matrix[:, 0, 0], correlation_matrix[:, 1, 1], correlation_matrix[:, 2, 2],])
            header.extend([f"e_1_e_1_{k}", f"e_2_e_2_{k}", f"e_3_e_3_{k}",])
        data = column_stack(data)
        savetxt(f"{self._directory_txt}/{key}_trajectory_time_correlation_matrix_eigenvectors_E.dat", data, header = " ".join(header))
            
    def __calculate_fitting_time_correlation_function_M_Ornstein_Ulhenbeck(self, average_M_rest: ndarray, elasticity_matrix_SIGMA: ndarray, time_correlation_function_M: ndarray, dt: int, t_int: int, key: str) -> tuple[ndarray, ndarray]:
        # inv_elasticity_matrix_SIGMA = linalg.inv(elasticity_matrix_SIGMA)
        # theoretical_omega_squared = average_M_rest * diag(inv_elasticity_matrix_SIGMA)
        # theoretical_omega = sqrt(abs(theoretical_omega_squared))
        # print(f"\nLos valores teóricos de las frecuencias de oscilación para la función de correlación temporal de los momentos centrales M son:\n\n{theoretical_omega}")
        
        x_col = 'Step'
        fitted_omega = zeros((1,3), float)
        fitted_gamma = zeros((3,3), float)
        
        def model_1(t, gamma):
            return A * exp(-gamma * t)

        def model_2(t, gamma, omega):
            return A * exp(-gamma * t) * cos(omega * t)
        
        for i in range(time_correlation_function_M.shape[1]):
            y_col = f'⟨M{i+1}(t) M{i+1}⟩'
            figure()
            y = time_correlation_function_M[0:t_int, i]
            x = arange(len(y))
            plot(
                x,
                y,
                linestyle = '--',
                marker = 'o',
                markerfacecolor = '#7ABBE6',
                markeredgecolor = '#7ABBE6',
                color = '#7ABBE6',
                linewidth = 0.5,
                markersize = 1,
                label = y_col,
                alpha = 0.5
            )
            title(f'{y_col} vs {x_col}', fontsize = 10)
            xlabel(x_col, fontsize = 10)
            ylabel(y_col, fontsize = 10)
            grid(True)
            
            # Comentario:
            # - Obtención de los "picos" máximos de cada componente de la
            #   función de correlación temporal M.
            idx_max, _ = find_peaks(abs(y), distance = 70, prominence = 0.0001)
            # if abs(y[0]) > abs(y[1]):
            #     idx_max = insert(idx_max, 0, 0)
            x_max = x[idx_max]
            y_max = abs(y[idx_max])
            # Fin Comentario.
            
            # Comentario:
            # - Ajuste de cada componente de la función de correlación 
            #   temporal M según el proceso de Ornstein - Ulhenbeck.
            A = abs(y_max[0])
            params_1, _ = curve_fit(model_1, x_max, abs(y_max), p0 = 0.0005)
            gamma_fit_1 = params_1[0]
            fitted_gamma[i, i] = gamma_fit_1
            x_fit = x
            y_fit_1 = model_1(x_fit, gamma_fit_1)
            y_fit_col = f'Gamma = {gamma_fit_1:.6f}'
            scatter(
                x_max,
                y_max,
                marker = 'x',
                color = '#ED1C1C',
                s = 20,
                label = 'max',
            )
            plot(
                x_fit,
                y_fit_1,
                linestyle = '--',
                marker = 'x',
                markerfacecolor = '#242323',
                markeredgecolor = '#242323',
                color = '#242323',
                linewidth = 0.5,
                markersize = 1,
                label = y_fit_col
            )
            legend()
            # Fin Comentario.
            data = column_stack((x, y, x_fit, y_fit_1))
            savetxt(f"{self._directory_txt}/{key}_fitting_time_correlation_function_M_{i+1}.dat", data, header = "x y x_fit y_fit")
            data = column_stack((x_max, y_max))
            savetxt(f"{self._directory_txt}/{key}_fitting_time_correlation_function_M_{i+1}_max.dat", data, header = "x_max y_max")
        show()
        
        # for i in range(time_correlation_function_M.shape[1]):
        #     y_col = f'⟨M{i+1}(t) M{i+1}⟩'
        #     figure()
        #     y = time_correlation_function_M[0:t_int, i]
        #     x = arange(len(y))
        #     plot(
        #         x,
        #         y,
        #         linestyle = '--',
        #         marker = 'o',
        #         markerfacecolor = '#7ABBE6',
        #         markeredgecolor = '#7ABBE6',
        #         color = '#7ABBE6',
        #         linewidth = 0.5,
        #         markersize = 1,
        #         label = y_col,
        #         alpha = 0.5
        #     )
        #     title(f'{y_col} vs {x_col}', fontsize = 10)
        #     xlabel(x_col, fontsize = 10)
        #     ylabel(y_col, fontsize = 10)
        #     grid(True)
            
        #     # Comentario:
        #     # - Obtención de la frecuencia de cada componente de la función
        #     #   de correlación temporal M.
        #     omega = self.__calculate_frecuency(y, dt)
        #     # Fin Comentario.
            
        #     # Comentario:
        #     # - Ajuste de cada componente de la función de correlación 
        #     #   temporal M según el proceso de Ornstein - Ulhenbeck.
        #     A = y[0]
        #     params_2, _ = curve_fit(model_2, x, y, p0 = [gamma_fit_1, omega])
        #     gamma_fit_2, omega_fit_2 = params_2
        #     fitted_omega[0, i] = omega_fit_2
        #     x_fit = x
        #     y_fit_2 = model_2(x_fit, gamma_fit_1, omega_fit_2)
        #     y_fit_col = (
        #         f'Gamma = {gamma_fit_2:.6f}\n'
        #         f'Omega = {omega_fit_2:.6f}'
        #     )
        #     plot(
        #         x_fit,
        #         y_fit_2,
        #         linestyle = '--',
        #         marker = 'x',
        #         markerfacecolor = '#242323',
        #         markeredgecolor = '#242323',
        #         color = '#242323',
        #         linewidth = 0.5,
        #         markersize = 1,
        #         label = y_fit_col
        #     )
        #     legend()
        #     # Fin Comentario.
        # show()
        
        print(f"\nEl valor de la matriz empírica Gamma para la función de correlación temporal de los momentos centrales M es:\n\n{fitted_gamma}")
        print(f"\nLos valores empíricos de las frecuencias de oscilación para la función de correlación temporal de los momentos centrales M son:\n\n{fitted_omega}")
        
        return [fitted_gamma, fitted_omega]
    
    def __calculate_frecuency(self, y: ndarray, dt: int) -> float:
        N = len(y)
        y = y - mean(y)
        Y = rfft(y)
        freqs = rfftfreq(N, d = dt)
        indice = argmax(abs(Y[1:])) + 1
        frecuency = freqs[indice]
        return frecuency
    
    def __calculate_fitting_time_correlation_matrix_eigenvectors_E(self, time_correlation_matrix_eigenvectors_E: ndarray, dt: int, t_int: int, key: str) -> ndarray:
        x_col = 'Step'
        fitted_A = zeros((1,3), float)
        fitted_D0 = zeros((3,3), float)
        
        def model_1(t, A):
            return (1/1) * exp(-A * t)
        
        for i in range(time_correlation_matrix_eigenvectors_E.shape[1]):
            y_col = f'⟨e{i+1}(t) e{i+1}⟩'
            figure()
            y = time_correlation_matrix_eigenvectors_E[0:t_int, i, i]
            x = arange(len(y))
            plot(
                x,
                y,
                linestyle = '--',
                marker = 'o',
                markerfacecolor = '#7ABBE6',
                markeredgecolor = '#7ABBE6',
                color = '#7ABBE6',
                linewidth = 0.5,
                markersize = 1,
                label = y_col,
                alpha = 0.5
            )
            title(f'{y_col} vs {x_col}', fontsize = 10)
            xlabel(x_col, fontsize = 10)
            ylabel(y_col, fontsize = 10)
            grid(True)
            
            # Comentario:
            # - Ajuste de cada componente de la función de correlación 
            #   temporal E_i según una función exponencial decreciente.
            params_1, _ = curve_fit(model_1, x, y, p0 = 0.0005)
            A_fit_1 = params_1[0]
            fitted_A[0, i] = A_fit_1
            x_fit = x
            y_fit_1 = model_1(x_fit, A_fit_1)
            y_fit_col = f'A = {A_fit_1:.6f}'
            plot(
                x_fit,
                y_fit_1,
                linestyle = '--',
                marker = 'x',
                markerfacecolor = '#242323',
                markeredgecolor = '#242323',
                color = '#242323',
                linewidth = 0.5,
                markersize = 1,
                label = y_fit_col
            )
            legend()
            # Fin Comentario.
            data = column_stack((x, y, x_fit, y_fit_1))
            savetxt(f"{self._directory_txt}/{key}_fitting_time_correlation_matrix_eigenvectors_E_{i+1}.dat", data, header = "x y x_fit y_fit")
        show()
        
        fitted_D0[0,0] = 0.5 * self.__beta * (fitted_A[0,1] + fitted_A[0,2] - fitted_A[0,0])
        fitted_D0[1,1] = 0.5 * self.__beta * (fitted_A[0,0] + fitted_A[0,2] - fitted_A[0,1])
        fitted_D0[2,2] = 0.5 * self.__beta * (fitted_A[0,0] + fitted_A[0,1] - fitted_A[0,2])
        
        print(f"\nEl valor de la matriz empírica D0 para la matriz de función de correlación temporal de los autovectores E es:\n\n{fitted_D0}")
        
        return fitted_D0
    
    def __calculate_average_kinetic_energy(self, results: dict, t: int, dt: int, df_custom: DataFrame, key: str) -> ndarray:
        if ":" in key:
            key_1, key_2 = key.split(":", 1)
        else:
            key_1 = key
            key_2 = key
        N_frames = int(t / dt + 1)
        kinetic_energies = []
        for value in results.values():
            if key_2 in value:
                kinetic_energy = value[key_2].squeeze()
                if len(kinetic_energy) == N_frames:
                    kinetic_energies.append(kinetic_energy)
        average_kinetic_energy = mean(kinetic_energies, axis = 0)
        df_custom[f"⟨{key_2}⟩"] = average_kinetic_energy
        data = column_stack((range(int(t / dt + 1)), average_kinetic_energy))
        savetxt(f"{self._directory_txt}/{key_1}_average_{key_2}.dat", data, header = f"Step average_{key_2}")
        return average_kinetic_energy
        
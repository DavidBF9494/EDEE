from matplotlib.pyplot import figure, pause, close
from IPython.display import display, clear_output
from numpy import array, zeros, ndarray, outer, linalg, cross, average, trace, eye, dot, full, nan, sum, mean, std
from pandas import DataFrame
from pathlib import Path
from scipy.cluster.vq import kmeans2
from sklearn.metrics import silhouette_score
from lammps import lammps
from MDAnalysis import Universe
from pickle import dump

from .EDEE_Auxiliary import EDEE_Auxiliary

class EDEE_MolecularDynamicSimulation(EDEE_Auxiliary):
   
    def __init__(self, n_sim: int, t_nvt: int, t_restart_nvt: int, t_nve: int, t_eq: int, t_neq: int, dt_nvt: int, dt_nve: int, dt_eq: int, dt_neq: int, temp: float, omega: float) -> None:
        super().__init__()
        
        self.__n_sim = n_sim
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
        self.__omega = omega
        
        self._delete_files()
        self.__init_lammps__()
        return None
      
    def __init_lammps__(self) -> None:
        print("\n<-- EDEE_MolecularDynamicSimulation - Simulaciones Dinámica Molecular -->")
        
        print("\n<- Fase NVT ->\n")
        print("Ejecutando Comandos LAMMPS:\n")
        self.__results_nvt = {}
        print("0|0 : Procesando Microestado...")
        with open(self._directory_input / self._directory_nvt / self._filename_input_nvt, "r") as f:
            script = f.read()
            script = script.replace("${directory_molecules}", str(self._directory_molecules))
            script = script.replace("${filename_molecule}", str(self._filename_molecule))
            script = script.replace("${directory_data}", str(self._directory_data))
            script = script.replace("${directory_input}", str(self._directory_input))
            script = script.replace("${directory_nvt}", str(self._directory_nvt))
            script = script.replace("${filename_traj_nvt}", str(self._filename_traj_nvt))
            script = script.replace("${filename_restart_nvt}", str(self._filename_restart_nvt))
            script = script.replace("${n_sim}", str(self.__n_sim))
            script = script.replace("${t_nvt}", str(self.__t_nvt))
            script = script.replace("${t_restart_nvt}", str(self.__t_restart_nvt))
            script = script.replace("${dt_nvt}", str(self.__dt_nvt))
            script = script.replace("${temp}", str(self.__temp))
        log_filename = f"{self._filename_log_nvt}_0.lammps"
        LMP = lammps(cmdargs = ["-screen", "none", "-log", str(self._directory_data / self._directory_nvt / log_filename)])
        LMP.commands_string(script)
        LMP.close()
        for index in range(self.__n_sim + 1):
            print(f"{index}|{self.__n_sim } : Calculando Universo...")
            self.__results_nvt[index] = self.__calculate_universe(self._directory_nvt, self._filename_traj_nvt, self.__dt_nvt, index)
        self.__results_nvt[0]["df_logs"] = self._load_log_file(self._directory_data / self._directory_nvt, log_filename)
        print("\nCalculando Índices Microestados Aproximadamente Isoenergéticos:\n")
        iso_idxs = self.__isoenergetic_microstates_indexes(self.__n_sim, self.__results_nvt)
        iso_idxs_len = len(iso_idxs)
        print("\nCalculando Velocidades Microestados Isoenergéticos:\n")
        alphas_isoenergetic_velocities = self.__calculate_alphas_isoenergetic_velocities(self.__n_sim, iso_idxs, self.__results_nvt[0]["df_logs"])
        for index in iso_idxs:
            self.__results_nvt[index]['isoenergetic_velocities'] = alphas_isoenergetic_velocities[index] * self.__results_nvt[index]["velocities"][-1]
        
        print("\n<- Fase NVE ->\n")
        print("Ejecutando Comandos LAMMPS:\n")
        self.__results_nve = {}
        count = 0
        for index in iso_idxs:
            count += 1
            print(f"{count}|{iso_idxs_len} : Procesando Microestado...")
            with open(self._directory_input / self._directory_nve / self._filename_input_nve, "r") as f:
                script = f.read()
                script = script.replace("${directory_data}", str(self._directory_data))
                script = script.replace("${directory_input}", str(self._directory_input))
                script = script.replace("${directory_nvt}", str(self._directory_nvt))
                script = script.replace("${directory_nve}", str(self._directory_nve))
                script = script.replace("${filename_traj_nve}", str(self._filename_traj_nve))
                script = script.replace("${filename_restart_nve}", str(self._filename_restart_nve))
                script = script.replace("${index}", str(index))
                script = script.replace("${t_nve}", str(self.__t_nve))
                script = script.replace("${dt_nve}", str(self.__dt_nve))
            isoenergetic_velocities = self.__results_nvt[index]['isoenergetic_velocities']
            lines = []
            for n, (vx, vy, vz) in enumerate(isoenergetic_velocities):
                line = f"set atom {n + 1} vx {vx:.10f} vy {vy:.10f} vz {vz:.10f}"
                lines.append(line)
            custom_commands = "\n".join(lines)
            script = script.replace("${custom_commands}", custom_commands)
            log_filename = f"{self._filename_log_nve}_{index}.lammps"
            LMP = lammps(cmdargs = ["-screen", "none", "-log", str(self._directory_data / self._directory_nve / log_filename)])
            LMP.commands_string(script)
            LMP.close()
            print(f"{count}|{iso_idxs_len} : Calculando Universo...")
            self.__results_nve[index] = self.__calculate_universe(self._directory_nve, self._filename_traj_nve, self.__dt_nve, index)
            self.__results_nve[index]["df_logs"] = self._load_log_file(self._directory_data / self._directory_nve, log_filename)
            
        print("\n<- Fase Equilibrio ->\n")
        print("Ejecutando Comandos LAMMPS:\n")
        self.__results_eq = {}
        count = 0
        for index in iso_idxs:
            count += 1
            print(f"{count}|{iso_idxs_len} : Procesando Microestado...")
            with open(self._directory_input / self._directory_eq / self._filename_input_eq, "r") as f:
                script = f.read()
                script = script.replace("${directory_data}", str(self._directory_data))
                script = script.replace("${directory_input}", str(self._directory_input))
                script = script.replace("${directory_nve}", str(self._directory_nve))
                script = script.replace("${directory_eq}", str(self._directory_eq))
                script = script.replace("${filename_traj_eq}", str(self._filename_traj_eq))
                script = script.replace("${filename_restart_nve}", str(self._filename_restart_nve))
                script = script.replace("${index}", str(index))
                script = script.replace("${t_eq}", str(self.__t_eq))
                script = script.replace("${dt_eq}", str(self.__dt_eq))
            log_filename = f"{self._filename_log_eq}_{index}.lammps"
            LMP = lammps(cmdargs = ["-screen", "none", "-log", str(self._directory_data / self._directory_eq / log_filename)])
            LMP.commands_string(script)
            LMP.close()
            print(f"{count}|{iso_idxs_len} : Calculando Universo...")
            self.__results_eq[index] = self.__calculate_universe(self._directory_eq, self._filename_traj_eq, self.__dt_eq, index)
            self.__results_eq[index]["df_logs"] = self._load_log_file(self._directory_data / self._directory_eq, log_filename)
            
        print("\n<- Fase No Equilibrio ->\n")
        print("Ejecutando Comandos LAMMPS:\n")
        self.__results_neq = {}
        count = 0
        for index in iso_idxs:
            count += 1
            print(f"{count}|{iso_idxs_len} : Procesando Microestado...")
            with open(self._directory_input / self._directory_neq / self._filename_input_neq, "r") as f:
                script = f.read()
                script = script.replace("${directory_data}", str(self._directory_data))
                script = script.replace("${directory_input}", str(self._directory_input))
                script = script.replace("${directory_nve}", str(self._directory_nve))
                script = script.replace("${directory_neq}", str(self._directory_neq))
                script = script.replace("${filename_traj_neq}", str(self._filename_traj_neq))
                script = script.replace("${filename_restart_nve}", str(self._filename_restart_nve))
                script = script.replace("${index}", str(index))
                script = script.replace("${t_neq}", str(self.__t_neq))
                script = script.replace("${dt_neq}", str(self.__dt_neq))
            intermediate_axis_rotation_velocities = self.__results_nve[index]['intermediate_axis_rotation_velocities']
            lines = []
            for n, (vx, vy, vz) in enumerate(intermediate_axis_rotation_velocities):
                line = f"set atom {n + 1} vx {vx:.10f} vy {vy:.10f} vz {vz:.10f}"
                lines.append(line)
            custom_commands = "\n".join(lines)
            script = script.replace("${custom_commands}", custom_commands)
            log_filename = f"{self._filename_log_neq}_{index}.lammps"
            LMP = lammps(cmdargs = ["-screen", "none", "-log", str(self._directory_data / self._directory_neq / log_filename)])
            LMP.commands_string(script)
            LMP.close()
            print(f"{count}|{iso_idxs_len} : Calculando Universo...")
            self.__results_neq[index] = self.__calculate_universe(self._directory_neq, self._filename_traj_neq, self.__dt_neq, index)
            self.__results_neq[index]["df_logs"] = self._load_log_file(self._directory_data / self._directory_neq, log_filename)
            
        results_save = {
            "results_nvt": self.__results_nvt,
            "results_nve": self.__results_nve,
            "results_eq": self.__results_eq,
            "results_neq": self.__results_neq,
        }
        for key, data in results_save.items():
            file_path = self._directory_variables / f"{key}.pkl"
            with open(file_path, "wb") as f:
                dump(data, f)
        return None
                
    def __calculate_universe(self, directory_ensemble: Path, filename_traj_ensamble: Path, dt: int, index: int) -> dict:
        traj_filename = f"{filename_traj_ensamble}_{index}.lammpstrj"
        universe = Universe(
            str(self._directory_molecules / self._filename_molecule), 
            str(self._directory_data / directory_ensemble / traj_filename), 
            topology_format = "DATA", 
            format = "LAMMPSDUMP",
            lammps_coordinate_convention = "unscaled",
            unwrap_images = False,
            dt = dt * 1e-3
        )
        
        masses = universe.atoms.masses
        n_particles = universe.atoms.n_atoms
        n_frames = universe.trajectory.n_frames
        positions = self.__calculate_positions(universe)
        velocities = self.__calculate_velocities(universe)
        
        df_custom = DataFrame({'Step': range(n_frames)})
        
        (
            gyration_tensor,
            gyration_tensor_eigenvalues,
            gyration_tensor_eigenvectors,
        ) = self.__calculate_gyration_tensor(
            masses, 
            n_particles, 
            n_frames,
            positions
        )
            
        (
            inertia_tensor,
            inertia_tensor_eigenvalues,
            inertia_tensor_eigenvectors,
        ) = self.__calculate_inertia_tensor(
            n_frames,
            gyration_tensor
        )
            
        (
            rotation_matrix    
        ) = self.__calculate_rotation_matrix(
            n_frames,
            gyration_tensor_eigenvectors,
        )
            
        (
            angular_momentum    
        ) = self.__calculate_angular_momentum(
            masses, 
            n_particles, 
            n_frames,
            positions,
            velocities
        )
            
        (
            angular_velocity
        ) = self.__calculate_angular_velocity(
            n_frames,
            inertia_tensor,
            angular_momentum
        )
            
        (
            angular_velocity_principal_axis
        ) = self.__calculate_angular_velocity_principal_axis(
           n_frames,
           rotation_matrix,
           dt
        )
            
        (
            dilation_momentum   
        ) = self.__calculate_dilation_momentum(
            masses,
            n_particles, 
            n_frames,
            positions,
            velocities,
            rotation_matrix,
            angular_velocity
        )
            
        (
            rotational_kinetic_energy
        ) = self.__calculate_rotational_kinetic_energy(
            n_frames,
            inertia_tensor,
            angular_velocity,
            df_custom
        )
            
        (
            dilational_kinetic_energy
        ) = self.__calculate_dilational_kinetic_energy(
            n_frames,
            gyration_tensor_eigenvalues,
            dilation_momentum,
            df_custom
        )
          
        (
            translational_kinetic_energy
        ) = self.__calculate_translational_kinetic_energy(
            n_frames,
            masses,
            velocities,
            df_custom
        )
        
        results = {
            'masses': masses,
            'n_particles': n_particles,
            'n_frames': n_frames,
            'positions': positions,
            'velocities': velocities,
            'gyration_tensor': gyration_tensor,
            'gyration_tensor_eigenvalues': gyration_tensor_eigenvalues,
            'gyration_tensor_eigenvectors': gyration_tensor_eigenvectors,
            'inertia_tensor': inertia_tensor,
            'inertia_tensor_eigenvalues': inertia_tensor_eigenvalues,
            'inertia_tensor_eigenvectors': inertia_tensor_eigenvectors,
            'rotation_matrix': rotation_matrix,
            'angular_momentum': angular_momentum,
            'angular_velocity': angular_velocity,
            'angular_velocity_principal_axis': angular_velocity_principal_axis,
            'dilation_momentum': dilation_momentum,
            'rotational_kinetic_energy': rotational_kinetic_energy,
            'dilational_kinetic_energy': dilational_kinetic_energy,
            'translational_kinetic_energy': translational_kinetic_energy,
            'df_custom': df_custom,
        }
        
        if (directory_ensemble == self._directory_nve):
            (
                intermediate_axis_rotation_velocities
            ) = self.__calculate_intermediate_axis_rotation_velocities(
                masses,
                n_particles,
                n_frames,
                positions,
                gyration_tensor_eigenvectors,
                self.__omega
            )
            results['intermediate_axis_rotation_velocities'] = intermediate_axis_rotation_velocities
        return results
    
    def __calculate_positions(self, universe: Universe) -> ndarray:
        positions = []
        for ts in universe.trajectory:
            p = universe.atoms.positions.copy()
            positions.append(p)
        positions = array(positions)
        return positions
    
    def __calculate_velocities(self, universe: Universe) -> ndarray:
        velocities = []
        for ts in universe.trajectory:
            v = universe.atoms.velocities.copy()
            velocities.append(v)
        velocities = array(velocities)
        return velocities
        
    def __calculate_gyration_tensor(self, masses: ndarray, n_particles: int, n_frames: int, positions: ndarray) -> tuple[ndarray, ndarray, ndarray]:
        gyration_tensor = []
        gyration_tensor_eigenvalues = []
        gyration_tensor_eigenvectors = []
        for t in range(n_frames):
            pos_t = positions[t]
            com_t = average(pos_t, axis = 0, weights = masses)
            pos_centered_t = pos_t - com_t
            
            sub_gyration_tensor = zeros((3,3), float)
            for i in range(n_particles):
                sub_gyration_tensor += (1/4) * masses[i] * outer(
                    pos_centered_t[i],
                    pos_centered_t[i]
                )
            sub_gyration_tensor_eigenvalues, sub_gyration_tensor_eigenvectors = linalg.eigh(sub_gyration_tensor)
            
            # Corrección: 
            # - Inversión autovectores (e <-> -e).
            if t > 0:
                prev_gyration_tensor_eigenvectors = gyration_tensor_eigenvectors[-1]
                for i in range(3):
                    if dot(prev_gyration_tensor_eigenvectors[:, i], sub_gyration_tensor_eigenvectors[:, i]) < 0:
                        sub_gyration_tensor_eigenvectors[:, i] *= -1
            # Fin Correción.
            
            gyration_tensor.append(sub_gyration_tensor)
            gyration_tensor_eigenvalues.append(sub_gyration_tensor_eigenvalues)
            gyration_tensor_eigenvectors.append(sub_gyration_tensor_eigenvectors)
        gyration_tensor = array(gyration_tensor)
        gyration_tensor_eigenvalues = array(gyration_tensor_eigenvalues)
        gyration_tensor_eigenvectors = array(gyration_tensor_eigenvectors)
        return gyration_tensor, gyration_tensor_eigenvalues, gyration_tensor_eigenvectors
    
    def __calculate_inertia_tensor(self, n_frames: int, gyration_tensor: ndarray) -> tuple[ndarray, ndarray, ndarray]:
        inertia_tensor = []
        inertia_tensor_eigenvalues = []
        inertia_tensor_eigenvectors = []
        for t in range(n_frames):
            sub_inertia_tensor = 4 * (trace(gyration_tensor[t]) * eye(3) - gyration_tensor[t])
            sub_inertia_tensor_eigenvalues, sub_inertia_tensor_eigenvectors = linalg.eigh(sub_inertia_tensor)
            
            # Corrección: 
            # - Inversión autovectores (e <-> -e).
            if t > 0:
                prev_inertia_tensor_eigenvectors = inertia_tensor_eigenvectors[-1]
                for i in range(3):
                    if dot(prev_inertia_tensor_eigenvectors[:, i], sub_inertia_tensor_eigenvectors[:, i]) < 0:
                        sub_inertia_tensor_eigenvectors[:, i] *= -1
            # Fin Correción.
            
            inertia_tensor.append(sub_inertia_tensor)
            inertia_tensor_eigenvalues.append(sub_inertia_tensor_eigenvalues)
            inertia_tensor_eigenvectors.append(sub_inertia_tensor_eigenvectors)
        inertia_tensor = array(inertia_tensor)
        inertia_tensor_eigenvalues = array(inertia_tensor_eigenvalues)
        inertia_tensor_eigenvectors = array(inertia_tensor_eigenvectors)
        return inertia_tensor, inertia_tensor_eigenvalues, inertia_tensor_eigenvectors
        
    def __calculate_rotation_matrix(self, n_frames: int, gyration_tensor_eigenvectors: ndarray) -> ndarray:
        rotation_matrix = []
        for t in range(n_frames):
            E_t = gyration_tensor_eigenvectors[t]
            sub_rotation_matrix = E_t.T
            rotation_matrix.append(sub_rotation_matrix)
        rotation_matrix = array(rotation_matrix)
        return rotation_matrix
    
    def __calculate_angular_momentum(self, masses: ndarray, n_particles: int, n_frames: int, positions: ndarray, velocities: ndarray) -> ndarray:
        angular_momentum = []
        for t in range(n_frames):
            pos_t = positions[t]
            vel_t = velocities[t]
            com_t = average(pos_t, axis = 0, weights = masses)
            v_com_t = average(vel_t, axis = 0, weights = masses)
            pos_centered_t = pos_t - com_t
            vel_centered_t = vel_t - v_com_t
            
            sub_angular_momentum = zeros((1,3), float)
            for i in range(n_particles):
                sub_angular_momentum += masses[i] * cross(pos_centered_t[i], vel_centered_t[i])
            angular_momentum.append(sub_angular_momentum)
        angular_momentum = array(angular_momentum)
        return angular_momentum
    
    def __calculate_angular_velocity(self, n_frames: int, inertia_tensor: ndarray, angular_momentum: ndarray) -> ndarray:
        angular_velocity = []
        for t in range(n_frames):
            I_t = inertia_tensor[t]
            L_t = angular_momentum[t].flatten()
            sub_angular_velocity = zeros((1, 3), float)
            sub_angular_velocity[0] = linalg.solve(I_t, L_t)
            angular_velocity.append(sub_angular_velocity)
        angular_velocity = array(angular_velocity)
        return angular_velocity
    
    def __calculate_angular_velocity_principal_axis(self, n_frames: int, rotation_matrix: ndarray, dt: int) -> ndarray:
        angular_velocity_principal_axis = []
        for t in range(n_frames):
            sub_angular_velocity_principal_axis = zeros((1, 3), float)
            if t < n_frames - 1:
                R_t = rotation_matrix[t]
                R_tt = rotation_matrix[t + 1]
                Q = dot(R_t.T, R_tt)
                omega_x = (Q - Q.T) / (2 * dt)
                sub_angular_velocity_principal_axis[0, 0] = omega_x[2, 1]
                sub_angular_velocity_principal_axis[0, 1] = omega_x[0, 2]
                sub_angular_velocity_principal_axis[0, 2] = omega_x[1, 0]
            angular_velocity_principal_axis.append(sub_angular_velocity_principal_axis)
        angular_velocity_principal_axis = array(angular_velocity_principal_axis)
        return angular_velocity_principal_axis
    
    def __calculate_dilation_momentum(self, masses: ndarray, n_particles: int, n_frames: int, positions: ndarray, velocities: ndarray, rotation_matrix: ndarray, angular_velocity: ndarray) -> ndarray:
        pos_o = []
        vel_o = []
        for t in range(n_frames):
            pos_t = positions[t]
            vel_t = velocities[t]
            angular_velocity_t = angular_velocity[t]
            rotation_matrix_t = rotation_matrix[t]
            com_t = average(pos_t, axis = 0, weights = masses)
            v_com_t = average(vel_t, axis = 0, weights = masses)
            pos_centered_t = pos_t - com_t
            vel_centered_t = vel_t - v_com_t
            cross_product_t = cross(angular_velocity_t, pos_centered_t)
            pos_centered_t_rotated = pos_centered_t @ rotation_matrix_t.T
            vel_centered_t_rotated = vel_centered_t @ rotation_matrix_t.T
            cross_product_t_rotated = cross_product_t @ rotation_matrix_t.T
            pos_o.append(pos_centered_t_rotated)
            vel_o.append(vel_centered_t_rotated - cross_product_t_rotated)
        pos_o = array(pos_o)
        vel_o = array(vel_o)
        
        dilation_momentum = full((n_frames, 3), nan)
        for t in range(n_frames):
            dilation_momentum_t = zeros(3)
            for i in range(n_particles):
                dilation_momentum_t += masses[i] * pos_o[t, i] * vel_o[t, i]
            dilation_momentum[t] = 0.5 * dilation_momentum_t
        return dilation_momentum
    
    def __calculate_rotational_kinetic_energy(self, n_frames: int, inertia_tensor: ndarray, angular_velocity: ndarray, df: DataFrame) -> ndarray:
        rotational_kinetic_energy = []
        for t in range(n_frames):
            I_t = inertia_tensor[t]
            omega_t = angular_velocity[t].flatten()
            sub_rotational_kinetic_energy = zeros((1, 1), float)
            sub_rotational_kinetic_energy[0,0] = 0.5 * dot(omega_t, dot(I_t, omega_t))
            rotational_kinetic_energy.append(sub_rotational_kinetic_energy)
        rotational_kinetic_energy = array(rotational_kinetic_energy)
        df['RotKinEng'] = rotational_kinetic_energy.squeeze()
        return rotational_kinetic_energy
    
    def __calculate_dilational_kinetic_energy(self, n_frames: int, gyration_tensor_eigenvalues: ndarray, dilation_momentum: ndarray, df: DataFrame) -> ndarray:
        dilational_kinetic_energy = []
        for t in range(n_frames):
            sub_dilational_kinetic_energy = zeros((1, 1), float)
            sub_dilational_kinetic_energy[0,0] = 0.5 * sum(dilation_momentum[t, :] ** 2 / gyration_tensor_eigenvalues[t, :])
            dilational_kinetic_energy.append(sub_dilational_kinetic_energy)
        dilational_kinetic_energy = array(dilational_kinetic_energy)
        df['DilKinEng'] = dilational_kinetic_energy.squeeze()
        return dilational_kinetic_energy
    
    def __calculate_translational_kinetic_energy(self, n_frames: int, masses: ndarray, velocities: ndarray, df: DataFrame) -> ndarray:
        M = sum(masses)
        translational_kinetic_energy = []
        for t in range(n_frames):
            vel_t = velocities[t]
            v_com_t = average(vel_t, axis = 0, weights = masses)
            sub_translational_kinetic_energy = zeros((1, 1), float)
            sub_translational_kinetic_energy[0,0] = 0.5 * M * dot(v_com_t, v_com_t)
            translational_kinetic_energy.append(sub_translational_kinetic_energy)
        translational_kinetic_energy = array(translational_kinetic_energy)
        df['TransKinEng'] = translational_kinetic_energy.squeeze()
        return translational_kinetic_energy
    
    def __isoenergetic_microstates_indexes(self, n_sim: int, results: dict) -> list:
        df_logs_restarts = {}
        keys = []
        items = list(results[0]["df_logs"].items())[-n_sim:]
        for i, (_, value) in enumerate(items):
            idx = i + 1
            df_logs_restarts[idx] = value
            keys.append(idx)
        array_restarts = []
        for index in df_logs_restarts:
            array_restarts.append([
                df_logs_restarts[index]["TotEng"].iloc[-1],
                # results[index]["angular_velocity"][-1][0][0],
                # results[index]["angular_velocity"][-1][0][1],
                # results[index]["angular_velocity"][-1][0][2],
            ])
        
        if n_sim == 1:
            best_k = 1
            return [keys[0]]
        else:
            array_restarts = (array_restarts - mean(array_restarts, axis = 0)) / std(array_restarts, axis = 0)
            best_k = 2
            best_score = -1
            max_k = 10
            for k in range(2, min(max_k + 1, n_sim)):
                centroids, labels = kmeans2(array_restarts, k = k, minit = '++')
                score = silhouette_score(array_restarts, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        # Comentario:
        # - Fijar directamente un valor de K = 1 si se quiere obviar la
        #   implementación del algoritmo de K-medias.
        best_k = 1
        # Fin Comentario.
        print(f"k óptimo: {best_k}")
        centroids, labels = kmeans2(array_restarts, k = best_k, minit = '++')
        cluster_dispersion = {}
        for cluster_id in range(best_k):
            cluster_points = array_restarts[labels == cluster_id]
            if len(cluster_points) <= 1:
                continue
            cluster_dispersion[cluster_id] = std(cluster_points)
        best_cluster_label = min(cluster_dispersion, key = cluster_dispersion.get)
        best_cluster_indices = []
        for i, label in enumerate(labels):
            if label == best_cluster_label:
                best_cluster_indices.append(keys[i])
        best_cluster_energies = []
        for idx in best_cluster_indices:
            energy = df_logs_restarts[idx]["TotEng"].iloc[-1]
            best_cluster_energies.append(energy)
        print(f"Índice del grupo más compacto: {best_cluster_label}")
        print(f"Dispersión del grupo más compacto: {cluster_dispersion[best_cluster_label]}")
        print(f"Índices del grupo más compacto: {','.join(map(str, best_cluster_indices))}")
        print(f"Energías del grupo más compacto: {','.join(map(str, best_cluster_energies))}")
        return best_cluster_indices
    
    def __calculate_alphas_isoenergetic_velocities(self, n_sim: int, iso_idxs: list, df: DataFrame) -> dict:
        df_logs_restarts = {}
        items = list(df.items())[-n_sim:]
        for i, (_, value) in enumerate(items):
            idx = i + 1
            df_logs_restarts[idx] = value
        
        energy_dict = {}
        potential_dict = {}
        kinetic_dict = {}
        for iso_idx in iso_idxs:
            energy_dict[iso_idx] = df_logs_restarts[iso_idx]["TotEng"].iloc[-1]
            potential_dict[iso_idx] = df_logs_restarts[iso_idx]["PotEng"].iloc[-1]
            kinetic_dict[iso_idx] = df_logs_restarts[iso_idx]["KinEng"].iloc[-1]
        meanTotEng = mean(list(energy_dict.values()))
        print("Energía media de los microestados iso en el último paso de integración:")
        print(f"\nE = {meanTotEng}")
        
        alphas_dict = {}
        for iso_idx in iso_idxs:
            alpha = ((meanTotEng - potential_dict[iso_idx]) / kinetic_dict[iso_idx]) ** 0.5
            alphas_dict[iso_idx] = alpha
        return alphas_dict
    
    def __calculate_intermediate_axis_rotation_velocities(self, masses: ndarray, n_particles: int, n_frames: int, positions: ndarray, gyration_tensor_eigenvectors: ndarray, omega: float) -> ndarray:
        pos_last = positions[n_frames - 1]
        com_last = average(pos_last, axis = 0, weights = masses)
        pos_centered_last = pos_last - com_last
        
        velocities = []
        for i in range(n_particles):
            velocity = omega * cross(
                gyration_tensor_eigenvectors[n_frames - 1,:,1],
                pos_centered_last[i]
            )
            velocities.append(velocity)
        velocities = array(velocities)
        return velocities
            
    def plot_positions(self, masses: ndarray, positions: ndarray, gyration_tensor_eigenvectors: ndarray, angular_momentum: ndarray, angular_velocity: ndarray) -> None:
        n_t, n_p, n_d = positions.shape
        fig = figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.set_xlim(positions[:,:,0].min(), positions[:,:,0].max())
        ax.set_ylim(positions[:,:,1].min(), positions[:,:,1].max())
        ax.set_zlim(positions[:,:,2].min(), positions[:,:,2].max())
        ax.set_xlabel("x", fontsize = 10)
        ax.set_ylabel("y", fontsize = 10)
        ax.set_zlabel("z", fontsize = 10)
        
        scatter = ax.scatter(
            positions[0,:,0],
            positions[0,:,1], 
            positions[0,:,2], 
            c = 'blue', 
            s = 20
        )
        quivers = []
        for t in range(n_t):
            com_t = average(positions[t], axis = 0, weights = masses)
            scatter._offsets3d = (
                positions[t,:,0],
                positions[t,:,1],
                positions[t,:,2]
            )
            for quiver in quivers:
                quiver.remove()
            quivers = []
            quivers.append(
                ax.quiver(com_t[0], com_t[1], com_t[2],
                      gyration_tensor_eigenvectors[t,0,0],
                      gyration_tensor_eigenvectors[t,1,0],
                      gyration_tensor_eigenvectors[t,2,0],
                      color = 'black',
                      length = 2.5,
                      normalize = True
                )
            )
            quivers.append(
                ax.quiver(com_t[0], com_t[1], com_t[2], 
                      gyration_tensor_eigenvectors[t,0,1],
                      gyration_tensor_eigenvectors[t,1,1],
                      gyration_tensor_eigenvectors[t,2,1],
                      color = 'black',
                      length = 2.5,
                      normalize = True
                )
            )
            quivers.append(
                ax.quiver(com_t[0], com_t[1], com_t[2],
                      gyration_tensor_eigenvectors[t,0,2],
                      gyration_tensor_eigenvectors[t,1,2],
                      gyration_tensor_eigenvectors[t,2,2],
                      color = 'black',
                      length = 2.5,
                      normalize = True
                )
            )
            quivers.append(
                ax.quiver(com_t[0], com_t[1], com_t[2],
                      angular_momentum[t,0,0],
                      angular_momentum[t,0,1],
                      angular_momentum[t,0,2],
                      color = 'green',
                      length = 2.5,
                        normalize = True
                )
            )
            quivers.append(
                ax.quiver(com_t[0], com_t[1], com_t[2],
                      angular_velocity[t,0,0],
                      angular_velocity[t,0,1],
                      angular_velocity[t,0,2],
                      color = 'red',
                      length = 2.5,
                      normalize = True
                )
            )
            ax.set_title(f"t = {t}", fontsize = 10)
            clear_output(wait = True)
            display(fig)
            pause(0.01)
        close(fig)
        return None
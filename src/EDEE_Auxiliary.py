from matplotlib.pyplot import figure, grid, legend, xlabel, ylabel, title, plot, savefig, show
from lammps.formats import LogFile
from pandas import DataFrame
from pathlib import Path

class EDEE_Auxiliary:
    
    def __init__(self) -> None:
        self._directory_molecules = Path("molecules")
        self._directory_variables = Path("variables")
        self._directory_input = Path("input")
        self._directory_data = Path("data")
        self._directory_img = Path("img")
        self._directory_txt = Path("txt")
        self._directory_nvt = Path("nvt")
        self._directory_nve = Path("nve")
        self._directory_eq = Path("eq")
        self._directory_neq = Path("neq")
        self._filename_molecule = Path("molecule_1_2_4.lmp")
        self._filename_input_nvt = Path("nvt_input.in")
        self._filename_input_nve = Path("nve_input.in")
        self._filename_input_eq = Path("eq_input.in")
        self._filename_input_neq = Path("neq_input.in")
        self._filename_log_nvt = Path("nvt_log")
        self._filename_log_nve = Path("nve_log")
        self._filename_log_eq = Path("eq_log")
        self._filename_log_neq = Path("neq_log")
        self._filename_traj_nvt = Path("nvt_traj")
        self._filename_traj_nve = Path("nve_traj")
        self._filename_traj_eq = Path("eq_traj")
        self._filename_traj_neq = Path("neq_traj")
        self._filename_restart_nvt = Path("nvt_restart")
        self._filename_restart_nve = Path("nve_restart")
        return None
    
    def _delete_files(self) -> None:
        directories = [
            self._directory_variables,
            self._directory_img,
            self._directory_txt,
            self._directory_input / self._directory_nvt,
            self._directory_input / self._directory_nve,
            self._directory_input / self._directory_eq,
            self._directory_input / self._directory_neq,
            self._directory_data / self._directory_nvt,
            self._directory_data / self._directory_nve,
            self._directory_data / self._directory_eq,
            self._directory_data / self._directory_neq,
        ]
        for directory in directories:
            if not directory.exists():
                continue
            for file in directory.iterdir():
                if file.is_file() and file.suffix != ".in":
                    file.unlink()
        return None
    
    def _load_log_file(self, directory: Path, log_filename: str) -> dict:
        logs = {}
        log_path = directory / log_filename
        
        if not log_path.exists():
            print(f"El fichero {log_path} no existe.")
            return logs  
        
        log_file = LogFile(str(log_path))
        runs = log_file.runs
        index = 0
        for run in runs:
            df = DataFrame(run)
            logs[index] = df
            index += 1
        return logs
    
    def plot_df_data(self, df: DataFrame, x_col: str, y_col: str, is_df_log: bool) -> None:
        match is_df_log:
            case True:
                plot_color = '#242323'
            case False:
                plot_color = '#090e52'
        if x_col in df and y_col in df:
            x = df[x_col]
            y = df[y_col]
            figure();
            plot(
                x, 
                y, 
                linestyle = '--', 
                marker = 'o', 
                markerfacecolor = plot_color, 
                markeredgecolor = plot_color, 
                color = plot_color,
                linewidth = 0.5,
                markersize = 3, 
                label = y_col
            )
            title(f"{y_col} vs {x_col}", fontsize = 10)
            xlabel(x_col, fontsize = 10)
            ylabel(y_col, fontsize = 10)
            grid(True)
            legend()
            savefig(f"{self._directory_img}/{y_col}_vs_{x_col}.svg", format = 'svg')
            show()
        return None
    
    def get(self, variable: str, ) -> dict:
        class_name = self.__class__.__name__
        variable_name = f"_{class_name}__{variable}"
        if hasattr(self, variable_name):
            return getattr(self, variable_name)
        raise AttributeError(f"La variable __{variable} no existe.")
import os
import sys

def config_lammps_routes() -> None:
    bin_route = r"C:\Users\Propietario\AppData\Local\LAMMPS 64-bit 30Mar2026\bin"
    python_route = r"C:\Users\Propietario\AppData\Local\LAMMPS 64-bit 30Mar2026\python"
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(bin_route)
    if python_route not in sys.path:
        sys.path.append(python_route)
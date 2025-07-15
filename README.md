# ZP Polychromatic

This repository provides a minimal code base for simulating Fresnel zone plate imaging with monochromatic and polychromatic illumination. The heavy computations are distributed using `mpi4py`.

## Requirements
- Python 3.11+
- numpy
- scipy
- mpi4py
- matplotlib

## Running the simulation
Execute the parallel simulation with:

```bash
mpiexec -n <ranks> python execution.py
```

Result files are written to the working directory as `*.data` files. These can be analysed using the utilities in `analysis.py`.

All simulation constants are defined in `parameters.py`.

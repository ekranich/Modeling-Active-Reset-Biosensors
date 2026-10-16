The code in this repository is a generalizable framework for evaluating active-reset strategies in continuous molecular sensing

1. run_python_slurm_simulation.sh sends parallel jobs to run faster simulations
2. biosensor_simulation_general.py is the actual simulation framework with user parameters
3. after the simulation is complete, file_format_slurm_output.py concatenates files for processing
4. process_intermittent_pre_equil_plots.py can be used to visualize the data with similar formatting to the paper figures

Additionally, the Cramer-Rao.ipynb notebook can be used to evaluate the Cramer-Rao lower bound limits.

#!/usr/bin/bash

# Define Python and Hydra arguments as arrays
python_args=("-W" "ignore")
hydra_args=(
    "gen_landscape.py"
    "static/postprocess/plt/theme@postprocess.plt.theme=seismic_redstar"
    "grid.ny=500"
    "grid.nx=500"
    "grid.nt=4000"
    "src.n_horz=21"
    "src.n_deep=21"
    "gpu=cuda:1"
    "batch_size=250"
    "src.lower_left[0]=0.4"
    "src.lower_left[1]=0.6"
    "src.upper_right[0]=0.6"
    "src.upper_right[1]=0.4"
)

# Run the command with nohup and capture the PID
nohup python "${python_args[@]}" "${hydra_args[@]}" >nohup.out 2>&1 &

# Capture the PID of the background process
pid=$!

# Echo the PID to the terminal
echo "Process running in the background with PID: $pid"

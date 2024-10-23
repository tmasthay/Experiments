#!/usr/bin/bash

# Define Python and Hydra arguments as arrays
python_args=("-W" "ignore")
hydra_args=(
    "gen_landscape.py"
    "static/postprocess/plt/theme@postprocess.plt.theme=seismic_redstar"
    # "grid.ny=500,3400"
    # "grid.nx=500,700"
    # "grid.nt=2000,3000,4000"
    "grid.ny=700"
    "grid.nx=700"
    "grid.nt=3000"
    "src.n_horz=11"
    "src.n_deep=11"
    "gpu='cuda:0'"
    "batch_size=50"
    "src.lower_left=[0.4,0.6]"
    "src.upper_right=[0.6,0.4]"
    # "rt/vp=hom"
    # "rt/vs=hom"
    # "rt/rho=hom"
    "dupe=true"
    "editor=null"
    # "--multirun"
)

# Run the command with nohup and capture the PID
# nohup python "${python_args[@]}" "${hydra_args[@]}" 2>&1 | tee nohup.out &
NOHUP_MODE="1"

# if $1 == "0" then run directly
if [ "$NOHUP_MODE" == "0" ]; then
    python "${python_args[@]}" "${hydra_args[@]}"
    exit 0
else
    echo "${hydra_args[@]}"
    nohup python "${python_args[@]}" "${hydra_args[@]}" >nohup.out 2>&1 &
fi

# Capture the PID of the background process
pid=$!

# Echo the PID to the terminal
echo "$pid"

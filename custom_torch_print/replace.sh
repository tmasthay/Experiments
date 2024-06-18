#!/bin/bash

TORCH_PATH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__))") || {
    echo "Failed to get torch path"
    exit 1
}

cd $TORCH_PATH
echo "WARNING: Replacing torch._tensor_str._tensor_str with custom version"
echo "Original file will be backed up as _tensor_str.py.bak"
mv _tensor_str.py _tensor_str.py.bak
wget https://raw.githubusercontent.com/tmasthay/Experiments/main/custom_torch_print/_tensor_str.py
if [ $? -ne 0 ]; then
    echo "Failed to download _tensor_str.py. Moving original file back"
    mv _tensor_str.py.bak _tensor_str.py
    exit 1
else
    echo "Successfully replaced _tensor_str.py"
fi
exit 0

#!/bin/bash

# source ~/start-pyenv
cd ~/xx_path/
# source ./python-env/bin/activate
cd ./scripts

chmod +x a3_mda_proj.r

## set params
param_path=../param_file1.csv

for idx in {1..2}; do
    # Preprocess & Project
    python a1_prelim.py $param_path $idx   
    matlab -nodisplay -nosplash -nodesktop -r "cd ~/xx_path/; a2_runSIFTED('$param_path',$idx,-1,''); exit;"

    # Predictions eval
    ./a3_mda_proj.r $param_path $idx
    python a3b_evalRproj.py $param_path $idx MDA mda_proj.csv

    # Footprint
    matlab -nodisplay -nosplash -nodesktop -r "a4_runTRACE('$param_path',$idx,'','mda_proj.csv'); exit;"
    
    # Instance selection
    python a5_instance_sel.py $param_path $idx mda -mda
    
done

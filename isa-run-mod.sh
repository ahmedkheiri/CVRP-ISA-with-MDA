#!/bin/bash

# source ~/start-pyenv
cd ~/xx_path/
# source ./python-env/bin/activate
cd ./scripts

chmod +x a3_mda_proj.r

## set params
param_path=../param_file1.csv
mfol=modded-8/

for idx in {1..2}; do
    # Preprocess & SIFTED
    python a6_prep_mod.py $param_path $idx $mfol 1
    matlab -nodisplay -nosplash -nodesktop -r "cd ~/xx_path/; a2_runSIFTED('$param_path',$idx,8,'$mfol'); exit;"

    # MDA project & predict
    ./a3_mda_proj.r $param_path $idx $mfol
    python a3b_evalRproj.py $param_path $idx MDA mda_proj.csv -fol $mfol
    
    # Footprint
    matlab -nodisplay -nosplash -nodesktop -r "a4_runTRACE('$param_path',$idx,'$mfol','mda_proj.csv'); exit;"
    
done   


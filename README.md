# CVRP-ISA-with-MDA

Please cite the following paper if you used any of these resources: 

Danielle Notice, Hamed Soleimani, Nicos G. Pavlidis, Ahmed Kheiri and Mario Andrés Muñoz (in press) Instance space analysis for the capacitated vehicle routing problem with mixture discriminant analysis. GECCO.


This repository contains the source code and data for the work on instance space analysis for the capacitated vehicle routing problem using mixture discriminant analysis to be presented at GECCO 2025 [1].

Performance data is generated using two state-of-the-art metaheuristic solvers for the CVRP [HGS](https://github.com/vidalt/HGS-CVRP) and [FILO](https://github.com/acco93/filo) for two time budgets:
- `cost_2s` - fixed 120 seconds,
- `cost_2f` - 120 seconds per 100 customers.

`instance generator.py` is a novel CVRP generator which modifies existing problem instances using the outliers of node clusters to produce relevant new CVRP instances. The metadata for instances generated using this generator are included [here](https://github.com/danotice/CVRP-ISA-with-MDA/tree/main/metadata/modded).

Mixture discriminant analysis (MDA) is used to obtain both a low dimensional representation of the instance space and a classifier of algorithm performance. The `scripts` folder contains code for the following:
- Preprocessing (`a1_prelim.py`, `a6_prep_mod.py`)
- Feature selection and footprint construction as in the original instance space analysis framework (`a2_runSIFTED.m`, `a4_runTRACE.m`
- Projection and prediction using MDA and evaluation (`a3_mda_proj.r`, `a3b_evalRproj.py`)
- Selection of instances to be modified for new instance generation (`a5_instance_sel.py`)

These scripts were run in sequence as in the bash scripts `isa-run-mda.sh` and `isa-run-mod.sh` for the initial and modified metadata respectively.

[1] Notice, D., H. Soleimani, N. G. Pavlidis, A. Kheiri and M. A. Muñoz (2025). Instance Space Analysis of the Capacitated Vehicle Routing Problem with Mixture Discriminant Analysis. <ins>Genetic and Evolutionary Computation Conference (GECCO '25)</ins>. Malaga, Spain, ACM.

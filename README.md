# align-mada-protein

## Nimbus Approach Replication
1. Structure Data Gathering
* Tools: ColabFold/AlphaFold2, ESMFold, or AlphaFold DB, Benchling API
* Object: mutant sequences structures, target enzyme structure
* Optional: gathering wildtype structure from BLAST NCBI
2. Substrate Parameterization
* Tools: PyRosetta, structure library?
* Object: substrate structure, PARAMS file for substrate
* Optional: parameterize any cofactors or metal ions --> why tho?
3. Mechanism & Pre-Transition State Modelling
* Prerequisite Knowledge: catalytic mechanism of enzyme, key catalytic residues, reaction coordinate, transition state geomertry, the pH dependent mechanisms 
* Tools: 
    * pyrosetta : load structure pdb enzym, identify active site, calculate constraints (atom distance, dihedral, coordinate), pH specific protonation states
* Object: 
* Optional:
4. Molecular Docking
* Tools: 
    * DiffDock: get top N poses and each confidance score, visual inspections
    * PyRosetta: docking refinement protocol
5. Energy Minimization & Parameter Gathering
6. Score Normalization & Linear Combination

## Discussed Approach
1. Gathering parameter using BioPython (net charge, extinction coeff.) and pH optimum prediction (model inference)
2. Kinetic modelling approach
3. Scoring activity and Coefficient of kinetic modelling

---------------------------------------------------------------------
TASK BREAKDOWN: (from the easiest to some hassle)
- Code for parameter gathering (biopython, pyrosetta, external model)
    - run through wildtype and mutants
    - data analysis check
- Docking
- Structure data gathering
- Kinetic modelling
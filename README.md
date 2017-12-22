# CyBayes
## Features

- Performs MCMC based phylogenetics.
- Can handle multistate characters.
- Can handle polymorphism (or synonymous states of a concept).
- Can handle Jukes-Cantor, Felsenstein-81 models.
- Handles only Rooted trees like in linguistics. So, the trees will have branch lengths and a root.
- Handles trees like Phylo class in ape package of R.

## Parameter space moves
- Performs tree moves using external SPR and NNI.
- Branch lengths are sampled according to a exponential distribution.
- Substitution parameters are sampled using Dirichlet and DualSampler move.


## Usage
> Compile code using ```python setup.py build_ext --inplace```
```
 usage: mat_mcmc.py [-h] [-i INPUT_FILE] [-m MODEL] [-n N_GEN] [-t THIN]
                   [-o OUTPUT_FILE] [-d bin]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input_file INPUT_FILE
                        Input a file in Phylip format with taxa and characters
                        separated by a TAB character
  -m MODEL, --model MODEL
                        JC/F81/GTR
  -n N_GEN, --n_gen N_GEN
                        Number of generations
  -t THIN, --thin THIN  Number of generations after to print to file
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        Name of the out file prefix
  
  ```
  Linguistic datasets are quite different from biological morphological datasets. A Jukes-Cantor model where all the transition rates are the same is useful for estimating trees with branch lengths.

Example Usage:
```
python3 mat_mcmc.py -i data/PIE.phy -m JC -n 10000 -t 100 -o temp
```  
# Output
- Returns a .params file that can be viewed in [Tracer] (http://tree.bio.ed.ac.uk/software/tracer/). The .params file consists of the likelihood and the tree length for each state.
- Returns a .trees file that can be viewed in [FigTree] (http://tree.bio.ed.ac.uk/software/figtree/).

# Tested with
- Python3
- Scipy (0.18.1)
- Numpy (1.12.0)
 

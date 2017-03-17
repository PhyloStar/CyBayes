# PyBayes

- Performs MCMC based phylogenetics.
- Can handle multistate characters.
- Only requires Python3 and Numpy and Scipy.
- Can handle Jukes-Cantor, F81, and GTR models.
- Handles only Rooted trees.
- Handles trees like Phylo class in ape package of R.

# Usage
```
 python3 mcmc.py -h
usage: mcmc.py [-h] [-i INPUT_FILE] [-m MODEL] [-n N_GEN] [-t THIN]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input_file INPUT_FILE
                        Input a file in Phylip format with taxa and characters
                        separated by a TAB character
  -m MODEL, --model MODEL
                        JC/F81/GTR
  -n N_GEN, --n_gen N_GEN
                        Number of generations
  -t THIN, --thin THIN  Number of generations
  
  ```
# Requirements
- Python3, Scipy, and Numpy
 

# SmileRNN

Generative RNN with optional policy gradient optimization to generate novel, unique molecules with optimized QED(1) scores, which indicate their pharmacological potential.

1. Bickerton, G. Richard, et al. "Quantifying the chemical beauty of drugs." Nature chemistry 4.2 (2012): 90-98.

Setup: conda env create --name SmileRNN --file=env.yml

Training data: "\n"-seperated SMILES strings in data/ directory, data accessed from ChEMBL (https://www.ebi.ac.uk/chembl/)

To train, run train.py

To optimize, run optimize.py with desired training weights

To generate molecules, run sample.py with desired training weights

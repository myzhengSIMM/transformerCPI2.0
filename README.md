# TransfomerCPI2.0
  We only disclose the inference models. TransformerCPI2.0 is based on TransformerCPI whose codes are all released. The details of TransformerCPI2.0 are described in our paper https://doi.org/10.1038/s41467-023-39856-w which is now published on Nature communications. Trained models are available at present.
  
# Abstract
  Drug development based on target proteins has been a successful approach in recent decades. However, the conventional structure-based drug design (SBDD) pipeline is a complex, human-engineered process with multiple independently optimized steps. Here, we propose a sequence-to-drug concept for computational drug design based on protein sequence information by end-to-end differentiable learning. We validate this concept in three stages. First, we design TransformerCPI2.0 as a core tool for the concept, which demonstrates generalization ability across proteins and compounds. Second, we interpret the binding knowledge that TransformerCPI2.0 learned. Finally, we use TransformerCPI2.0 to discover new hits for challenging drug targets, and identify new target for an existing drug based on an inverse application of the concept. Overall, this proof-of-concept study shows that the sequence-to-drug concept adds a perspective on drug design. It can serve as an alternative method to SBDD, particularly for proteins that do not yet have high-quality 3D structures available.
  ![image](https://github.com/995884191/png/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-12-04%20191421.jpg?raw=true.png)
  
## Setup and dependencies 
`environment.yaml` is the conda environment of this project.

## Inference
`predict.py` makes the inference, the input are protein sequence and compound SMILES. `featurizer.py` tokenizes and encodes the protein sequence and compounds. `mutation_analysis.py` conducts drug mutation analysis to predict binding sites. `substitution_analysis.py` conducts substitution analysis.

## Trained models
Trained models is now available freely at https://drive.google.com/drive/folders/1X7i1eO-EykCQcvqMeWeB7QXT3E9eLG08?usp=sharing. The current open source version only aims to reproduce the results reported in the article, so the inference speed is limited.

## Requirements
python = 3.8.8 

pytorch = 1.9 

tape-proteins = 0.5 

rdkit = 2021.03.5 

numpy = 1.19.5 

scikit-learn = 0.24.1 


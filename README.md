# TransfomerCPI2.0:Sequence-based drug design as a concept in computational drug design
## Background
  We only disclose the inference models. TransformerCPI2.0 is based on TransformerCPI whose codes are all released. The details of TransformerCPI2.0 are described in our paper https://doi.org/10.1038/s41467-023-39856-w which is now published on Nature communications. Trained models are available at present.
## Abstract of article
  Drug development based on target proteins has been a successful approach in recent decades. However, the conventional structure-based drug design (SBDD) pipeline is a complex, human-engineered process with multiple independently optimized steps. Here, we propose a sequence-to-drug concept for computational drug design based on protein sequence information by end-to-end differentiable learning. We validate this concept in three stages. First, we design TransformerCPI2.0 as a core tool for the concept, which demonstrates generalization ability across proteins and compounds. Second, we interpret the binding knowledge that TransformerCPI2.0 learned. Finally, we use TransformerCPI2.0 to discover new hits for challenging drug targets, and identify new target for an existing drug based on an inverse application of the concept. Overall, this proof-of-concept study shows that the sequence-to-drug concept adds a perspective on drug design. It can serve as an alternative method to SBDD, particularly for proteins that do not yet have high-quality 3D structures available.
  ![image](https://github.com/995884191/png/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-12-04%20191421.jpg?raw=true.png)
  
## Setup and dependencies 
Set up the conda environment of this project.
`conda env create -f environment.yaml`

## Train
The data format refers to the file `chembl.zip` in release. 
The first step for model training is to tokenize and encode the protein sequence and compounds by `python train_featurizer.py`.
Then, use these feature for model training by `python main_amp.py`.

## Trained models
Trained models is now available freely at https://drive.google.com/drive/folders/1X7i1eO-EykCQcvqMeWeB7QXT3E9eLG08?usp=sharing. The current open source version only aims to reproduce the results reported in the article, so the inference speed is limited.

## Inference
Make the inference between the protein sequence and compound SMILES.
`python predict.py`

Conduct drug mutation analysis to predict binding sites. 
`python mutation_analysis.py`

Conduct substitution analysis.
`python substitution_analysis.py`

## Requirements
python = 3.8.8 

pytorch = 1.9 

tape-proteins = 0.5 

rdkit = 2021.03.5 

numpy = 1.19.5 

scikit-learn = 0.24.1 

## Related Efforts
TransformerCPI: https://github.com/myzhengSIMM/transformerCPI

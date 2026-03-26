# Optimizing Bayesian Neural Networks for Genomic Prediction

[![Paper](https://img.shields.io/badge/Paper-Canadian%20AI%202026-blue)](main.pdf)
[![Conference](https://img.shields.io/badge/Conference-The%2039th%20Canadian%20Conference%20on%20Artificial%20Intelligence-green)](https://caiac.ca/en/conferences/canadian-ai-2026)

This repository contains the official implementation of the paper **"Optimizing Bayesian Neural Networks for Genomic Prediction: A Multi Phenotype Study on Feature Selection and Architecture"**, presented at the 39th Canadian Conference on Artificial Intelligence (Canadian AI 2026).

## Authors
- **Raeein Bagheri** - University of Guelph, Department of Computer Science
- **Yan Yan** - University of Guelph, Department of Computer Science

## Abstract
Genome-wide association studies (GWAS) scan the genome for genetic variants, typically single nucleotide polymorphisms (SNPs), whose alleles are associated with phenotypic variation across individuals. GWAS and genomic prediction face a core challenge: learning from extremely high-dimensional genotype matrices under limited sample sizes. Bayesian neural networks offer uncertainty aware prediction and the capacity to represent non-linear genetic effects, but their practical performance depends on feature selection and architectural choices that interact with the inference mechanism. This paper presents an empirical study that improves a Bayesian neural network pipeline for genomic prediction by tuning input selection strategies, network depth and width, and activation functions under Hamiltonian Monte Carlo inference. We validate the pipeline across three distinct datasets: Ear Height in the Tassel tutorial dataset, flowering time (FT10) in Arabidopsis thaliana, and top second leaf length (TSLL) in foxtail millet. Under a five-fold cross-validation protocol, we compare three approaches: a deterministic ResNet baseline, a standard Bayesian neural network, and an optimized Bayesian neural network produced through targeted tuning. Results show that while feature selection is necessary for stable learning, the optimal configuration varies by phenotype. Across all datasets, the optimized BNN achieves superior predictive performance and consistent uncertainty calibration, outperforming the standard BNN and the deterministic baseline.

## Project Structure
```text
.
├── bnn_elastic_net_five_folds.py        # Main BNN implementation with Elastic Net CV
├── PreliminaryFeatureSelection_proposed.R # Preliminary feature selection and data parsing (R)
├── install-packages.R                  # R dependency installer
├── jobsch6.sh                          # SLURM submission script for HPC
├── requirements.txt                    # Python dependencies
├── main.pdf                            # The research paper
└── Geno_merged_EarHT_df_prop.csv       # Sample dataset (Ear Height)
```

## Getting Started

### Prerequisites
The project uses both Python and R for data processing and model training.

#### Python Environment
We recommend using Python 3.10+. Install the required packages using pip:
```bash
pip install -r requirements.txt
```
Key Python dependencies:
- **JAX** & **NumPyro**: For high-performance Bayesian inference with Hamiltonian Monte Carlo (HMC).
- **Scikit-learn**: For Elastic Net feature selection and cross-validation utilities.
- **Pandas** & **NumPy**: For data manipulation.

#### R Environment
To run the preliminary feature selection scripts, you will need R 4.4+. You can install the required R packages using:
```bash
Rscript install-packages.R
```

### Usage

#### 1. Running the BNN Pipeline
The main entry point is `bnn_elastic_net_five_folds.py`. This script performs feature selection using Elastic Net, followed by five-fold cross-validation of the Bayesian Neural Network using the No-U-Turn Sampler (NUTS).

```bash
python bnn_elastic_net_five_folds.py
```

#### 2. HPC Execution (SLURM)
For training on a cluster (e.g., Compute Canada), use the provided SLURM script:
```bash
sbatch jobsch6.sh
```

#### 3. Preliminary Feature Selection (Optional)
The R script `PreliminaryFeatureSelection_proposed.R` can be used for initial filtering of high-dimensional genotype data (`.ped`/`.pheno` formats).

## Datasets
The study evaluates the pipeline on three plant phenotypes:
1. **Tassel (Ear Height)**: N=282, P=54,488 SNPs.
2. **Arabidopsis (FT10)**: N=1,050, P=194,000 SNPs.
3. **Foxtail Millet (TSLL)**: N=827, P=120,000 SNPs.

## Methodology
- **Inference**: Hamiltonian Monte Carlo (HMC) using the No-U-Turn Sampler (NUTS) provided by NumPyro.
- **Feature Selection**: Comparison between ANOVA filtering and Elastic Net (L1/L2) penalized regression.
- **Architectural Tuning**: Exploration of depth (1-4 layers), width (32-256 units), and smooth activation functions (GELU, Leaky ReLU).

## Citation
If you use this code or findings in your research, please cite our paper:

```bibtex
@inproceedings{bagheri2026optimizing,
  title={Optimizing Bayesian Neural Networks for Genomic Prediction: A Multi Phenotype Study on Feature Selection and Architecture},
  author={Bagheri, Raeein and Yan, Yan},
  booktitle={Proceedings of the 39th Canadian Conference on Artificial Intelligence},
  year={2026}
}
```

## Acknowledgments
This research was supported by the University of Guelph and grants from the Natural Sciences and Engineering Research Council of Canada (NSERC).

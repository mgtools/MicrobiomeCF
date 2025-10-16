# MicrobiomeCF

# MicroKPNN_cf: Confounder-Free Metagenomic Classification

**MicroKPNN_cf** is a PyTorch-based framework designed to tackle confounder effects in metagenomic classification tasks. It integrates **hierarchical masking** and **correlation-based losses** to reduce dependency on confounders, ensuring robust classification.

## Quick Start

```bash
# Clone the repository and navigate into the project directory
git clone https://github.com/YourUsername/MicroKPNN_cf.git
cd MicroKPNN_cf

# Create and activate the conda environment
conda env create -f environment.yml
conda activate microbiome_cf

# Prepare your data: Place metadata and abundance CSV files in the `dataset/` folder and change the cofing script based on your data.

# Train the model (FNN confounder free)
python FNN_encoder_confounder_free_lib/main.py

# Train the model (MicroKPNN confounder free)
python MicroKPNN_encoder_confounder_free_lib/run_pipeline.py
python MicroKPNN_encoder_confounder_free_lib/main.py

# Generate SHAP-based explainability results
python MicroKPNN_encoder_confounder_free_lib/explainability.py

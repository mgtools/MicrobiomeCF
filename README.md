# MicrobiomeCF

# MicroKPNN_cf: Confounder-Free Metagenomic Classification

**MicroKPNN_cf** is a PyTorch-based framework designed to tackle confounder effects in metagenomic classification tasks. It integrates **hierarchical masking** and **correlation-based losses** to reduce dependency on confounders, ensuring robust classification.

## Quick Starttt

```bash
# Clone the repository and navigate into the project directory
git clone https://github.com/YourUsername/MicroKPNN_cf.git
cd MicroKPNN_cf

# Create and activate the conda environment
conda env create -f environment.yml
conda activate MicroKPNN_cf

# Prepare your data: Place metadata and abundance CSV files in the `data/` folder.
# Update the file paths in `scripts/main.py` if needed.

# Train the model
python scripts/MicroKPNN_CF_main.py

# Generate SHAP-based explainability results
python src/explainability.py

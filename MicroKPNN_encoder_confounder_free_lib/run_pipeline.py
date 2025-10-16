#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys

from config import config

def main():
    # 1) Read paths & taxonomy level from config.py
    inp = config["data"]["train_abundance_path"]
    taxonomy = str(config["taxonomy_level"])      # add "taxonomy_level": <0–5> in your config.py
    out = "Results/MicroKPNN_encoder_confounder_free_plots/required_data/"

    # 2) Prepare output folder
    os.makedirs(out, exist_ok=True)

    # 3) Run taxonomy_info.py → species_info.pkl
    print("[1/3] Running taxonomy_info.py …")
    subprocess.check_call([
        sys.executable, "MicroKPNN_encoder_confounder_free_lib/taxonomy_info.py",
        "--inp", inp,
        "--out", out
    ])

    # 4) Run create_edges.py → EdgeList.csv + node lists
    print("[2/3] Running create_edges.py …")
    subprocess.check_call([
        sys.executable, "MicroKPNN_encoder_confounder_free_lib/create_edges.py",
        "--inp", inp,
        "--taxonomy", taxonomy,
        "--out", out
    ])


    # # 5) Run your main training & plotting script
    # print("[3/3] Running main.py …")
    # subprocess.check_call([sys.executable, "MicroKPNN_encoder_confounder_free_lib/main.py"])

    # print("\n✅ Pipeline complete — all outputs in", out)

if __name__ == "__main__":
    main()

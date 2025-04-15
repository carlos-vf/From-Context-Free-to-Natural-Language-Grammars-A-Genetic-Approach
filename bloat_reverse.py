import os
import pandas as pd
import re

# Folder containing the .txt files
FOLDER_PATH = "results/esp/bloat_variation/" 

# Fitness columns to update
FITNESS_COLUMNS = [
    "avg_fitness",
    "best_fitness_gen",
    "best_fitness_all"
]

def process_file(filepath, alpha):
    # Read the file as a DataFrame
    df = pd.read_csv(filepath, sep="\t")
    
    for col in FITNESS_COLUMNS:
        if col in df.columns:
            df[col] = (df[col] + alpha * df["avg_size"]) / (1 - alpha)

    df.to_csv(filepath, sep="\t", index=False)

def extract_last_float(filename):
    matches = re.findall(r"\d+(?:\.\d+)?", filename)
    return float(matches[-1]) if matches else None

def main():
    for filename in os.listdir(FOLDER_PATH):
        if filename.endswith(".txt") or '.' not in filename:
            alpha = extract_last_float(filename)
            filepath = os.path.join(FOLDER_PATH, filename)
            process_file(filepath, alpha)

if __name__ == "__main__":
    main()
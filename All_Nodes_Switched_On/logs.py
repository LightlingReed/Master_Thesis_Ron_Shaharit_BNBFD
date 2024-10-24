# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:26:40 2024

@author: ronsh
"""

import os
import pandas as pd

# Constants
LATENCY_DATA_FILE = "latency_data.csv"
CPU_USAGE_DATA_FILE = "cpu_usage_data.csv"
LATENCY_LOGS_DIR = "latency_logs"
CPU_USAGE_LOGS_DIR = "cpu_usage_logs"
SEEDS = range(0, 30)  # Define the seed range

# Function to create directories and split data by seed
def split_data_by_seed_and_create_dirs(data_file, log_dir, log_type):
    """
    Splits the data in the CSV file by Seed, creates directories for each seed, and saves individual per-seed log files.
    """
    try:
        # Load the data
        df = pd.read_csv(data_file)

        # Check if the required columns exist
        if not all(col in df.columns for col in ["Iteration", "Seed", "Value", "Scheduler"]):
            raise ValueError(f"Missing required columns in {data_file}. Expected: Iteration, Seed, Value, Scheduler")

        # Iterate over each seed and save a separate file in its own directory
        for seed in SEEDS:
            seed_df = df[df['Seed'] == seed]
            if seed_df.empty:
                print(f"Warning: No data found for seed {seed} in {data_file}. Skipping seed.")
                continue
            
            # Create directory for the current seed
            seed_dir = os.path.join(log_dir, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)

            # Create the output file path within the seed directory
            output_file = os.path.join(seed_dir, f"{log_type}_{seed}.csv")
            seed_df.to_csv(output_file, index=False)
            print(f"Saved {log_type} data for seed {seed} to {output_file}")

    except FileNotFoundError:
        print(f"Error: {data_file} not found.")
    except Exception as e:
        print(f"Error processing {data_file}: {e}")

# Split latency data into per-seed directories and log files
split_data_by_seed_and_create_dirs(LATENCY_DATA_FILE, LATENCY_LOGS_DIR, "latency")

# Split CPU usage data into per-seed directories and log files
split_data_by_seed_and_create_dirs(CPU_USAGE_DATA_FILE, CPU_USAGE_LOGS_DIR, "cpu_usage")

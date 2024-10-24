# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:24:03 2024

@author: ronsh
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the CSV files
csv_files = {
    "energy": "energy_data.csv",
    "latency": "latency_data.csv",
    "cpu_usage": "cpu_usage_data.csv"
}

# Check if all files exist
for key, csv_file in csv_files.items():
    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"{csv_file} not found. Please make sure the file exists in the correct directory.")

# Set plot style
sns.set(style="whitegrid")

# Create directory for plots if it doesn't exist
output_directory = "scheduler_combined_plots"
os.makedirs(output_directory, exist_ok=True)

# Iterate over each CSV file to generate plots for each seed
for key, csv_file in csv_files.items():
    # Load the data
    df = pd.read_csv(csv_file)

    # Check if all required columns are present
    required_columns = ["Value", "Iteration", "Seed", "Scheduler"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file {csv_file} must contain columns: {', '.join(required_columns)}.")

    # Get unique seeds
    unique_seeds = df['Seed'].unique()

    # Iterate through each seed and plot the data
    for seed in unique_seeds:
        seed_df = df[df['Seed'] == seed]
        # Filter data to include only every 5th iteration for clarity
        seed_df = seed_df[seed_df['Iteration'] % 5 == 0]

        # Initialize the figure
        plt.figure(figsize=(15, 8))

        # Plot the data for each scheduler, colored by scheduler type, add transparency (alpha) for overlapping lines
        sns.lineplot(
            data=seed_df,
            x="Iteration",
            y="Value",
            hue="Scheduler",
            style="Scheduler",
            markers=True,
            dashes=False,
            palette="tab10",
            linewidth=2,
            alpha=0.7  # Adjust transparency for overlapping lines
        )

        # Add labels and title
        plt.xlabel("Iteration")
        if key == "energy":
            plt.ylabel("Energy Usage (Watts per Second)")
            plt.title(f"Combined Energy Usage for Different Schedulers (Seed {seed})")
        elif key == "latency":
            plt.ylabel("Latency (Number of Pending Items)")
            plt.title(f"Combined Latency for Different Schedulers (Seed {seed})")
        elif key == "cpu_usage":
            plt.ylabel("CPU Usage (Units)")
            plt.title(f"Combined CPU Usage for Different Schedulers (Seed {seed})")

        plt.legend(title="Scheduler")

        # Define the output path for the plot
        plot_filename = f"{key}_combined_usage_seed_{seed}.png"
        plot_output_path = os.path.join(output_directory, plot_filename)

        # Save the plot to a file
        plt.savefig(plot_output_path)
        plt.close()
        print(f"Plot saved to {plot_output_path}")

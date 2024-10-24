# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 23:58:04 2024

@author: ronsh
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
SCHEDULERS = ['1D', '4D', 'Defult']
SEEDS = range(0, 30)  # Seeds 0 to 29
TIME_STEPS = 200
ENERGY_LOGS_DIR = 'energy_logs'  # Root directory containing energy logs
LATENCY_LOGS_DIR = 'latency_logs'  # Latency logs directory
CPU_USAGE_LOGS_DIR = 'cpu_usage_logs'  # CPU usage logs directory
AVERAGED_LOGS_DIR = 'averaged_logs'  # Directory containing averaged logs
OUTPUT_ENERGY_PLOT = 'aggregated_energy_comparison.png'
OUTPUT_LATENCY_PLOT = 'aggregated_latency_comparison.png'
OUTPUT_CPU_PLOT = 'aggregated_cpu_usage_comparison.png'
POINT_DISPLAY_INTERVAL = 5  # Display markers every 5 iterations

# Function to read logs
def read_logs(file_path, column_name):
    """
    Generic function to read energy, latency, or CPU logs, ensuring the correct column exists.
    """
    if not os.path.isfile(file_path):
        print(f"Warning: File {file_path} does not exist.")
        return None
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            print(f"Warning: '{column_name}' column missing in {file_path}. Available columns: {df.columns.tolist()}")
            return None
        if len(df) < TIME_STEPS:
            print(f"Warning: Not enough iterations in {file_path}. Expected {TIME_STEPS}, got {len(df)}.")
            return None
        return df[column_name].values[:TIME_STEPS]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Functions for specific log types
def read_energy_logs(scheduler, seed):
    file_path = os.path.join(ENERGY_LOGS_DIR, scheduler, f'energy_{scheduler}_seed{seed}.csv')
    return read_logs(file_path, 'Energy_Consumed')

def read_latency_logs(scheduler, seed):
    if scheduler == 'Defult':
        return np.zeros(TIME_STEPS)  # Force all latency values for Defult to be zero
    file_path = os.path.join(LATENCY_LOGS_DIR, f'seed_{seed}', f'latency_{seed}.csv')  # Correct path format
    return read_logs(file_path, 'Value')

def read_cpu_logs(scheduler, seed):
    file_path = os.path.join(CPU_USAGE_LOGS_DIR, f'seed_{seed}', f'cpu_usage_{seed}.csv')  # Correct path format
    return read_logs(file_path, 'Value')

# Aggregating data (energy, latency, or CPU usage)
def aggregate_data(scheduler, read_func, log_type):
    data_matrix = []
    for seed in SEEDS:
        data = read_func(scheduler, seed)
        if data is not None:
            data_matrix.append(data)
        else:
            print(f"Skipping seed {seed} for scheduler {scheduler} due to previous warnings.")
    
    if not data_matrix:
        print(f"No {log_type} data found for scheduler {scheduler}.")
        return np.zeros(TIME_STEPS)  # Return an array of zeros if no data is found

    data_matrix = np.array(data_matrix)  # Shape: (num_seeds, TIME_STEPS)
    average_data = np.mean(data_matrix, axis=0)  # Average over seeds

    # Force Defult latency to zero if the log_type is latency
    if scheduler == 'Defult' and log_type == 'latency':
        average_data = np.zeros(TIME_STEPS)

    return average_data

# Plotting aggregated data
def plot_aggregated_data(average_data, ylabel, output_plot):
    plt.figure(figsize=(14, 8))

    # Define distinct colors and markers for each scheduler
    scheduler_styles = {
        '1D': {'color': 'blue', 'marker': 'o'},
        '4D': {'color': 'green', 'marker': 's'},
        'Defult': {'color': 'red', 'marker': '^'}
    }

    for scheduler, avg_data in average_data.items():
        iterations = np.arange(1, TIME_STEPS + 1)
        style = scheduler_styles.get(scheduler, {'color': 'black', 'marker': 'x'})

        scatter_iterations = iterations[::POINT_DISPLAY_INTERVAL]
        scatter_values = avg_data[::POINT_DISPLAY_INTERVAL]

        # Plot lines connecting scatter points
        plt.plot(scatter_iterations, scatter_values, color=style['color'], linestyle='-', linewidth=1.5, label=f'{scheduler} Scheduler')

        # Plot scatter points
        plt.scatter(scatter_iterations, scatter_values, color=style['color'], marker=style['marker'], edgecolors='k', s=50, zorder=5)

    plt.title(f'Average {ylabel} Over Time for Different Schedulers', fontsize=16)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(f'Average {ylabel} (Units)', fontsize=14)
    plt.legend(title="Scheduler", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(range(0, TIME_STEPS + 1, 10))  # Adjust x-ticks for readability
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.show()
    print(f"Plot saved as '{output_plot}'.")

# Main function
def main():
    # Aggregating energy
    print("\nAggregating energy data for all schedulers...")
    average_energy = {scheduler: aggregate_data(scheduler, read_energy_logs, 'energy') for scheduler in SCHEDULERS}
    plot_aggregated_data(average_energy, 'Energy Consumption', OUTPUT_ENERGY_PLOT)

    # Aggregating latency
    print("\nAggregating latency data for all schedulers...")
    average_latency = {scheduler: aggregate_data(scheduler, read_latency_logs, 'latency') for scheduler in SCHEDULERS}
    plot_aggregated_data(average_latency, 'Latency', OUTPUT_LATENCY_PLOT)

    # Aggregating CPU usage
    print("\nAggregating CPU usage data for all schedulers...")
    average_cpu_usage = {scheduler: aggregate_data(scheduler, read_cpu_logs, 'cpu usage') for scheduler in SCHEDULERS}
    plot_aggregated_data(average_cpu_usage, 'CPU Usage', OUTPUT_CPU_PLOT)

if __name__ == "__main__":
    main()

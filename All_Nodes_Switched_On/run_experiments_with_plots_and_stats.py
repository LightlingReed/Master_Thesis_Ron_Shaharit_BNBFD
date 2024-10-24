# run_experiments_with_plots_and_stats.py

import numpy as np
import pandas as pd
import subprocess
import os
import logging
import re
import matplotlib.pyplot as plt

# Configure logging to capture detailed debug information
logging.basicConfig(
    filename='simulation.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Set to DEBUG to capture detailed logs
)

# Define constants
NUM_EXPERIMENTS = 30  # Run simulations for seeds 0 to 29
RESULT_FILE = 'scheduler_results.csv'  # Output CSV file for storing results
STATS_FILE = 'scheduler_summary_stats.csv'  # CSV file for summary statistics
PLOT_OUTPUT_DIR = "scheduler_combined_plots"  # Directory to store plots

# Generate unique seeds for the experiments
unique_seeds = list(range(NUM_EXPERIMENTS))  # Seeds from 0 to 29

# List of scheduler scripts
scheduler_files = [
    'batch_bin_packing__1D.py',
    'batch_bin_packing__4D.py',
    'batch_bin_packing__Defult.py'
]

# Function to run scheduler script and capture the results
def run_scheduler(script_name, seed):
    try:
        logging.info(f"Starting {script_name} with seed {seed}")
        process = subprocess.Popen(
            ['python', script_name, '--seed', str(seed)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capture stderr separately
            text=True
        )
        # Wait for the process to complete and capture the output
        stdout, stderr = process.communicate()

        logging.debug(f"Output from {script_name} with seed {seed}:\n{stdout}")
        logging.debug(f"Error output from {script_name} with seed {seed}:\n{stderr}")

        if not stdout.strip():
            logging.error(f"No output produced by {script_name} with seed {seed}.")
            return None

        return stdout  # Return only stdout for parsing
    except Exception as e:
        logging.error(f"Failed to run {script_name} with seed {seed}: {e}")
        return None


# Updated parse_output function to include node-level data
def parse_output(output):
    parsed_data = []
    node_utilization_data = []
    lines = output.strip().splitlines()
    energy_consumed = None
    parsing_started = False
    parsing_node_utilization = False

    for line in lines:
        logging.debug(f"Parsing line: {line}")
        if "Total energy consumed:" in line:
            try:
                energy_consumed = float(re.search(r"Total energy consumed:\s*([\d.]+)", line).group(1))
                logging.debug(f"Parsed energy consumption: {energy_consumed}")
            except (ValueError, AttributeError) as e:
                logging.error(f"Error parsing energy consumption: {e}")
            continue
        if line.startswith('Plots saved to directory'):
            # Skip this line
            continue
        if line.startswith('Node') and 'Total_CPU' in line:
            parsing_node_utilization = True
            continue  # Skip header line
        if parsing_node_utilization:
            # Parse node-level resource utilization
            match = re.match(r'Node\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line.strip())
            if match:
                try:
                    node_utilization_data.append({
                        'Node': int(match.group(1)),
                        'Total_CPU': float(match.group(2)),
                        'Total_Memory': float(match.group(3)),
                        'Total_Disk': float(match.group(4)),
                        'Total_Network': float(match.group(5)),
                        'Energy_Consumed': energy_consumed
                    })
                except ValueError as e:
                    logging.error(f"Error converting values on line: {line} - {e}")
            else:
                # Reached end of node utilization section
                parsing_node_utilization = False
        elif line.startswith('Node') and 'Pod' in line and 'CPU' in line:
            parsing_started = True
            continue  # Skip header line
        elif parsing_started:
            # Existing code to parse pod-level data
            match = re.match(r'Node\s+(\d+)\s+Pod\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line.strip())
            if match:
                try:
                    parsed_data.append({
                        'Node': int(match.group(1)),
                        'Pod': int(match.group(2)),
                        'CPU': float(match.group(3)),
                        'Memory': float(match.group(4)),
                        'Disk': float(match.group(5)),
                        'Network': float(match.group(6)),
                        'Node_Energy_Consumed': float(match.group(7)),
                        'Energy_Consumed': energy_consumed  # Total energy consumption
                    })
                except ValueError as e:
                    logging.error(f"Error converting values on line: {line} - {e}")
            else:
                # Reached end of pod data
                parsing_started = False
        else:
            # Not parsing any section
            continue

    return parsed_data, energy_consumed, node_utilization_data

# Function to combine the energy consumption plots for all schedulers
def combine_energy_plots(df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get unique seeds and schedulers
    unique_seeds = df['Seed'].unique()

    for seed in unique_seeds:
        seed_df = df[df['Seed'] == seed]  # Filter by seed
        plt.figure(figsize=(10, 6))
        
        for scheduler in scheduler_files:
            scheduler_df = seed_df[seed_df['Scheduler'] == scheduler]
            if not scheduler_df.empty:
                plt.plot(scheduler_df['Node'], scheduler_df['Energy_Consumed'], label=f'{scheduler}')
        
        plt.xlabel('Node')
        plt.ylabel('Energy Consumed (Watts)')
        plt.title(f'Combined Energy Consumption for Seed {seed}')
        plt.legend()
        
        # Save plot with unique seed
        plot_filename = f'combined_energy_plot_seed_{seed}.png'
        plot_filepath = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_filepath)
        plt.close()

        logging.info(f"Saved combined energy plot for seed {seed} to {plot_filepath}")

if __name__ == "__main__":
    # Remove existing result files if any
    if os.path.exists(RESULT_FILE):
        os.remove(RESULT_FILE)
    if os.path.exists(STATS_FILE):
        os.remove(STATS_FILE)

    # Run the experiments
    all_run_results = []

    for exp_num in range(NUM_EXPERIMENTS):
        seed = unique_seeds[exp_num]
        logging.info(f"Running experiment {exp_num + 1} with seed {seed}...")

        for script in scheduler_files:
            output = run_scheduler(script, seed)
            if output:
                parsed_data, energy_consumed, node_utilization_data = parse_output(output)
                if parsed_data:
                    # Add 'Seed' and 'Scheduler' to each entry in parsed_data
                    for entry in parsed_data:
                        entry['Seed'] = seed
                        entry['Scheduler'] = script
                    # Save the parsed results
                    df = pd.DataFrame(parsed_data)
                    write_header = not os.path.exists(RESULT_FILE)
                    df.to_csv(RESULT_FILE, mode='a', header=write_header, index=False)
                    logging.info(f"Saved results for {script} with seed {seed}.")

    # After all experiments, combine the plots
    df = pd.read_csv(RESULT_FILE)
    combine_energy_plots(df, PLOT_OUTPUT_DIR)
    logging.info(f"All experiments completed. Combined plots saved to {PLOT_OUTPUT_DIR}.")

    # Compute summary statistics
    per_run_stats = []

    for (scheduler, seed), group in df.groupby(['Scheduler', 'Seed']):
        nodes_used = group['Node'].nunique()
        total_energy_consumed = group['Energy_Consumed'].iloc[0]  # Should be same for all rows in group
        per_node_usage = group.groupby('Node').agg({
            'CPU': 'sum',
            'Memory': 'sum',
            'Disk': 'sum',
            'Network': 'sum'
        })
        avg_cpu_per_node = per_node_usage['CPU'].mean()
        avg_memory_per_node = per_node_usage['Memory'].mean()
        avg_disk_per_node = per_node_usage['Disk'].mean()
        avg_network_per_node = per_node_usage['Network'].mean()
        
        per_run_stats.append({
            'Scheduler': scheduler,
            'Seed': seed,
            'Nodes_Used': nodes_used,
            'Avg_CPU_Per_Node': avg_cpu_per_node,
            'Avg_Memory_Per_Node': avg_memory_per_node,
            'Avg_Disk_Per_Node': avg_disk_per_node,
            'Avg_Network_Per_Node': avg_network_per_node,
            'Total_Energy_Consumed': total_energy_consumed
        })

    # Convert per_run_stats to DataFrame
    per_run_df = pd.DataFrame(per_run_stats)

    # Now, compute summary statistics per Scheduler
    summary_stats = per_run_df.groupby('Scheduler').agg({
        'Nodes_Used': ['mean', 'std'],
        'Avg_CPU_Per_Node': ['mean', 'std'],
        'Avg_Memory_Per_Node': ['mean', 'std'],
        'Avg_Disk_Per_Node': ['mean', 'std'],
        'Avg_Network_Per_Node': ['mean', 'std'],
        'Total_Energy_Consumed': ['mean', 'std']
    })

    # Flatten the MultiIndex columns
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]

    # Reset index to make 'Scheduler' a column
    summary_stats = summary_stats.reset_index()

    # Save to CSV
    summary_stats.to_csv(STATS_FILE, index=False)

    logging.info(f"Summary statistics saved to {STATS_FILE}.")

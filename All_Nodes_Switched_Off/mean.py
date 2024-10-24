import os
import pandas as pd

# Constants
SEEDS = range(0, 30)  # Define the seed range
SCHEDULERS = ['1D', '4D', 'Defult']  # Define the schedulers
TIME_STEPS = 200  # Number of iterations
LATENCY_LOGS_DIR = "latency_logs"
CPU_USAGE_LOGS_DIR = "cpu_usage_logs"
ENERGY_LOGS_DIR = "energy_logs"
AVERAGED_LOGS_DIR = "averaged_logs"  # Directory to store averaged logs

# Ensure the output directory exists
os.makedirs(AVERAGED_LOGS_DIR, exist_ok=True)

def aggregate_and_average_logs(log_type, logs_dir, output_filename, is_energy=False):
    """
    Aggregates the data from all seeds for each scheduler and computes the average.
    Saves the averaged data to a CSV file.
    
    Args:
    - log_type: The type of logs to process (e.g., 'latency', 'cpu_usage', 'energy').
    - logs_dir: Directory containing the log files.
    - output_filename: Name of the output CSV file.
    - is_energy: Flag to indicate if this is processing energy logs (since file structure is different).
    """
    try:
        # Dictionary to store data for each scheduler
        scheduler_data = {scheduler: [] for scheduler in SCHEDULERS}

        # Iterate over all schedulers and aggregate the data for each seed
        for scheduler in SCHEDULERS:
            for seed in SEEDS:
                if is_energy:
                    # Correct path for energy logs based on the scheduler
                    seed_file = os.path.join(logs_dir, scheduler, f"energy_{scheduler}_seed{seed}.csv")
                    iteration_column = "Time_Step"  # Correct iteration column name for energy logs
                    value_column = "Energy_Consumed"  # Correct value column name for energy logs
                else:
                    # Correct path for latency and CPU usage logs
                    seed_file = os.path.join(logs_dir, f"seed_{seed}", f"{log_type}_{seed}.csv")
                    iteration_column = "Iteration"
                    value_column = "Value"

                if not os.path.exists(seed_file):
                    print(f"Warning: {seed_file} does not exist. Skipping seed {seed} for scheduler {scheduler}.")
                    continue
                
                # Read the data
                seed_df = pd.read_csv(seed_file)
                
                # Check if the required columns exist
                if not all(col in seed_df.columns for col in [iteration_column, value_column]):
                    # Debug print to identify what columns are present in the file
                    print(f"Warning: Missing required columns in {seed_file}. Available columns: {list(seed_df.columns)}. Skipping.")
                    continue
                
                # Collect the values for averaging
                scheduler_data[scheduler].append(seed_df[value_column].values[:TIME_STEPS])

        # Compute the average for each scheduler and save to CSV
        for scheduler, data in scheduler_data.items():
            if not data:
                print(f"No data found for scheduler {scheduler}. Skipping.")
                continue
            
            # Compute the average across seeds (axis=0 for averaging across rows)
            averaged_values = pd.DataFrame(data).mean(axis=0)

            # Create output dataframe
            output_df = pd.DataFrame({
                "Iteration": range(1, TIME_STEPS + 1),
                "Average_Value": averaged_values
            })

            # Save the output file
            output_file = os.path.join(AVERAGED_LOGS_DIR, f"averaged_{log_type}_{scheduler}.csv")
            output_df.to_csv(output_file, index=False)
            print(f"Saved averaged {log_type} data for scheduler {scheduler} to {output_file}")

    except Exception as e:
        print(f"Error processing {log_type} logs: {e}")

# Aggregate and average latency logs
aggregate_and_average_logs("latency", LATENCY_LOGS_DIR, "averaged_latency.csv")

# Aggregate and average CPU usage logs
aggregate_and_average_logs("cpu_usage", CPU_USAGE_LOGS_DIR, "averaged_cpu_usage.csv")

# Aggregate and average energy logs (using correct energy log file paths)
aggregate_and_average_logs("energy", ENERGY_LOGS_DIR, "averaged_energy.csv", is_energy=True)

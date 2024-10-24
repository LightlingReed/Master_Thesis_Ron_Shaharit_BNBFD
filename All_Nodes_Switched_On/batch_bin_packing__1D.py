# batch_bin_packing__1D NEXT BEST FIT DEACREASING.py

import numpy as np
import csv
import os
import argparse
import pandas as pd
from collections import defaultdict  # Import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for headless environments
import matplotlib.pyplot as plt

# Constants
MAX_NODES = 200  # Maximum number of nodes available
ACTIVE_NODES = 200  # Number of initially active nodes (start with zero)
TIME_STEPS = 200
LIFESPAN_SECONDS = 10
CAPACITY = np.array([80, 80, 80, 80])  # Capacities for CPU, Memory, Disk, Network
LIFESPAN_ITERATIONS = 5  # Pods live for 5 iterations
FIXED_CPU_CORES = 4  # Fixed number of CPU cores per node

def save_to_csv(data, scheduler, filename="plot_data.csv"):
    """
    Save the data to a CSV file.
    data: A list of tuples containing (value, iteration_number, seed, scheduler).
    scheduler: The name of the scheduler used (e.g., '1D', '4D', 'Default').
    filename: The name of the CSV file.
    """
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write the header if file doesn't exist
            writer.writerow(['Value', 'Iteration', 'Seed', 'Scheduler'])
        for row in data:
            writer.writerow(row + (scheduler,))

def generate_random_item(num_resources):
    # Generate items up to 50% of capacity
    return np.random.rand(num_resources) * (CAPACITY * 0.5)

def generate_poisson_items(lam, num_resources):
    num_items = np.random.poisson(lam)
    items = [generate_random_item(num_resources) for _ in range(num_items)]
    return items

def lyapunov(virtual_queues, capacity):
    # If there are no items in the virtual queue, Lyapunov value should be 0
    if np.all(virtual_queues == 0):
        return 0.0

    normalized_queues = virtual_queues / capacity
    lyapunov_value = np.sum(normalized_queues ** 2)
    max_lyapunov_value = np.sum(np.ones_like(capacity))

    # Ensure Lyapunov value is zero by the last iteration
    if lyapunov_value > 0 and np.sum(virtual_queues) == 0:
        return 0.0

    return lyapunov_value / max_lyapunov_value

def one_dimensional_scheduler_best_fit_decreasing(item, nodes, capacity, last_assigned_node):
    """
    1D Best Fit Decreasing (BNFD) Scheduler: Assigns pods based on a single prioritized dimension (CPU).
    Chooses the node that leaves the least leftover capacity in the prioritized dimension after assignment.
    Ignores the last_assigned_node parameter as it's not used in 1D scheduling.
    
    Args:
        item (np.array): Resource requirements of the pod.
        nodes (list): List of current nodes with their pods.
        capacity (np.array): Capacity of each resource dimension.
        last_assigned_node (int): Index of the last node to which a pod was assigned (ignored).
    
    Returns:
        int or None: Index of the node to assign the pod or None if assignment fails.
        int: Updated index of the last node assigned.
    """
    prioritized_dimension = 0  # Fixed to CPU (index 0)
    
    best_node = None
    min_leftover = None
    for i, node in enumerate(nodes):
        # Calculate current load for the node
        node_load = np.sum([pod['item'] for pod in node], axis=0) if node else np.zeros_like(capacity)
        # Check if the node can accommodate the pod in all dimensions
        if np.all(node_load + item <= capacity):
            # Calculate leftover in the prioritized dimension
            leftover = capacity[prioritized_dimension] - (node_load[prioritized_dimension] + item[prioritized_dimension])
            if leftover >= 0:
                if (min_leftover is None) or (leftover < min_leftover):
                    best_node = i
                    min_leftover = leftover

    if best_node is not None:
        print(f"1D BNFD Scheduler: Assigning pod to Node {best_node + 1}")
        return best_node, best_node  # Assign to this node and update last_assigned_node

    # If no existing node can accommodate, activate a new node if possible
    if len(nodes) < MAX_NODES:
        new_node_index = len(nodes)
        nodes.append([])  # Initialize the new node
        print(f"1D BNFD Scheduler: Activating new Node {new_node_index + 1}")
        return new_node_index, new_node_index  # Assign to the new node and update last_assigned_node
    else:
        # Maximum nodes reached; cannot assign the pod
        print("1D BNFD Scheduler: Maximum nodes reached. Cannot assign the pod.")
        return None, -1  # Return None and keep last_assigned_node unchanged

def assign_items_to_nodes(capacity, num_active_nodes, scheduler_func):
    nodes = [[] for _ in range(num_active_nodes)]  # Start with active nodes
    virtual_queues = np.zeros((MAX_NODES, 4))
    pending_items = []
    usage_history = defaultdict(list)  # Use defaultdict to handle dynamic keys
    energy_history = []
    latency_history = []
    unassigned_pods_history = []
    lam = 2  # Poisson parameter
    last_assigned_node = -1  # To track the last node assigned

    # Check if 'energy_consumption_log.csv' exists
    csv_file_path = 'energy_consumption_log.csv'
    if not os.path.isfile(csv_file_path):
        print(f"Error: '{csv_file_path}' not found. Please ensure the file exists.")
        return nodes, usage_history, energy_history, latency_history, unassigned_pods_history, []

    try:
        energy_df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading '{csv_file_path}': {e}")
        return nodes, usage_history, energy_history, latency_history, unassigned_pods_history, []

    for iteration in range(1, TIME_STEPS + 1):
        current_time = iteration
        new_items = generate_poisson_items(lam, 4)
        for new_item in new_items:
            pending_items.append({'item': new_item, 'timestamp': current_time})

        unassigned_pods = 0

        # Remove pods that have exceeded their lifespan
        for node in nodes:
            node[:] = [pod for pod in node if current_time - pod['timestamp'] < LIFESPAN_ITERATIONS]

        # Assign pods in batches every 10 iterations
        if iteration % 10 == 0 and pending_items:
            print(f"Iteration {iteration}: Assigning pods...")
            # Sort pending items in decreasing order based on prioritized dimension (CPU)
            items_to_process = sorted(pending_items, key=lambda x: x['item'][0], reverse=True)
            pending_items = []
            for item_dict in items_to_process:
                item = item_dict['item']
                target_node, last_assigned_node = scheduler_func(item, nodes, capacity, last_assigned_node)
                if target_node is not None:
                    nodes[target_node].append({'item': item, 'timestamp': current_time})
                else:
                    pending_items.append(item_dict)
                    unassigned_pods += 1

        unassigned_pods_history.append(unassigned_pods)

        # Record resource usage
        for i, node in enumerate(nodes):
            used_resources = np.sum([pod['item'] for pod in node], axis=0) if node else np.zeros_like(capacity)
            usage_history[i].append(used_resources)

        # Calculate energy usage
        total_energy, node_energy_list = calculate_energy_usage(nodes, energy_df)
        energy_history.append(total_energy)

        # Calculate latency
        latency = len(pending_items)
        latency_history.append(latency)

    return nodes, usage_history, energy_history, latency_history, unassigned_pods_history, node_energy_list


# Assuming power_df contains the power (in Watts) data
def calculate_energy_usage(nodes, power_df, time_step_duration=1):
    total_energy = 0
    node_energy_list = []

    # Use 'Compute1_Watts' for energy calculations, or adjust based on available power columns
    if 'Compute1_Watts' not in power_df.columns:
        raise ValueError("CSV file must contain 'Compute1_Watts' for power consumption.")
    
    power_watts = power_df['Compute1_Watts'].to_numpy()  # Use 'Compute1_Watts'
    power_data = power_df[['CPU_Cores', 'CPU_Load', 'Memory_Usage', 'Disk_IO', 'Network_Bandwidth']].to_numpy()

    for node in nodes:
        if len(node) > 0:
            node_usage = np.sum([pod['item'] for pod in node], axis=0)
            query = np.array([FIXED_CPU_CORES, node_usage[0], node_usage[1], node_usage[2], node_usage[3]])  # Correct mapping
            distances = np.linalg.norm(power_data - query, axis=1)
            closest_idx = np.argmin(distances)
            node_power = power_watts[closest_idx]
            node_energy = node_power * time_step_duration  # Convert power to energy for this timestep
            total_energy += node_energy
            node_energy_list.append(node_energy)
        else:
            # If the node is idle, assign a default idle energy consumption
            idle_power = power_watts.min()  # Use the minimum power as idle power
            node_energy = idle_power * time_step_duration
            total_energy += node_energy
            node_energy_list.append(node_energy)

    return total_energy, node_energy_list

def plot_usage_diagram(usage_history, seed, scheduler, save_path=None):
    max_length = max(len(usage_list) for usage_list in usage_history.values()) if usage_history else 0
    if max_length == 0:
        print("No usage data to plot.")
        return

    aggregated_usage = np.zeros((max_length, 4))

    for usage_list in usage_history.values():
        for i, usage_data in enumerate(usage_list):
            aggregated_usage[i] += usage_data

    plt.figure(figsize=(12, 6))
    plt.plot(aggregated_usage[:, 0], label='CPU Usage')
    plt.plot(aggregated_usage[:, 1], label='Memory Usage')
    plt.plot(aggregated_usage[:, 2], label='Disk Usage')
    plt.plot(aggregated_usage[:, 3], label='Network Usage')
    plt.title('Aggregated Resource Usage Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Resource Usage')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Usage diagram saved to {save_path}")
    else:
        plt.show()
    plt.close()
    
    # Save to CSV
    usage_data_to_save = [(aggregated_usage[i, 0], i + 1, seed) for i in range(max_length)]
    save_to_csv(usage_data_to_save, scheduler, filename="cpu_usage_data.csv")

def plot_energy_diagram(energy_history, num_active_nodes, csv_file_path, seed, scheduler, save_path=None, show=False):
    """
    Plot the energy consumption over time and save the energy log.

    Args:
        energy_history (list): List of total energy consumed at each time step.
        num_active_nodes (int): Number of active nodes.
        csv_file_path (str): Path to the energy consumption CSV file.
        seed (int): Random seed used for the simulation.
        scheduler (str): Name of the scheduler (e.g., '1D').
        save_path (str, optional): Path to save the energy plot. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to False.
    """
    try:
        # Load the CSV data to find the idle power consumption
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading '{csv_file_path}': {e}")
        return

    if df.empty:
        print(f"'{csv_file_path}' is empty. Cannot plot energy diagram.")
        return

    try:
        # Extract the first line's resource usage values to calculate idle power
        first_line = df.iloc[0]
        cpu_load = first_line['CPU_Load']
        memory_usage = first_line['Memory_Usage']
        disk_io = first_line['Disk_IO']
        network_bandwidth = first_line['Network_Bandwidth']

        # Calculate the Euclidean distance using the first line of the CSV
        query = np.array([FIXED_CPU_CORES, cpu_load, memory_usage, disk_io, network_bandwidth])

        # Directly use 'Compute1_Watts' for idle power consumption
        idle_energy_per_node = first_line['Compute1_Watts']

        # Total idle energy for all nodes
        total_idle_energy = idle_energy_per_node * num_active_nodes

        # Adjust the energy history to include the idle energy at each time step
        adjusted_energy_history = [total_idle_energy + step_energy for step_energy in energy_history]

        # Plot combined energy usage over time as a connected time series plot
        time_steps = range(1, len(adjusted_energy_history) + 1)
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, adjusted_energy_history, color='purple', label='Total Energy Usage', marker='o')

        plt.xlabel('Time Steps')
        plt.ylabel('Energy Usage (W)')
        plt.title(f'{scheduler} Scheduler: Combined Energy Usage Over Time')
        plt.legend(loc='upper right')
        plt.grid(True)

        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            print(f"Energy diagram saved to {save_path}")
        if show:
            plt.show()
        plt.close()

        # **Save energy history to energy_logs/{scheduler}/energy_{scheduler}_seed{seed}.csv**
        energy_log_dir = os.path.join('energy_logs', scheduler)
        os.makedirs(energy_log_dir, exist_ok=True)
        energy_log_path = os.path.join(energy_log_dir, f'energy_{scheduler}_seed{seed}.csv')

        energy_df = pd.DataFrame({
            'Time_Step': range(1, len(adjusted_energy_history) + 1),
            'Energy_Consumed': adjusted_energy_history
        })
        energy_df.to_csv(energy_log_path, index=False)
        print(f"Energy log saved to {energy_log_path}")

    except Exception as e:
        print(f"Error plotting energy diagram: {e}")

def plot_combined_energy_diagram(energy_history, scheduler, seed, save_path=None, show=False):
    """
    Plot the combined energy consumption over time and save the energy log.

    Args:
        energy_history (list): List of total energy consumed at each time step.
        scheduler (str): Name of the scheduler (e.g., '1D').
        seed (int): Random seed used for the simulation.
        save_path (str, optional): Path to save the energy plot. Defaults to None.
        show (bool, optional): Whether to display the plot. Defaults to False.
    """
    if not energy_history:
        print("No energy history data to plot.")
        return

    time_steps = range(1, len(energy_history) + 1)
    adjusted_energy_history = [np.sum(step_energy) for step_energy in energy_history]

    plt.figure(figsize=(40, 6))
    plt.plot(time_steps, adjusted_energy_history, color='purple', label='Total Energy Usage', marker='o')

    plt.xlabel('Time Steps')
    plt.ylabel('Energy Usage (W)')
    plt.title('Combined Energy Usage Over Time')
    plt.ylim(0, max(adjusted_energy_history) * 1.1)  # Ensure the full Y-axis starts from 0
    plt.legend(loc='upper right')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Combined energy diagram saved to {save_path}")
    if show:
        plt.show(block=False)
    plt.close()

    # Save to CSV
    energy_data_to_save = [(adjusted_energy_history[i], i + 1, seed) for i in range(len(adjusted_energy_history))]
    save_to_csv(energy_data_to_save, scheduler, filename="energy_data.csv")

def plot_latency_diagram(latency_history, seed, scheduler, save_path=None):
    if not latency_history:
        print("No latency data to plot.")
        return

    time_steps = range(1, len(latency_history) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, latency_history, color='blue', label='Latency', marker='o')
    plt.xlabel('Time Steps')
    plt.ylabel('Latency (Number of Pending Items)')
    plt.title('Latency Over Time')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Latency diagram saved to {save_path}")
    else:
        plt.show()
    plt.close()
    
    # Save to CSV
    latency_data_to_save = [(latency_history[i], i + 1, seed) for i in range(len(latency_history))]
    save_to_csv(latency_data_to_save, scheduler, filename="latency_data.csv")

def plot_unassigned_pods(unassigned_pods_history, seed, scheduler, save_path=None):
    if not unassigned_pods_history:
        print("No unassigned pods data to plot.")
        return

    time_steps = range(1, len(unassigned_pods_history) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, unassigned_pods_history, color='red', label='Unassigned Pods', marker='o')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Unassigned Pods')
    plt.title('Unassigned Pods Over Time')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Unassigned pods diagram saved to {save_path}")
    else:
        plt.show()
    plt.close()
    
    # Save to CSV
    unassigned_pods_to_save = [(unassigned_pods_history[i], i + 1, seed) for i in range(len(unassigned_pods_history))]
    save_to_csv(unassigned_pods_to_save, scheduler, filename="unassigned_pods_data.csv")

def save_plots_to_directory(energy_history, latency_history, usage_history, unassigned_pods_history, num_active_nodes, seed, csv_file_path, scheduler, directory="plots_1D"):
    """
    Save all relevant plots and CSV data to the specified directory.

    Args:
        energy_history (list): List of total energy consumed at each time step.
        latency_history (list): List of latency values at each time step.
        usage_history (defaultdict): Resource usage history per node.
        unassigned_pods_history (list): List of unassigned pod counts at each time step.
        num_active_nodes (int): Number of active nodes.
        seed (int): Random seed used for the simulation.
        csv_file_path (str): Path to the energy consumption CSV file.
        scheduler (str): Name of the scheduler (e.g., '1D').
        directory (str, optional): Directory to save the plots. Defaults to "plots_1D".
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

    plot_filenames = {
        "energy_plot": "energy_plot",
        "combined_energy_plot": "combined_energy_plot",
        "latency_plot": "latency_plot",
        "usage_plot": "usage_plot",
        "unassigned_pods_plot": "unassigned_pods_plot"
    }

    # Function to generate a unique filename by appending a number if necessary
    def get_unique_filename(base_name, directory, extension=".png"):
        counter = 0
        while True:
            filename = f"{base_name}{counter}{extension}"
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                return filepath
            counter += 1

    # Define file paths
    energy_plot_path = get_unique_filename(plot_filenames["energy_plot"], directory)
    combined_energy_plot_path = get_unique_filename(plot_filenames["combined_energy_plot"], directory)
    latency_plot_path = get_unique_filename(plot_filenames["latency_plot"], directory)
    usage_plot_path = get_unique_filename(plot_filenames["usage_plot"], directory)
    unassigned_pods_plot_path = get_unique_filename(plot_filenames["unassigned_pods_plot"], directory)

    # Save the plots and save data to CSV
    plot_energy_diagram(
        energy_history=energy_history,
        num_active_nodes=num_active_nodes,
        csv_file_path=csv_file_path,
        seed=seed,
        scheduler=scheduler,
        save_path=energy_plot_path
    )
    plot_combined_energy_diagram(
        energy_history=energy_history,
        scheduler=scheduler,
        seed=seed,
        save_path=combined_energy_plot_path
    )
    plot_latency_diagram(
        latency_history=latency_history,
        seed=seed,
        scheduler=scheduler,
        save_path=latency_plot_path
    )
    plot_usage_diagram(
        usage_history=usage_history,
        seed=seed,
        scheduler=scheduler,
        save_path=usage_plot_path
    )
    plot_unassigned_pods(
        unassigned_pods_history=unassigned_pods_history,
        seed=seed,
        scheduler=scheduler,
        save_path=unassigned_pods_plot_path
    )

    print(f"Plots saved to directory '{directory}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    np.random.seed(args.seed)

    num_active_nodes = ACTIVE_NODES  # Should be 0 for this script

    # Call assign_items_to_nodes with the 1D BNFD scheduler function
    nodes, usage_history, energy_history, latency_history, unassigned_pods_history, node_energy_list = assign_items_to_nodes(
        CAPACITY, num_active_nodes, one_dimensional_scheduler_best_fit_decreasing)

    # Calculate total energy consumption
    total_energy = sum(energy_history)
    print(f"\nTotal energy consumed: {total_energy} units")

    # Define csv_file_path for plotting functions
    csv_file_path = 'energy_consumption_log.csv'

    # Save plots and data
    save_plots_to_directory(
        energy_history=energy_history,
        latency_history=latency_history,
        usage_history=usage_history,
        unassigned_pods_history=unassigned_pods_history,
        num_active_nodes=num_active_nodes,
        seed=args.seed,
        csv_file_path=csv_file_path,
        scheduler="1D",
        directory="plots_1D"
    )

    # Print the results in the expected format
    print('Node\tPod\tCPU\tMemory\tDisk\tNetwork\tNode_Energy_Consumed\tScheduler')
    for i, node in enumerate(nodes):
        node_energy_consumed = node_energy_list[i]
        for j, pod in enumerate(node):
            print(f'Node {i + 1}\tPod {j + 1}\t{pod["item"][0]:.2f}\t'
                  f'{pod["item"][1]:.2f}\t{pod["item"][2]:.2f}\t'
                  f'{pod["item"][3]:.2f}\t{node_energy_consumed:.2f}\t1D')

    # Calculate and print per-node resource utilization
    print('Node\tTotal_CPU\tTotal_Memory\tTotal_Disk\tTotal_Network\tScheduler')
    for i, node in enumerate(nodes):
        node_resources = np.sum([pod['item'] for pod in node], axis=0) if node else np.zeros_like(CAPACITY)
        print(f'Node {i + 1}\t{FIXED_CPU_CORES}\t{node_resources[0]:.2f}\t{node_resources[1]:.2f}\t{node_resources[2]:.2f}\t{node_resources[3]:.2f}\t1D')

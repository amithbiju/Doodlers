import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import heapq
from haversine import haversine

# Load and filter dataset
df = pd.read_csv('modified_contrail_data.csv')
df_filtered = df[(df['flight_level'] >= 270) & (df['flight_level'] <= 280)]

# Ensure start and end points are in the dataset
start = (9.710497856140137, 75.509033203125, 270)
end = (10.960497856140137, 77.509033203125, 280)  # Adjusted flight level to be within range

# Add start and end points to the dataset if they don't exist
df_filtered = pd.concat([df_filtered, pd.DataFrame([{'longitude': start[0], 'latitude': start[1], 'flight_level': start[2]},
                                                   {'longitude': end[0], 'latitude': end[1], 'flight_level': end[2]}])],
                       ignore_index=True)

# Define risk threshold
threshold = df_filtered['expected_effective_energy_forcing'].quantile(0.8)  # Lowered to 80th percentile
high_risk_zones = df_filtered[df_filtered['expected_effective_energy_forcing'] > threshold]

# Check if high-risk zones are empty
if high_risk_zones.empty:
    print("No high-risk zones found. Proceeding without risk penalty.")
    high_risk_tree = None
else:
    # Create KDTree for high-risk zones
    high_risk_tree = KDTree(high_risk_zones[['longitude', 'latitude', 'flight_level']])

# Create KDTree for the dataset
tree = KDTree(df_filtered[['longitude', 'latitude', 'flight_level']])

# Fuel consumption dictionary
fuel_consumption = {270: 10.0, 280: 9.5, 290: 9.0, 300: 8.5, 310: 8.0, 320: 7.5,
                    330: 7.0, 340: 6.5, 350: 6.0, 360: 5.5, 370: 5.0, 380: 4.5,
                    390: 4.0, 400: 3.5}

# Default fuel consumption for undefined flight levels
default_fuel_consumption = min(fuel_consumption.values())

# Function to calculate risk penalty
def calculate_risk_penalty(neighbor_node, high_risk_tree, radius=10.0):
    if high_risk_tree is None:
        return 0  # No risk penalty if no high-risk zones
    # Query high-risk zones within the radius
    risk_indices = high_risk_tree.query_radius([neighbor_node], r=radius)[0]
    return len(risk_indices) * 5  # Penalty multiplier

# Function to get fuel consumption (with interpolation for undefined flight levels)
def get_fuel_consumption(altitude, fuel_consumption):
    if altitude in fuel_consumption:
        return fuel_consumption[altitude]
    # Find nearest defined flight levels
    defined_levels = sorted(fuel_consumption.keys())
    lower = max([level for level in defined_levels if level < altitude], default=min(defined_levels))
    upper = min([level for level in defined_levels if level > altitude], default=max(defined_levels))
    # Linear interpolation
    return fuel_consumption[lower] + (fuel_consumption[upper] - fuel_consumption[lower]) * (altitude - lower) / (upper - lower)

# A* algorithm with fuel consumption and risk penalty
def find_optimal_path(start, end, tree, high_risk_tree, fuel_consumption, default_fuel_consumption):
    open_queue = [(0, start, [])]
    heapq.heapify(open_queue)
    visited = set()
    
    while open_queue:
        total_cost, current_node, path = heapq.heappop(open_queue)
        
        print(f"Exploring Node: {current_node}, Total Cost: {total_cost}")
        
        if current_node == end:
            return path + [current_node]
        
        if current_node in visited:
            continue
        visited.add(current_node)
        
        # Query neighbors within a dynamic radius
        neighbors = tree.query_radius([current_node], r=50.0)[0]  # Increase radius to 50.0
        print(f"Found {len(neighbors)} neighbors for Node: {current_node}")
        
        for neighbor_idx in neighbors:
            neighbor_node = tuple(tree.data[neighbor_idx])
            
            # Calculate risk penalty
            risk_penalty = calculate_risk_penalty(neighbor_node, high_risk_tree)
            
            # Calculate distance and fuel consumption
            distance = haversine(current_node[:2], neighbor_node[:2])
            neighbor_altitude = neighbor_node[2]
            fuel_used = distance * get_fuel_consumption(neighbor_altitude, fuel_consumption)
            
            # Calculate heuristic (Haversine distance to end)
            heuristic = haversine(neighbor_node[:2], end[:2])
            
            # Total cost = cost so far + fuel used + heuristic + risk penalty
            total_cost_neighbor = total_cost + fuel_used + heuristic + risk_penalty
            
            heapq.heappush(open_queue, (total_cost_neighbor, neighbor_node, path + [current_node]))
    
    return None

# Find nearest points to start and end
start_nearest_idx = tree.query([start], k=1)[1][0][0]
end_nearest_idx = tree.query([end], k=1)[1][0][0]

start_nearest = tuple(tree.data[start_nearest_idx])
end_nearest = tuple(tree.data[end_nearest_idx])

print("Nearest Start Point:", start_nearest)
print("Nearest End Point:", end_nearest)

# Find optimal path
optimal_path = find_optimal_path(start_nearest, end_nearest, tree, high_risk_tree, fuel_consumption, default_fuel_consumption)

if optimal_path:
    print("Optimal Path:", optimal_path)
    
    # Visualization
    # Prepare high-risk zones DataFrame for plotting
    if not high_risk_zones.empty:
        high_risk_zones_df = high_risk_zones[['longitude', 'latitude']]
    
    # Plot the path
    plt.figure(figsize=(10, 6))
    
    # Plot high-risk zones (if any)
    if not high_risk_zones.empty:
        plt.scatter(high_risk_zones_df['longitude'], high_risk_zones_df['latitude'], c='red', label='High-Risk Zones', alpha=0.5)
    
    # Plot optimal path
    plt.plot([x[0] for x in optimal_path], [x[1] for x in optimal_path], marker='o', linestyle='-', color='blue', label='Optimal Path')
    
    # Plot start and end points
    plt.scatter(start[0], start[1], c='green', marker='*', s=200, label='Start')
    plt.scatter(end[0], end[1], c='purple', marker='*', s=200, label='End')
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Flight Path Optimization")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot altitude profile
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(optimal_path)), [x[2] for x in optimal_path], marker='o', linestyle='-', color='b')
    plt.xlabel("Step in Path")
    plt.ylabel("Flight Level (FL)")
    plt.title("Altitude Profile of Optimal Flight Path")
    plt.grid()
    plt.show()
else:
    print("No optimal path found.")
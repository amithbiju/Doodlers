import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import heapq
from haversine import haversine

# Load and filter dataset modified_
df = pd.read_csv('modified_contrail_data.csv')
df_filtered = df[(df['flight_level'] >= 270) & (df['flight_level'] <= 400)]

# Ensure start and end points are in the dataset
start = (9.710497856140137,75.509033203125,270)
end = (10.960497856140137,77.509033203125,280)
       
df_filtered = pd.concat([df_filtered, pd.DataFrame([{'longitude': start[0], 'latitude': start[1], 'flight_level': start[2]},
                                                     {'longitude': end[0], 'latitude': end[1], 'flight_level': end[2]}])],
                        ignore_index=True)

# Define risk threshold
threshold = df_filtered['expected_effective_energy_forcing'].quantile(0.9)
high_risk_zones = df_filtered[df_filtered['expected_effective_energy_forcing'] > threshold]

# Create KDTree
tree = KDTree(df_filtered[['longitude', 'latitude', 'flight_level']])

# Fuel consumption dictionary
fuel_consumption = {270: 10.0, 280: 9.5, 290: 9.0}

def find_optimal_path(start, end, tree, high_risk_zones):
    open_queue = [(0, start, [])]
    heapq.heapify(open_queue)
    visited = set()
    
    while open_queue:
        total_cost, current_node, path = heapq.heappop(open_queue)
        
        if current_node == end:
            return path + [current_node]
        
        if current_node in visited:
            continue
        visited.add(current_node)
        
        dist, idx = tree.query([current_node], k=20)  # Increase k for broader search
        for i in idx[0]:
            neighbor_node = tuple(tree.data[i])
            
            # Apply penalty instead of outright skipping high-risk zones
            risk_penalty = sum(1 for _, row in high_risk_zones.iterrows() if haversine(neighbor_node[:2], (row['longitude'], row['latitude'])) < 5) * 10
            
            distance = haversine(current_node[:2], neighbor_node[:2])
            neighbor_altitude = neighbor_node[2]
            fuel_used = distance * fuel_consumption.get(neighbor_altitude, min(fuel_consumption.values()))  # Default to lowest fuel value
            heuristic = haversine(neighbor_node[:2], end[:2])
            total_cost_neighbor = total_cost + fuel_used + heuristic + risk_penalty  # Include penalty
            
            heapq.heappush(open_queue, (total_cost_neighbor, neighbor_node, path + [current_node]))
    
    return None

# Find optimal path
optimal_path = find_optimal_path(start, end, tree, high_risk_zones)
print("Optimal Path:", optimal_path)

# Prepare high-risk zones DataFrame for plotting
high_risk_zones_df = high_risk_zones[['longitude', 'latitude']]

# Plot the path
if optimal_path:
    plt.figure(figsize=(10, 6))
    
    # Plot high-risk zones
    plt.scatter(high_risk_zones_df['longitude'], high_risk_zones_df['latitude'], c='red', label='High-Risk Zones')
    
    # Plot optimal path
    plt.plot([x[0] for x in optimal_path], [x[1] for x in optimal_path], marker='o', label='Optimal Path')
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Flight Path Optimization")
    plt.legend()
    plt.show()

# Plot altitude profile
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(optimal_path)), [x[2] for x in optimal_path], marker='o', linestyle='-', color='b')
    plt.xlabel("Step in Path")
    plt.ylabel("Flight Level (FL)")
    plt.title("Altitude Profile of Optimal Flight Path")
    plt.grid()
    plt.show()

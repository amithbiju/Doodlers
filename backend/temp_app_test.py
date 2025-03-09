import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import heapq
from haversine import haversine

# Load and filter dataset
df = pd.read_csv('modified_contrail_data.csv')
df_filtered = df[(df['flight_level'] >= 270) & (df['flight_level'] <= 400)]

# Define start and end points
start = (10.210497856140137, 75.759033203125, 350)
end = (10.960497856140137, 77.509033203125, 430)

# Define risk threshold
threshold = df_filtered['expected_effective_energy_forcing'].quantile(0.9)
high_risk_zones = df_filtered[df_filtered['expected_effective_energy_forcing'] > threshold]

# Create direct path points between start and end
# This ensures we have a valid path even if the dataset doesn't have sufficient connectivity
num_intermediate_points = 20
direct_path = []

for i in range(num_intermediate_points + 1):
    # Linear interpolation between start and end
    t = i / num_intermediate_points
    lon = start[0] + t * (end[0] - start[0])
    lat = start[1] + t * (end[1] - start[1])
    fl = start[2] + t * (end[2] - start[2])
    direct_path.append({'longitude': lon, 'latitude': lat, 'flight_level': fl})

# Add direct path points to the dataset
direct_path_df = pd.DataFrame(direct_path)
enhanced_df = pd.concat([df_filtered, direct_path_df], ignore_index=True)

# Create KDTree with the enhanced dataset
tree_data = enhanced_df[['longitude', 'latitude', 'flight_level']].values
tree = KDTree(tree_data)

# Complete fuel consumption dictionary for all possible flight levels
fuel_consumption = {fl: 10.0 - (fl - 270) * 0.05 for fl in range(270, 431, 10)}
# Ensure we have values for both start and end flight levels
fuel_consumption[350] = 10.0 - (350 - 270) * 0.05
fuel_consumption[430] = 10.0 - (430 - 270) * 0.05

def distance_score(point1, point2):
    """Calculate distance score including both horizontal and vertical components"""
    horizontal_dist = haversine(point1[:2], point2[:2])
    vertical_dist = abs(point1[2] - point2[2]) * 0.1  # Scale altitude difference
    return horizontal_dist + vertical_dist

def find_optimal_path(start, end, tree, high_risk_zones, max_iterations=10000):
    start = tuple(start)
    end = tuple(end)
    
    # Calculate initial heuristic
    initial_h = distance_score(start, end)
    
    # Priority queue: (f_score, iteration_count, current_node, path, total_cost)
    # Using iteration_count as tiebreaker for equal f_scores
    open_queue = [(initial_h, 0, start, [], 0)]
    heapq.heapify(open_queue)
    
    # Keep track of visited nodes and their best costs
    visited = {}  # {node_key: best_g_score}
    
    iterations = 0
    
    while open_queue and iterations < max_iterations:
        iterations += 1
        
        # Pop node with lowest f_score
        f_score, _, current, path, g_score = heapq.heappop(open_queue)
        
        # Round current node coordinates for consistent key generation
        current_key = f"{current[0]:.5f}_{current[1]:.5f}_{current[2]:.0f}"
        
        # Check if we've found the end point (or are very close)
        if distance_score(current, end) < 5:
            full_path = path + [current]
            if current != end:
                full_path.append(end)
            print(f"Path found after {iterations} iterations with {len(full_path)} points")
            return full_path
        
        # Skip if we've visited this node with a better cost already
        if current_key in visited and visited[current_key] <= g_score:
            continue
        
        # Mark as visited with current cost
        visited[current_key] = g_score
        
        # If we're getting close to the end point, add it as a neighbor
        if distance_score(current, end) < 50:
            neighbors = [end]
        else:
            # Find closest points in the tree
            dists, indices = tree.query([current], k=30)
            neighbors = [tuple(tree_data[idx]) for idx in indices[0]]
        
        for neighbor in neighbors:
            # Skip if it's effectively the same as current node
            neighbor_key = f"{neighbor[0]:.5f}_{neighbor[1]:.5f}_{neighbor[2]:.0f}"
            if neighbor_key == current_key:
                continue
            
            # Calculate distance cost
            dist = distance_score(current, neighbor)
            
            # Skip if the distance is too large for a single step
            if dist > 150 and neighbor != end:  # Allow larger steps to the end point
                continue
            
            # Calculate fuel cost
            flight_level = int(round(current[2] / 10) * 10)  # Round to nearest 10
            if flight_level not in fuel_consumption:
                flight_level = min(fuel_consumption.keys(), key=lambda x: abs(x - flight_level))
            
            horizontal_dist = haversine(current[:2], neighbor[:2])
            fuel_cost = horizontal_dist * fuel_consumption[flight_level]
            
            # Calculate altitude change cost
            altitude_change = abs(current[2] - neighbor[2])
            altitude_cost = altitude_change * 0.5
            
            # Calculate risk penalty
            risk_factor = 0
            for _, row in high_risk_zones.iterrows():
                risk_point = (row['longitude'], row['latitude'])
                distance_to_risk = haversine(neighbor[:2], risk_point)
                if distance_to_risk < 20:  # Within 20km
                    # Penalty decreases with distance
                    risk_factor += max(0, 20 - distance_to_risk) * 2
            
            # Update g_score (cost from start to neighbor)
            new_g_score = g_score + fuel_cost + altitude_cost + risk_factor
            
            # Only consider this neighbor if we haven't found a better path to it
            if neighbor_key not in visited or new_g_score < visited[neighbor_key]:
                # Calculate heuristic (estimated cost to end)
                h_score = distance_score(neighbor, end)
                
                # Calculate f_score (total estimated cost)
                new_f_score = new_g_score + h_score
                
                # Add to queue with iteration count as tiebreaker
                heapq.heappush(open_queue, (
                    new_f_score, 
                    iterations, 
                    neighbor, 
                    path + [current], 
                    new_g_score
                ))
    
    print(f"No path found after {iterations} iterations")
    
    # If no path found, return the direct path as fallback
    direct_path_points = [start] + [tuple(point.values()) for point in direct_path[1:-1]] + [end]
    print(f"Returning direct path with {len(direct_path_points)} points as fallback")
    return direct_path_points

# Find optimal path
optimal_path = find_optimal_path(start, end, tree, high_risk_zones)

# Calculate path statistics
if optimal_path:
    # Calculate total distance and fuel consumption
    total_distance = 0
    total_fuel = 0
    high_risk_exposure = 0
    
    for i in range(len(optimal_path) - 1):
        point1 = optimal_path[i]
        point2 = optimal_path[i + 1]
        
        leg_distance = haversine(point1[:2], point2[:2])
        total_distance += leg_distance
        
        fl = int(round(point1[2] / 10) * 10)  # Round to nearest 10
        if fl not in fuel_consumption:
            fl = min(fuel_consumption.keys(), key=lambda x: abs(x - fl))
        
        leg_fuel = leg_distance * fuel_consumption[fl]
        total_fuel += leg_fuel
        
        # Calculate risk exposure
        for _, row in high_risk_zones.iterrows():
            risk_point = (row['longitude'], row['latitude'])
            midpoint = ((point1[0] + point2[0])/2, (point1[1] + point2[1])/2)
            distance_to_risk = haversine(midpoint, risk_point)
            if distance_to_risk < 20:
                high_risk_exposure += 1
    
    print(f"Total distance: {total_distance:.2f} km")
    print(f"Total fuel consumption: {total_fuel:.2f} units")
    print(f"High-risk zone exposures: {high_risk_exposure}")

    # Plot the path
    plt.figure(figsize=(14, 10))
    
    # Plot high-risk zones with a heatmap-like effect
    if not high_risk_zones.empty:
        plt.scatter(high_risk_zones['longitude'], high_risk_zones['latitude'], 
                    c='red', alpha=0.5, s=100, label='High-Risk Zones')
    
    # Create buffer around high-risk zones for visualization
    if not high_risk_zones.empty:
        for _, row in high_risk_zones.iterrows():
            risk_lon = row['longitude']
            risk_lat = row['latitude']
            circle = plt.Circle((risk_lon, risk_lat), 0.1, color='red', fill=False, alpha=0.3)
            plt.gca().add_patch(circle)
    
    # Plot optimal path
    path_lons = [p[0] for p in optimal_path]
    path_lats = [p[1] for p in optimal_path]
    plt.plot(path_lons, path_lats, 'b-', linewidth=2.5, label='Optimized Path')
    plt.scatter(path_lons, path_lats, c='blue', s=40, zorder=5)
    
    # Highlight start and end
    plt.scatter([start[0], end[0]], [start[1], end[1]], 
                c=['green', 'purple'], s=150, zorder=10,
                label='Start/End Points')
    plt.annotate('START', (start[0], start[1]), xytext=(10, 10), 
                 textcoords='offset points', fontsize=12)
    plt.annotate('END', (end[0], end[1]), xytext=(10, 10), 
                 textcoords='offset points', fontsize=12)
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Flight Path Optimization with Contrail Avoidance")
    plt.legend(loc='best')
    plt.grid(True)
    
    # Plot altitude profile
    plt.figure(figsize=(14, 5))
    distances = [0]
    current_distance = 0
    
    for i in range(1, len(optimal_path)):
        current_distance += haversine(optimal_path[i-1][:2], optimal_path[i][:2])
        distances.append(current_distance)
    
    plt.plot(distances, [p[2] for p in optimal_path], 'b-o', linewidth=2)
    plt.fill_between(distances, [p[2] for p in optimal_path], alpha=0.2)
    
    # Add flight level grid lines
    for fl in range(300, 450, 10):
        plt.axhline(y=fl, color='gray', linestyle='--', alpha=0.3)
    
    plt.xlabel("Distance (km)")
    plt.ylabel("Flight Level")
    plt.title("Altitude Profile of Optimized Flight Path")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import heapq
from haversine import haversine
import random

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

# Generate fake weather data
def generate_fake_weather():
    # Define the area bounds (slightly larger than our flight path area)
    lon_min, lon_max = 10.0, 11.2
    lat_min, lat_max = 75.5, 77.8
    fl_min, fl_max = 270, 430
    
    # Generate different types of weather events
    weather_data = []
    
    # Thunderstorms (severe, localized)
    num_thunderstorms = random.randint(2, 4)
    for _ in range(num_thunderstorms):
        center_lon = random.uniform(lon_min, lon_max)
        center_lat = random.uniform(lat_min, lat_max)
        radius = random.uniform(0.05, 0.15)  # Radial size in degrees
        severity = random.choice([8, 9, 10])  # Higher values = more severe
        
        # Thunderstorms typically affect a range of altitudes
        base_fl = random.randint(fl_min, fl_max - 100)
        top_fl = base_fl + random.randint(80, 150)
        top_fl = min(top_fl, fl_max)
        
        # Create multiple points to represent the storm
        num_points = 15
        for i in range(num_points):
            # Create a point within the storm radius
            angle = random.uniform(0, 2 * np.pi)
            dist = random.triangular(0, radius, radius * 0.7)  # More dense toward center
            point_lon = center_lon + dist * np.cos(angle)
            point_lat = center_lat + dist * np.sin(angle)
            point_fl = random.randint(base_fl, top_fl)
            
            weather_data.append({
                'longitude': point_lon,
                'latitude': point_lat,
                'flight_level': point_fl,
                'type': 'Thunderstorm',
                'severity': severity
            })
    
    # Turbulence areas (moderate, larger areas)
    num_turbulence = random.randint(3, 6)
    for _ in range(num_turbulence):
        center_lon = random.uniform(lon_min, lon_max)
        center_lat = random.uniform(lat_min, lat_max)
        x_radius = random.uniform(0.1, 0.3)  # Larger areas
        y_radius = random.uniform(0.1, 0.3)
        severity = random.choice([3, 4, 5, 6])  # Moderate severity
        
        # Turbulence can be at specific flight levels
        base_fl = random.randint(fl_min, fl_max - 50)
        top_fl = base_fl + random.randint(30, 80)
        top_fl = min(top_fl, fl_max)
        
        num_points = 12
        for i in range(num_points):
            # Create elliptical distribution
            angle = random.uniform(0, 2 * np.pi)
            dist_x = random.uniform(0, x_radius)
            dist_y = random.uniform(0, y_radius)
            point_lon = center_lon + dist_x * np.cos(angle)
            point_lat = center_lat + dist_y * np.sin(angle)
            point_fl = random.randint(base_fl, top_fl)
            
            weather_data.append({
                'longitude': point_lon,
                'latitude': point_lat,
                'flight_level': point_fl,
                'type': 'Turbulence',
                'severity': severity
            })
    
    # Icing conditions (moderate to severe, altitude specific)
    num_icing = random.randint(2, 5)
    for _ in range(num_icing):
        center_lon = random.uniform(lon_min, lon_max)
        center_lat = random.uniform(lat_min, lat_max)
        radius = random.uniform(0.1, 0.25)
        severity = random.choice([4, 5, 6, 7])
        
        # Icing typically occurs in specific altitude bands
        base_fl = random.randint(fl_min, 350)  # More common at lower altitudes
        thickness = random.randint(20, 60)
        top_fl = base_fl + thickness
        top_fl = min(top_fl, fl_max)
        
        num_points = 10
        for i in range(num_points):
            angle = random.uniform(0, 2 * np.pi)
            dist = random.uniform(0, radius)
            point_lon = center_lon + dist * np.cos(angle)
            point_lat = center_lat + dist * np.sin(angle)
            # Icing has narrow vertical distribution
            point_fl = random.randint(base_fl, top_fl)
            
            weather_data.append({
                'longitude': point_lon,
                'latitude': point_lat,
                'flight_level': point_fl,
                'type': 'Icing',
                'severity': severity
            })
    
    return pd.DataFrame(weather_data)

# Generate weather data
weather_df = generate_fake_weather()

# Create a KDTree for weather data for efficient proximity searches
weather_tree = KDTree(weather_df[['longitude', 'latitude', 'flight_level']].values)

def distance_score(point1, point2):
    """Calculate distance score including both horizontal and vertical components"""
    horizontal_dist = haversine(point1[:2], point2[:2])
    vertical_dist = abs(point1[2] - point2[2]) * 0.1  # Scale altitude difference
    return horizontal_dist + vertical_dist

def find_optimal_path(start, end, tree, high_risk_zones, weather_df, weather_tree, max_iterations=10000):
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
            
            # Calculate contrail risk penalty
            contrail_risk_factor = 0
            for _, row in high_risk_zones.iterrows():
                risk_point = (row['longitude'], row['latitude'])
                distance_to_risk = haversine(neighbor[:2], risk_point)
                if distance_to_risk < 20:  # Within 20km
                    # Penalty decreases with distance
                    contrail_risk_factor += max(0, 20 - distance_to_risk) * 2
            
            # Calculate weather penalty
            weather_penalty = 0
            # Search for weather near the neighbor point
            weather_dists, weather_indices = weather_tree.query([neighbor], k=5)
            for w_dist, w_idx in zip(weather_dists[0], weather_indices[0]):
                if w_dist < 0.2:  # Within ~20km
                    weather_point = weather_df.iloc[w_idx]
                    # Calculate penalty based on weather severity and distance
                    severity = weather_point['severity']
                    # Exponential penalty based on severity and proximity
                    weather_type_multiplier = {
                        'Thunderstorm': 3.0,  # Highest danger
                        'Turbulence': 1.5,    # Moderate danger
                        'Icing': 2.0          # Significant danger
                    }.get(weather_point['type'], 1.0)
                    
                    # Calculate altitude component - weather affects specific altitude bands
                    weather_fl = weather_point['flight_level']
                    fl_diff = abs(neighbor[2] - weather_fl)
                    vertical_factor = max(0, 1 - (fl_diff / 50))  # Reduced effect beyond 50 flight levels
                    
                    # Combined penalty
                    point_penalty = severity * weather_type_multiplier * (0.2 - w_dist) * vertical_factor * 30
                    weather_penalty += point_penalty
            
            # Update g_score (cost from start to neighbor)
            new_g_score = g_score + fuel_cost + altitude_cost + contrail_risk_factor + weather_penalty
            
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

# Find optimal path considering both contrail risk and weather
optimal_path = find_optimal_path(start, end, tree, high_risk_zones, weather_df, weather_tree)

# Calculate path statistics
if optimal_path:
    # Calculate total distance and fuel consumption
    total_distance = 0
    total_fuel = 0
    high_risk_exposure = 0
    weather_exposure = 0
    
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
        
        # Calculate midpoint of leg for exposure checking
        midpoint = ((point1[0] + point2[0])/2, (point1[1] + point2[1])/2, (point1[2] + point2[2])/2)
        
        # Calculate contrail risk exposure
        for _, row in high_risk_zones.iterrows():
            risk_point = (row['longitude'], row['latitude'])
            distance_to_risk = haversine(midpoint[:2], risk_point)
            if distance_to_risk < 20:
                high_risk_exposure += 1
        
        # Calculate weather exposure
        weather_dists, weather_indices = weather_tree.query([midpoint], k=5)
        for w_dist, w_idx in zip(weather_dists[0], weather_indices[0]):
            if w_dist < 0.2:  # Within ~20km
                weather_point = weather_df.iloc[w_idx]
                # Only count significant exposures
                if abs(midpoint[2] - weather_point['flight_level']) < 30:
                    weather_exposure += 1
    
    print(f"Total distance: {total_distance:.2f} km")
    print(f"Total fuel consumption: {total_fuel:.2f} units")
    print(f"High-risk contrail zone exposures: {high_risk_exposure}")
    print(f"Weather hazard exposures: {weather_exposure}")

    # Plot the path with both contrail and weather hazards
    plt.figure(figsize=(14, 10))
    
    # Plot high-risk contrail zones
    if not high_risk_zones.empty:
        plt.scatter(high_risk_zones['longitude'], high_risk_zones['latitude'], 
                   c='red', alpha=0.5, s=100, label='Contrail Risk Zones')
    
    # Plot weather hazards by type
    weather_types = weather_df['type'].unique()
    for weather_type in weather_types:
        weather_subset = weather_df[weather_df['type'] == weather_type]
        
        if weather_type == 'Thunderstorm':
            color = 'purple'
        elif weather_type == 'Turbulence':
            color = 'orange'
        elif weather_type == 'Icing':
            color = 'cyan'
        else:
            color = 'gray'
        
        plt.scatter(weather_subset['longitude'], weather_subset['latitude'], 
                   c=color, alpha=0.4, s=80, label=f'{weather_type}')
    
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
    plt.title("Flight Path Optimization with Contrail and Weather Avoidance")
    plt.legend(loc='best')
    plt.grid(True)
    
    # Plot altitude profile with weather indicators
    plt.figure(figsize=(14, 6))
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
    
    # Indicate weather on altitude profile
    for i in range(len(optimal_path) - 1):
        point1 = optimal_path[i]
        point2 = optimal_path[i + 1]
        midpoint = ((point1[0] + point2[0])/2, (point1[1] + point2[1])/2, (point1[2] + point2[2])/2)
        midpoint_dist = distances[i] + (distances[i+1] - distances[i])/2
        
        # Check for nearby weather
        weather_dists, weather_indices = weather_tree.query([midpoint], k=5)
        for w_dist, w_idx in zip(weather_dists[0], weather_indices[0]):
            if w_dist < 0.2:  # Within ~20km
                weather_point = weather_df.iloc[w_idx]
                
                # Skip if the altitude difference is large
                if abs(midpoint[2] - weather_point['flight_level']) > 30:
                    continue
                
                w_type = weather_point['type']
                
                # Different markers for different weather types
                if w_type == 'Thunderstorm':
                    color = 'purple'
                    marker = '*'
                elif w_type == 'Turbulence':
                    color = 'orange'
                    marker = 's'
                elif w_type == 'Icing':
                    color = 'cyan'
                    marker = '^'
                
                plt.scatter(midpoint_dist, weather_point['flight_level'], 
                           c=color, marker=marker, s=100, alpha=0.6)
    
    # Add a legend for weather types
    plt.scatter([], [], c='purple', marker='*', s=100, label='Thunderstorm')
    plt.scatter([], [], c='orange', marker='s', s=100, label='Turbulence')
    plt.scatter([], [], c='cyan', marker='^', s=100, label='Icing')
    
    plt.xlabel("Distance (km)")
    plt.ylabel("Flight Level")
    plt.title("Altitude Profile with Weather Hazards")
    plt.legend(loc='best')
    plt.grid(True)
    
    # Create a 3D visualization
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot flight path in 3D
    ax.plot([p[0] for p in optimal_path], 
            [p[1] for p in optimal_path], 
            [p[2] for p in optimal_path], 
            'b-', linewidth=2, label='Flight Path')
    
    # Plot high-risk contrail zones
    if not high_risk_zones.empty:
        ax.scatter(high_risk_zones['longitude'], 
                  high_risk_zones['latitude'], 
                  high_risk_zones['flight_level'],
                  c='red', alpha=0.3, s=50, label='Contrail Risk')
    
    # Plot weather by type
    for weather_type in weather_types:
        weather_subset = weather_df[weather_df['type'] == weather_type]
        
        if weather_type == 'Thunderstorm':
            color = 'purple'
        elif weather_type == 'Turbulence':
            color = 'orange'
        elif weather_type == 'Icing':
            color = 'cyan'
        else:
            color = 'gray'
        
        ax.scatter(weather_subset['longitude'], 
                  weather_subset['latitude'], 
                  weather_subset['flight_level'],
                  c=color, alpha=0.4, s=40, label=weather_type)
    
    # Plot start and end points
    ax.scatter([start[0]], [start[1]], [start[2]], 
              c='green', s=100, label='Start')
    ax.scatter([end[0]], [end[1]], [end[2]], 
              c='purple', s=100, label='End')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Flight Level')
    ax.set_title('3D Flight Path with Weather and Contrail Avoidance')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
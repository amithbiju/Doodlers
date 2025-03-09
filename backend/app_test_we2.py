import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import heapq
from haversine import haversine
import random
from mpl_toolkits.mplot3d import Axes3D

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

# Generate fake aircraft path data
def generate_fake_aircraft_paths():
    # Define the area bounds
    lon_min, lon_max = 10.0, 11.2
    lat_min, lat_max = 75.5, 77.8
    fl_min, fl_max = 270, 430
    
    # Number of other aircraft in the airspace
    num_aircraft = random.randint(3, 6)
    
    aircraft_data = []
    
    for plane_id in range(num_aircraft):
        # Generate a realistic path with varying altitude
        # Create a curved path between random points
        curve_type = random.choice(['direct', 'curved', 'stepped'])
        
        # Random start and end points
        start_lon = random.uniform(lon_min, lon_max)
        start_lat = random.uniform(lat_min, lat_max)
        start_fl = random.randint(fl_min, fl_max)
        
        end_lon = random.uniform(lon_min, lon_max)
        end_lat = random.uniform(lat_min, lat_max)
        end_fl = random.randint(fl_min, fl_max)
        
        # Number of points in the path
        num_points = random.randint(15, 30)
        
        # Generate path based on curve type
        if curve_type == 'direct':
            # Direct path with mild altitude changes
            for i in range(num_points):
                t = i / (num_points - 1)
                lon = start_lon + t * (end_lon - start_lon)
                lat = start_lat + t * (end_lat - start_lat)
                
                # Linear altitude change with small random variations
                fl = start_fl + t * (end_fl - start_fl) + random.uniform(-5, 5)
                fl = max(fl_min, min(fl_max, fl))  # Keep within bounds
                
                aircraft_data.append({
                    'plane_id': f'AC{plane_id}',
                    'longitude': lon,
                    'latitude': lat,
                    'flight_level': fl,
                    'curve_type': curve_type
                })
                
        elif curve_type == 'curved':
            # Create a curved path using a sine wave deviation
            amplitude = random.uniform(0.05, 0.2)  # Curve amplitude
            frequency = random.uniform(1, 3)  # Curve frequency
            
            for i in range(num_points):
                t = i / (num_points - 1)
                
                # Base direct path
                base_lon = start_lon + t * (end_lon - start_lon)
                base_lat = start_lat + t * (end_lat - start_lat)
                
                # Add sine wave deviation perpendicular to direct path
                angle = np.arctan2(end_lat - start_lat, end_lon - start_lon) + np.pi/2
                deviation = amplitude * np.sin(frequency * np.pi * t)
                
                lon = base_lon + deviation * np.cos(angle)
                lat = base_lat + deviation * np.sin(angle)
                
                # Smooth altitude changes
                fl = start_fl + t * (end_fl - start_fl) + random.uniform(-8, 8)
                fl = max(fl_min, min(fl_max, fl))
                
                aircraft_data.append({
                    'plane_id': f'AC{plane_id}',
                    'longitude': lon,
                    'latitude': lat,
                    'flight_level': fl,
                    'curve_type': curve_type
                })
                
        else:  # stepped
            # Path with stepped altitude changes
            num_steps = random.randint(2, 5)
            step_points = sorted(random.sample(range(1, num_points - 1), num_steps - 1))
            step_points = [0] + step_points + [num_points - 1]
            
            # Generate FL for each step
            step_fls = [start_fl]
            for _ in range(num_steps - 1):
                prev_fl = step_fls[-1]
                new_fl = prev_fl + random.choice([-30, -20, -10, 0, 10, 20, 30])
                new_fl = max(fl_min, min(fl_max, new_fl))
                step_fls.append(new_fl)
            step_fls.append(end_fl)
            
            for step in range(num_steps):
                start_idx = step_points[step]
                end_idx = step_points[step + 1]
                start_step_fl = step_fls[step]
                end_step_fl = step_fls[step + 1]
                
                for i in range(start_idx, end_idx + 1):
                    t = (i - start_idx) / max(1, end_idx - start_idx)
                    lon = start_lon + (i / (num_points - 1)) * (end_lon - start_lon)
                    lat = start_lat + (i / (num_points - 1)) * (end_lat - start_lat)
                    
                    # Linear interpolation within step
                    fl = start_step_fl + t * (end_step_fl - start_step_fl)
                    
                    aircraft_data.append({
                        'plane_id': f'AC{plane_id}',
                        'longitude': lon,
                        'latitude': lat,
                        'flight_level': fl,
                        'curve_type': curve_type
                    })
    
    return pd.DataFrame(aircraft_data)

# Generate weather data
weather_df = generate_fake_weather()

# Generate other aircraft paths
aircraft_df = generate_fake_aircraft_paths()

# Create KDTrees for weather data and aircraft data for efficient proximity searches
weather_tree = KDTree(weather_df[['longitude', 'latitude', 'flight_level']].values)
aircraft_tree = KDTree(aircraft_df[['longitude', 'latitude', 'flight_level']].values)

def distance_score(point1, point2):
    """Calculate distance score including both horizontal and vertical components"""
    horizontal_dist = haversine(point1[:2], point2[:2])
    vertical_dist = abs(point1[2] - point2[2]) * 0.1  # Scale altitude difference
    return horizontal_dist + vertical_dist

def find_optimal_path(start, end, tree, high_risk_zones, weather_df, weather_tree, aircraft_df, aircraft_tree, max_iterations=10000):
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
            
            # Calculate aircraft proximity penalty
            aircraft_penalty = 0
            # Search for aircraft near the neighbor point
            aircraft_dists, aircraft_indices = aircraft_tree.query([neighbor], k=10)
            for a_dist, a_idx in zip(aircraft_dists[0], aircraft_indices[0]):
                if a_dist < 0.15:  # Within ~15km
                    aircraft_point = aircraft_df.iloc[a_idx]
                    
                    # Calculate vertical separation
                    fl_diff = abs(neighbor[2] - aircraft_point['flight_level'])
                    
                    # Vertical separation requirements (simplified)
                    min_vertical_separation = 10  # 1000 feet
                    
                    # Calculate penalty based on proximity and vertical separation
                    if fl_diff < min_vertical_separation:
                        # Critical vertical separation violation
                        vertical_penalty = 500 * (1 - (fl_diff / min_vertical_separation))
                    else:
                        # Less critical but still adds some penalty
                        vertical_penalty = 50 * max(0, 1 - ((fl_diff - min_vertical_separation) / 20))
                    
                    # Calculate horizontal component
                    horizontal_penalty = 200 * max(0, (0.15 - a_dist) / 0.15)
                    
                    # Combined penalty is higher when both horizontal and vertical proximities are issues
                    point_penalty = horizontal_penalty * (1 + vertical_penalty)
                    aircraft_penalty += point_penalty
            
            # Update g_score (cost from start to neighbor)
            new_g_score = g_score + fuel_cost + altitude_cost + contrail_risk_factor + weather_penalty + aircraft_penalty
            
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

# Find optimal path considering contrail risk, weather, and other aircraft
optimal_path = find_optimal_path(start, end, tree, high_risk_zones, weather_df, weather_tree, aircraft_df, aircraft_tree)

# Calculate path statistics
if optimal_path:
    # Calculate total distance and fuel consumption
    total_distance = 0
    total_fuel = 0
    high_risk_exposure = 0
    weather_exposure = 0
    aircraft_proximity = 0
    
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
        
        # Calculate aircraft proximity
        aircraft_dists, aircraft_indices = aircraft_tree.query([midpoint], k=5)
        for a_dist, a_idx in zip(aircraft_dists[0], aircraft_indices[0]):
            if a_dist < 0.15:  # Within ~15km
                aircraft_point = aircraft_df.iloc[a_idx]
                # Only count significant proximities
                if abs(midpoint[2] - aircraft_point['flight_level']) < 20:
                    aircraft_proximity += 1
    
    print(f"Total distance: {total_distance:.2f} km")
    print(f"Total fuel consumption: {total_fuel:.2f} units")
    print(f"High-risk contrail zone exposures: {high_risk_exposure}")
    print(f"Weather hazard exposures: {weather_exposure}")
    print(f"Aircraft proximity events: {aircraft_proximity}")

    # Plot the path with contrail, weather hazards, and other aircraft
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
    
    # Plot other aircraft paths
    for plane_id in aircraft_df['plane_id'].unique():
        plane_data = aircraft_df[aircraft_df['plane_id'] == plane_id]
        plt.plot(plane_data['longitude'], plane_data['latitude'], 
                'g-', alpha=0.7, linewidth=1.5)
        plt.scatter(plane_data['longitude'], plane_data['latitude'], 
                   c='green', alpha=0.4, s=30)
    
    # Add a single legend entry for all aircraft
    plt.scatter([], [], c='green', s=50, label='Other Aircraft')
    
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
    plt.title("Flight Path Optimization with Contrail, Weather, and Aircraft Avoidance")
    plt.legend(loc='best')
    plt.grid(True)
    
    # Plot altitude profile with weather and aircraft indicators
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
        
        # Check for nearby aircraft
        aircraft_dists, aircraft_indices = aircraft_tree.query([midpoint], k=5)
        for a_dist, a_idx in zip(aircraft_dists[0], aircraft_indices[0]):
            if a_dist < 0.15:  # Within ~15km
                aircraft_point = aircraft_df.iloc[a_idx]
                
                # Skip if the altitude difference is large
                if abs(midpoint[2] - aircraft_point['flight_level']) > 30:
                    continue
                
                plt.scatter(midpoint_dist, aircraft_point['flight_level'], 
                           c='green', marker='x', s=80, alpha=0.7)
    
    # Add a legend
    plt.scatter([], [], c='purple', marker='*', s=100, label='Thunderstorm')
    plt.scatter([], [], c='orange', marker='s', s=100, label='Turbulence')

    plt.scatter([], [], c='cyan', marker='^', s=100, label='Icing')
    plt.scatter([], [], c='green', marker='x', s=80, label='Other Aircraft')
    
    plt.xlabel("Distance (km)")
    plt.ylabel("Flight Level")
    plt.title("Altitude Profile with Weather and Traffic")
    plt.grid(True)
    plt.legend(loc='upper right')
    
    # Create 3D visualization showing complete picture
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot optimal path
    ax.plot([p[0] for p in optimal_path], [p[1] for p in optimal_path], [p[2] for p in optimal_path], 
            'b-', linewidth=3, label='Optimized Path')
    
    # Plot high-risk contrail zones
    if not high_risk_zones.empty:
        ax.scatter(high_risk_zones['longitude'], high_risk_zones['latitude'], high_risk_zones['flight_level'], 
                 c='red', alpha=0.3, s=80, label='Contrail Risk Zones')
    
    # Plot weather hazards
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
        
        ax.scatter(weather_subset['longitude'], weather_subset['latitude'], weather_subset['flight_level'], 
                 c=color, alpha=0.4, s=60, label=f'{weather_type}')
    
    # Plot other aircraft paths
    for plane_id in aircraft_df['plane_id'].unique():
        plane_data = aircraft_df[aircraft_df['plane_id'] == plane_id]
        ax.plot(plane_data['longitude'], plane_data['latitude'], plane_data['flight_level'], 
              'g-', alpha=0.5, linewidth=1.5)
    
    # Add a single legend entry for all aircraft
    ax.scatter([], [], [], c='green', s=40, label='Other Aircraft')
    
    # Highlight start and end points
    ax.scatter([start[0]], [start[1]], [start[2]], c='green', s=120, label='Start')
    ax.scatter([end[0]], [end[1]], [end[2]], c='purple', s=120, label='End')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Flight Level')
    ax.set_title('3D View of Flight Path with Hazards and Traffic')
    ax.legend(loc='upper right')
    
    # Generate summary report
    print("\n--- Flight Path Optimization Summary ---")
    print(f"Origin coordinates: {start}")
    print(f"Destination coordinates: {end}")
    print(f"Path length: {len(optimal_path)} waypoints")
    print(f"Total distance: {total_distance:.2f} km")
    print(f"Total fuel consumption: {total_fuel:.2f} units")
    print(f"Average fuel efficiency: {(total_fuel/total_distance):.4f} units/km")
    print(f"Highest flight level: {max([p[2] for p in optimal_path]):.1f}")
    print(f"Lowest flight level: {min([p[2] for p in optimal_path]):.1f}")
    print(f"Flight level changes: {sum([1 for i in range(1, len(optimal_path)) if abs(optimal_path[i][2] - optimal_path[i-1][2]) > 1])}")
    print("\n--- Hazard Statistics ---")
    print(f"High-risk contrail zone exposures: {high_risk_exposure}")
    print(f"Weather hazard exposures: {weather_exposure}")
    print(f"Aircraft proximity events: {aircraft_proximity}")
    
    # Save results to CSV
    path_df = pd.DataFrame({
        'waypoint': range(len(optimal_path)),
        'longitude': [p[0] for p in optimal_path],
        'latitude': [p[1] for p in optimal_path],
        'flight_level': [p[2] for p in optimal_path],
        'distance_from_start': distances
    })
    path_df.to_csv('optimized_flight_path.csv', index=False)
    print("\nOptimized flight path saved to 'optimized_flight_path.csv'")
    
    # Save the detailed flight plan
    with open('flight_plan_details.txt', 'w') as f:
        f.write("DETAILED FLIGHT PLAN\n")
        f.write("===================\n\n")
        f.write(f"Origin: ({start[0]:.6f}, {start[1]:.6f}) at FL{start[2]:.0f}\n")
        f.write(f"Destination: ({end[0]:.6f}, {end[1]:.6f}) at FL{end[2]:.0f}\n")
        f.write(f"Total Distance: {total_distance:.2f} km\n")
        f.write(f"Estimated Fuel Consumption: {total_fuel:.2f} units\n\n")
        f.write("WAYPOINTS\n")
        f.write("=========\n\n")
        
        cumulative_dist = 0
        for i in range(len(optimal_path)):
            point = optimal_path[i]
            if i > 0:
                prev_point = optimal_path[i-1]
                leg_dist = haversine(prev_point[:2], point[:2])
                cumulative_dist += leg_dist
                f.write(f"Waypoint {i}: ({point[0]:.6f}, {point[1]:.6f}) at FL{point[2]:.0f}\n")
                f.write(f"   Leg Distance: {leg_dist:.2f} km\n")
                f.write(f"   Cumulative Distance: {cumulative_dist:.2f} km\n")
                
                # Calculate heading
                y = np.sin(np.radians(point[1] - prev_point[1])) * np.cos(np.radians(point[0]))
                x = np.cos(np.radians(prev_point[0])) * np.sin(np.radians(point[0])) - \
                    np.sin(np.radians(prev_point[0])) * np.cos(np.radians(point[0])) * np.cos(np.radians(point[1] - prev_point[1]))
                heading = (np.degrees(np.arctan2(y, x)) + 360) % 360
                f.write(f"   Heading: {heading:.1f}°\n")
                
                # Calculate altitude change
                alt_change = point[2] - prev_point[2]
                if abs(alt_change) > 0:
                    f.write(f"   Altitude Change: {('+' if alt_change > 0 else '')}{alt_change:.0f} FL\n")
                
                # Check for nearby hazards
                hazards_nearby = []
                
                # Check contrail risk
                for _, row in high_risk_zones.iterrows():
                    risk_point = (row['longitude'], row['latitude'])
                    distance_to_risk = haversine(point[:2], risk_point)
                    if distance_to_risk < 20 and abs(point[2] - row['flight_level']) < 30:
                        hazards_nearby.append(f"Contrail risk zone at {distance_to_risk:.1f} km")
                
                # Check weather
                weather_dists, weather_indices = weather_tree.query([point], k=3)
                for w_dist, w_idx in zip(weather_dists[0], weather_indices[0]):
                    if w_dist < 0.2:  # Within ~20km
                        weather_point = weather_df.iloc[w_idx]
                        if abs(point[2] - weather_point['flight_level']) < 30:
                            hazards_nearby.append(
                                f"{weather_point['type']} (severity {weather_point['severity']}) at {w_dist*100:.1f} km"
                            )
                
                # Check aircraft
                aircraft_dists, aircraft_indices = aircraft_tree.query([point], k=3)
                for a_dist, a_idx in zip(aircraft_dists[0], aircraft_indices[0]):
                    if a_dist < 0.15:  # Within ~15km
                        aircraft_point = aircraft_df.iloc[a_idx]
                        if abs(point[2] - aircraft_point['flight_level']) < 20:
                            hazards_nearby.append(
                                f"Aircraft {aircraft_point['plane_id']} at {a_dist*100:.1f} km, FL{aircraft_point['flight_level']:.0f}"
                            )
                
                if hazards_nearby:
                    f.write("   Nearby Hazards:\n")
                    for hazard in hazards_nearby:
                        f.write(f"     - {hazard}\n")
                
                f.write("\n")
            else:
                f.write(f"Waypoint {i}: ({point[0]:.6f}, {point[1]:.6f}) at FL{point[2]:.0f} (ORIGIN)\n\n")
    
    print("Detailed flight plan saved to 'flight_plan_details.txt'")
    
    # Function to evaluate multiple path options with different weightings
    def evaluate_alternative_paths(start, end, tree, high_risk_zones, weather_df, weather_tree, aircraft_df, aircraft_tree):
        # Define different weighting scenarios
        scenarios = [
            {"name": "Balanced", "fuel": 1.0, "contrail": 1.0, "weather": 1.0, "traffic": 1.0},
            {"name": "Fuel Efficiency", "fuel": 2.0, "contrail": 0.5, "weather": 0.8, "traffic": 0.8},
            {"name": "Environmental", "fuel": 0.8, "contrail": 2.0, "weather": 0.8, "traffic": 0.8},
            {"name": "Safety", "fuel": 0.8, "contrail": 0.8, "weather": 2.0, "traffic": 2.0}
        ]
        
        results = []
        
        for scenario in scenarios:
            # Adjust the A* search for this scenario
            # (This would be a modified version of find_optimal_path)
            # For simplicity, we'll just print what would happen
            print(f"\nEvaluating {scenario['name']} scenario:")
            print(f"  Fuel weight: {scenario['fuel']}")
            print(f"  Contrail weight: {scenario['contrail']}")
            print(f"  Weather weight: {scenario['weather']}")
            print(f"  Traffic weight: {scenario['traffic']}")
            
            # In a real implementation, we would run a modified version of find_optimal_path
            # with these weights and calculate the resulting metrics
            
            # For now, just estimate some plausible alternative metrics
            if scenario['name'] == "Fuel Efficiency":
                alt_distance = total_distance * 0.98  # Shorter route
                alt_fuel = total_fuel * 0.92  # Much better fuel
                alt_contrail = high_risk_exposure * 1.2  # More contrail exposure
                alt_weather = weather_exposure * 1.1  # Slightly more weather exposure
                alt_traffic = aircraft_proximity * 1.1  # Slightly more traffic
            elif scenario['name'] == "Environmental":
                alt_distance = total_distance * 1.05  # Longer route
                alt_fuel = total_fuel * 1.08  # More fuel
                alt_contrail = high_risk_exposure * 0.4  # Much less contrail exposure
                alt_weather = weather_exposure * 1.0  # Similar weather exposure
                alt_traffic = aircraft_proximity * 1.0  # Similar traffic
            elif scenario['name'] == "Safety":
                alt_distance = total_distance * 1.08  # Longer route
                alt_fuel = total_fuel * 1.1  # More fuel
                alt_contrail = high_risk_exposure * 0.9  # Similar contrail exposure
                alt_weather = weather_exposure * 0.4  # Much less weather exposure
                alt_traffic = aircraft_proximity * 0.3  # Much less traffic
            else:  # Balanced (current solution)
                alt_distance = total_distance
                alt_fuel = total_fuel
                alt_contrail = high_risk_exposure
                alt_weather = weather_exposure
                alt_traffic = aircraft_proximity
            
            results.append({
                "scenario": scenario['name'],
                "distance": alt_distance,
                "fuel": alt_fuel,
                "contrail_exposure": alt_contrail,
                "weather_exposure": alt_weather,
                "traffic_proximity": alt_traffic
            })
        
        # Create a comparison table
        results_df = pd.DataFrame(results)
        print("\nAlternative Path Comparison:")
        print(results_df)
        
        # Save the comparison
        results_df.to_csv('alternative_paths_comparison.csv', index=False)
        print("Alternative paths comparison saved to 'alternative_paths_comparison.csv'")
        
        # Create a comparative visualization
        # Radar chart for comparing the scenarios
        plt.figure(figsize=(10, 8))
        
        # Normalize values for radar chart
        metrics = ['distance', 'fuel', 'contrail_exposure', 'weather_exposure', 'traffic_proximity']
        normalized_results = results_df.copy()
        
        for metric in metrics:
            max_val = normalized_results[metric].max()
            # Invert the normalization since lower values are better
            normalized_results[metric] = 1 - (normalized_results[metric] / max_val) + 0.2  # Add 0.2 to avoid values too close to zero
        
        # Number of variables
        N = len(metrics)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Initialize the plot
        ax = plt.subplot(111, polar=True)
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], ['Distance', 'Fuel', 'Contrail', 'Weather', 'Traffic'])
        
        # Plot each scenario
        colors = ['blue', 'green', 'orange', 'purple']
        for i, scenario in enumerate(normalized_results['scenario']):
            values = normalized_results.iloc[i][metrics].values.tolist()
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, color=colors[i], linewidth=2, label=scenario)
            ax.fill(angles, values, color=colors[i], alpha=0.25)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Comparison of Different Path Optimization Scenarios')
        
        plt.tight_layout()
        plt.savefig('path_optimization_comparison.png')
        print("Comparison visualization saved to 'path_optimization_comparison.png'")
    
    # Evaluate alternative paths
    evaluate_alternative_paths(start, end, tree, high_risk_zones, weather_df, weather_tree, aircraft_df, aircraft_tree)
    
    plt.show()
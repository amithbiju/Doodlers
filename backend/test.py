import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Define aircraft components with different demand patterns
components = [
    {
        "part_id": "B777-ENG-001",
        "name": "Turbine Blade (CF6 Engine)",
        "aircraft_type": "Boeing 777",
        "lead_time": 45,
        "base_demand": 0.1,
        "trend": 0.02,
        "seasonality_amplitude": 2.5
    },
    {
        "part_id": "A320-HYD-002",
        "name": "Hydraulic Pump",
        "aircraft_type": "Airbus A320",
        "lead_time": 30,
        "base_demand": 0.3,
        "trend": 0.01,
        "seasonality_amplitude": 1.8
    },
    {
        "part_id": "B737-LG-003",
        "name": "Landing Gear Actuator",
        "aircraft_type": "Boeing 737",
        "lead_time": 60,
        "base_demand": 0.05,
        "trend": 0.015,
        "seasonality_amplitude": 1.2
    },
    {
        "part_id": "A350-AV-004",
        "name": "Avionics Cooling Fan",
        "aircraft_type": "Airbus A350",
        "lead_time": 25,
        "base_demand": 0.2,
        "trend": 0.03,
        "seasonality_amplitude": 1.5
    }
]

def generate_component_demand(component, start_date="2022-01-01", end_date="2023-12-31"):
    date_range = pd.date_range(start=start_date, end=end_date)
    days = len(date_range)
    
    # Base demand with trend
    trend = np.linspace(0, component['trend'], days)
    
    # Weekly seasonality (higher demand during weekends)
    weekly_seasonality = component['seasonality_amplitude'] * np.sin(
        np.arange(days) * (2 * np.pi / 7)
    )
    
    # Yearly seasonality (higher in summer/winter months)
    yearly_seasonality = component['seasonality_amplitude'] * np.sin(
        np.arange(days) * (2 * np.pi / 365) + 1.5
    )
    
    # Random spikes (Poisson process)
    spikes = np.random.poisson(0.01, days)
    
    # Combine components
    demand = (
        component['base_demand'] +
        trend +
        weekly_seasonality +
        yearly_seasonality +
        spikes
    )
    
    # Convert to actual counts (non-negative)
    demand = np.round(np.abs(demand)).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'ds': date_range,
        'y': demand,
        'part_id': component['part_id'],
        'aircraft_type': component['aircraft_type']
    })
    
    return df

# Generate dataset for all components
full_dataset = pd.concat(
    [generate_component_demand(c) for c in components],
    ignore_index=True
)

# Add current stock information
inventory_data = pd.DataFrame([{
    'part_id': c['part_id'],
    'current_stock': np.random.randint(5, 20),
    'lead_time': c['lead_time'],
    'min_stock': np.random.randint(2, 5)
} for c in components])

# Save sample datasets
full_dataset.to_csv("aircraft_demand_dataset.csv", index=False)
inventory_data.to_csv("component_inventory.csv", index=False)

print("Sample Demand Data:")
print(full_dataset.head())
print("\nInventory Data:")
print(inventory_data.head())
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
from fireb import update,initialize_firebase


initialize_firebase()
update()




# Load datasets
demand_data = pd.read_csv("aircraft_demand_dataset.csv")
inventory_data = pd.read_csv("component_inventory.csv")

# Convert 'ds' to datetime
demand_data["ds"] = pd.to_datetime(demand_data["ds"])

def train_prophet_model(part_data):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(part_data[["ds", "y"]])
    return model

def generate_forecasts(model, periods=365):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

# Example: Forecast for one part
part_id = "B777-ENG-001"
part_data = demand_data[demand_data["part_id"] == part_id]
model = train_prophet_model(part_data)
forecast = generate_forecasts(model)


def calculate_reorder_points(forecast, inventory_data, part_id):
    part_inventory = inventory_data[inventory_data["part_id"] == part_id].iloc[0]
    current_stock = part_inventory["current_stock"]
    lead_time = part_inventory["lead_time"]
    min_stock = part_inventory["min_stock"]
    
    # Calculate cumulative demand over lead time
    forecast["cumulative_demand"] = forecast["yhat"].cumsum()
    
    # Find the date when stock will drop below min_stock
    reorder_date = forecast[forecast["cumulative_demand"] > (current_stock - min_stock)]["ds"].min()
    
    return reorder_date, lead_time

# Example: Calculate reorder point for one part
reorder_date, lead_time = calculate_reorder_points(forecast, inventory_data, part_id)
print(f"Reorder date for {part_id}: {reorder_date}, Lead time: {lead_time} days")


def identify_urgent_orders(reorder_date, lead_time, current_date):
    # Calculate the latest date to place the order
    lead_time = int(lead_time)

    latest_order_date = reorder_date - timedelta(days=lead_time)
    
    # Calculate days left to order
    days_left = (latest_order_date - current_date).days
    
    # Check if urgent (less than 7 days left)
    is_urgent = days_left < 40
    return days_left, is_urgent

# Example: Check if order is urgent
current_date = datetime.now()
date_string = "2025-1-1 15:30:45.123456"

current_date = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S.%f")
days_left, is_urgent = identify_urgent_orders(reorder_date, lead_time, current_date)
print(f"Days left to order: {days_left}, Urgent: {is_urgent}")


# Results storage
orders_to_place = []

# Loop through all parts
for part_id in inventory_data["part_id"].unique():
    # Filter data for the part
    part_data = demand_data[demand_data["part_id"] == part_id]
    part_inventory = inventory_data[inventory_data["part_id"] == part_id].iloc[0]
    
    # Train Prophet model and generate forecasts
    model = train_prophet_model(part_data)
    forecast = generate_forecasts(model)
    
    # Calculate reorder point
    reorder_date, lead_time = calculate_reorder_points(forecast, inventory_data, part_id)
    
    # Check if order is urgent
    days_left, is_urgent = identify_urgent_orders(reorder_date, lead_time, current_date)
    
    # Add to results if urgent
    if is_urgent:
        orders_to_place.append({
            "part_id": part_id,
            "reorder_date": reorder_date,
            "lead_time": lead_time,
            "days_left": days_left,
            "current_stock": part_inventory["current_stock"],
            "min_stock": part_inventory["min_stock"]
        })

# Convert results to DataFrame
orders_df = pd.DataFrame(orders_to_place)
print("Orders to place:")
print(orders_df)
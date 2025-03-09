from flask import Flask, jsonify
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
from fireb import update, initialize_firebase
import numpy as np

app = Flask(__name__)

# Initialize Firebase
def initialize():
    initialize_firebase()
    update()

initialize()

# Load datasets
demand_data = pd.read_csv("aircraft_demand_dataset.csv")
inventory_data = pd.read_csv("component_inventory.csv")

# Convert 'ds' to datetime
demand_data["ds"] = pd.to_datetime(demand_data["ds"])

def train_prophet_model(part_data):
    if part_data.shape[0] < 2:
        return None  # Not enough data to train
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(part_data[["ds", "y"]])
    return model

def generate_forecasts(model, periods=365):
    if model is None:
        return None
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]]

def calculate_reorder_points(forecast, inventory_data, part_id):
    part_inventory = inventory_data[inventory_data["part_id"] == part_id].iloc[0]
    current_stock = part_inventory["current_stock"]
    lead_time = part_inventory["lead_time"]
    min_stock = part_inventory["min_stock"]
    
    if forecast is None or forecast.empty:
        return None, lead_time
    
    forecast["cumulative_demand"] = forecast["yhat"].cumsum()
    reorder_date = forecast[forecast["cumulative_demand"] > (current_stock - min_stock)]["ds"].min()
    
    return reorder_date, lead_time

def identify_urgent_orders(reorder_date, lead_time, current_date):
    if pd.isna(reorder_date):
        return float("inf"), False  # No valid reorder date
    
    latest_order_date = reorder_date - timedelta(days=int(lead_time))
    days_left = (latest_order_date - current_date).days
    is_urgent = days_left < 40
    return days_left, is_urgent

@app.route("/get_orders", methods=["GET"])
def get_orders():
    orders_to_place = []
    current_date = datetime.strptime("2025-1-1 15:30:45.123456", "%Y-%m-%d %H:%M:%S.%f")
    
    for part_id in inventory_data["part_id"].unique():
        part_data = demand_data[demand_data["part_id"] == part_id]
        if part_data.empty:
            continue
        
        part_inventory = inventory_data[inventory_data["part_id"] == part_id].iloc[0]
        model = train_prophet_model(part_data)
        forecast = generate_forecasts(model)
        reorder_date, lead_time = calculate_reorder_points(forecast, inventory_data, part_id)
        days_left, is_urgent = identify_urgent_orders(reorder_date, lead_time, current_date)
        
        if is_urgent:
            orders_to_place.append({
                "part_id": part_id,
                "reorder_date": reorder_date.strftime('%Y-%m-%d') if reorder_date else None,
                "lead_time": lead_time,
                "days_left": days_left,
                "current_stock": part_inventory["current_stock"],
                "min_stock": part_inventory["min_stock"]
            })

    for order in orders_to_place:
        for key, value in order.items():
            if isinstance(value, (np.integer, np.floating)):
                order[key] = value.item()  # Convert to native Python type


    return jsonify(orders_to_place)

if __name__ == "__main__":
    app.run(debug=True)

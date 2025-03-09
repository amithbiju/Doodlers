from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Handle Prophet import (optional dependency)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    logger.warning("Prophet library not found. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

# Handle Firebase initialization (optional)
try:
    from fireb import update, initialize_firebase
    
    def initialize():
        logger.info("Initializing Firebase connection...")
        initialize_firebase()
        update()
    
    initialize()
    FIREBASE_AVAILABLE = True
except ImportError:
    logger.warning("Firebase module not found. Some functionality may be limited.")
    FIREBASE_AVAILABLE = False

# Load datasets with error handling
def load_data():
    try:
        # Check if files exist
        demand_path = "aircraft_demand_dataset.csv"
        inventory_path = "component_inventory.csv"
        
        if not os.path.exists(demand_path):
            logger.error(f"File not found: {demand_path}")
            return None, None
            
        if not os.path.exists(inventory_path):
            logger.error(f"File not found: {inventory_path}")
            return None, None
            
        # Load data
        demand_data = pd.read_csv(demand_path)
        inventory_data = pd.read_csv(inventory_path)
        
        # Convert 'ds' column to datetime
        if 'ds' in demand_data.columns:
            demand_data["ds"] = pd.to_datetime(demand_data["ds"])
        else:
            logger.error("Column 'ds' not found in demand data")
            return None, None
            
        logger.info(f"Loaded {len(demand_data)} demand records and {len(inventory_data)} inventory records")
        return demand_data, inventory_data
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None, None

# Load datasets
demand_data, inventory_data = load_data()

def train_prophet_model(part_data):
    if not PROPHET_AVAILABLE:
        logger.warning("Prophet not available, skipping forecast")
        return None
        
    if part_data.shape[0] < 2:
        logger.warning(f"Not enough data to train model for part (only {part_data.shape[0]} records)")
        return None
        
    try:
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(part_data[["ds", "y"]])
        return model
    except Exception as e:
        logger.error(f"Error training Prophet model: {str(e)}")
        return None

def generate_forecasts(model, periods=365):
    if model is None:
        return None
        
    try:
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast[["ds", "yhat"]]
    except Exception as e:
        logger.error(f"Error generating forecasts: {str(e)}")
        return None

def calculate_reorder_points(forecast, inventory_data, part_id):
    try:
        part_inventory = inventory_data[inventory_data["part_id"] == part_id]
        if part_inventory.empty:
            logger.warning(f"Part ID {part_id} not found in inventory data")
            return None, 0
            
        part_inventory = part_inventory.iloc[0]
        current_stock = part_inventory["current_stock"]
        lead_time = part_inventory["lead_time"]
        min_stock = part_inventory["min_stock"]
        
        if forecast is None or forecast.empty:
            return None, lead_time
        
        forecast["cumulative_demand"] = forecast["yhat"].cumsum()
        reorder_points = forecast[forecast["cumulative_demand"] > (current_stock - min_stock)]
        
        if reorder_points.empty:
            logger.info(f"No reorder points needed for part {part_id} within forecast period")
            return None, lead_time
            
        reorder_date = reorder_points["ds"].min()
        
        return reorder_date, lead_time
    except Exception as e:
        logger.error(f"Error calculating reorder points: {str(e)}")
        return None, 0

def identify_urgent_orders(reorder_date, lead_time, current_date):
    if pd.isna(reorder_date):
        return float("inf"), False  # No valid reorder date
    
    try:
        latest_order_date = reorder_date - timedelta(days=int(lead_time))
        days_left = (latest_order_date - current_date).days
        is_urgent = days_left < 40
        return days_left, is_urgent
    except Exception as e:
        logger.error(f"Error identifying urgent orders: {str(e)}")
        return float("inf"), False

@app.route("/")
def index():
    return "Aircraft Parts Forecasting API. Use /get_orders to retrieve urgent orders."

@app.route("/get_orders", methods=["GET"])
def get_orders():
    if demand_data is None or inventory_data is None:
        return jsonify({"error": "Data files not loaded correctly"}), 500
        
    orders_to_place = []
    # Use current date or fixed date for testing
    try:
        current_date = datetime.now()
        # For testing with fixed date:
        current_date = datetime.strptime("2025-01-01 15:30:45", "%Y-%m-%d %H:%M:%S")
        
        logger.info(f"Processing orders as of {current_date}")
        
        for part_id in inventory_data["part_id"].unique():
            part_data = demand_data[demand_data["part_id"] == part_id]
            if part_data.empty:
                logger.warning(f"No demand data for part ID {part_id}")
                continue
            
            part_inventory = inventory_data[inventory_data["part_id"] == part_id].iloc[0]
            model = train_prophet_model(part_data)
            forecast = generate_forecasts(model)
            reorder_date, lead_time = calculate_reorder_points(forecast, inventory_data, part_id)
            days_left, is_urgent = identify_urgent_orders(reorder_date, lead_time, current_date)
            
            if is_urgent:
                order_info = {
                    "part_id": part_id,
                    "reorder_date": reorder_date.strftime('%Y-%m-%d') if reorder_date else None,
                    "lead_time": lead_time,
                    "days_left": days_left,
                    "current_stock": part_inventory["current_stock"],
                    "min_stock": part_inventory["min_stock"]
                }
                
                # Convert numpy types to native Python types
                for key, value in order_info.items():
                    if isinstance(value, (np.integer, np.floating)):
                        order_info[key] = value.item()
                
                orders_to_place.append(order_info)
                
        logger.info(f"Found {len(orders_to_place)} urgent orders")
        return jsonify(orders_to_place)
        
    except Exception as e:
        logger.error(f"Error processing orders: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    status = {
        "status": "ok",
        "prophet_available": PROPHET_AVAILABLE,
        "firebase_available": FIREBASE_AVAILABLE,
        "demand_data_loaded": demand_data is not None,
        "inventory_data_loaded": inventory_data is not None
    }
    return jsonify(status)

if __name__ == "__main__":
    logger.info("Starting Flask application...")
    app.run(debug=True, host="0.0.0.0", port=5000)
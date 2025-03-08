# app.py - Main Flask Application

from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore, db
import numpy as np
import tensorflow as tf
import weather_service
import flight_service
import optimization_service
from ml_models import contrail_prediction_model

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Firebase
cred = credentials.Certificate("path/to/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://smart-contrails-system.firebaseio.com'
})
firestore_db = firestore.client()
realtime_db = db.reference()

# Load ML model
model = contrail_prediction_model.load_model("models/contrail_predictor_v1.h5")

@app.route('/api/weather', methods=['GET'])
def get_weather():
    """Get current weather conditions for a specified area."""
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    altitude = request.args.get('altitude', type=float)
    
    try:
        weather_data = weather_service.get_conditions(lat, lon, altitude)
        return jsonify(weather_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/flights', methods=['GET'])
def get_flights():
    """Get active flights data."""
    try:
        flights = flight_service.get_active_flights()
        return jsonify(flights), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict_contrails', methods=['POST'])
def predict_contrails():
    """Predict contrail formation for a flight path."""
    data = request.json
    flight_path = data.get('flight_path', [])
    flight_id = data.get('flight_id')
    
    try:
        # Get weather data for each point in flight path
        enriched_path = []
        for point in flight_path:
            weather = weather_service.get_conditions(
                point['lat'], point['lon'], point['altitude']
            )
            point['weather'] = weather
            enriched_path.append(point)
        
        # Run prediction model
        predictions = []
        for point in enriched_path:
            # Prepare input data for model
            input_data = prepare_model_input(point)
            
            # Run prediction
            prediction = model.predict(input_data)
            
            # Get alternative altitude recommendations
            alt_predictions = []
            for alt_adjustment in [-2000, -1000, 0, 1000, 2000]:
                alt_point = point.copy()
                alt_point['altitude'] += alt_adjustment
                alt_weather = weather_service.get_conditions(
                    alt_point['lat'], alt_point['lon'], alt_point['altitude']
                )
                alt_point['weather'] = alt_weather
                alt_input = prepare_model_input(alt_point)
                alt_pred = model.predict(alt_input)
                alt_predictions.append({
                    'altitude_adjustment': alt_adjustment,
                    'contrail_probability': float(alt_pred[0][0])
                })
            
            predictions.append({
                'position': {
                    'lat': point['lat'],
                    'lon': point['lon'],
                    'altitude': point['altitude']
                },
                'contrail_probability': float(prediction[0][0]),
                'alternatives': alt_predictions
            })
        
        # Store prediction in Firebase
        if flight_id:
            realtime_db.child('predictions').child(flight_id).set(predictions)
        
        return jsonify({
            'flight_id': flight_id,
            'predictions': predictions
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimize_altitude', methods=['POST'])
def optimize_altitude():
    """Generate optimized altitude recommendations."""
    data = request.json
    flight_id = data.get('flight_id')
    predictions = data.get('predictions', [])
    current_altitude = data.get('current_altitude')
    fuel_priority = data.get('fuel_priority', 0.5)  # 0-1 scale, higher means prioritize fuel over contrails
    
    try:
        # Get optimized altitude plan
        optimization_result = optimization_service.optimize_altitude(
            predictions, current_altitude, fuel_priority
        )
        
        # Store recommendation in Firebase
        if flight_id:
            realtime_db.child('recommendations').child(flight_id).set(optimization_result)
        
        return jsonify(optimization_result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/submit_adjustment', methods=['POST'])
def submit_adjustment():
    """Submit approved altitude adjustment to ATC systems."""
    data = request.json
    flight_id = data.get('flight_id')
    adjustment = data.get('adjustment')
    operator_id = data.get('operator_id')
    
    try:
        # Submit to ATC system
        atc_response = flight_service.submit_adjustment_to_atc(flight_id, adjustment)
        
        # Log adjustment in Firebase
        if flight_id:
            firestore_db.collection('adjustments').add({
                'flight_id': flight_id,
                'adjustment': adjustment,
                'operator_id': operator_id,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'status': atc_response.get('status'),
                'atc_response': atc_response
            })
        
        return jsonify(atc_response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def prepare_model_input(point):
    """Prepare model input from point data."""
    # Extract relevant features for model
    features = np.array([[
        point['altitude'],
        point['weather']['temperature'],
        point['weather']['humidity'],
        point['weather']['pressure'],
        point['weather']['wind_speed'],
        point['weather']['ice_supersaturation']
    ]])
    return features

if __name__ == '__main__':
    app.run(debug=True)
import pandas as pd
from prophet import Prophet

# Load dummy data
df = pd.read_csv("aircraft_demand_dataset.csv")

# Filter data for a specific part
part_id = "B777-ENG-001"
part_data = df[df["part_id"] == part_id][["ds", "y"]]

# Convert 'ds' to datetime
part_data["ds"] = pd.to_datetime(part_data["ds"])

# Train Prophet model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.05
)
model.fit(part_data)

# Generate forecasts
future = model.make_future_dataframe(periods=365)  # Forecast for 1 year
forecast = model.predict(future)

# Save forecasts to CSV
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
    f"{part_id}_forecast.csv", index=False
)
print(f"Forecasts saved to {part_id}_forecast.csv")
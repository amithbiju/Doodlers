import requests
import pandas as pd
import xarray as xr
import tempfile
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature




# Define time range and parameters
t0 = "2024-04-11T03:00:00"
t1 = "2024-04-11T06:00:00"
params = {
    "bbox": [
        -120,
        20,
        30,
        65,
    ],
    "aircraft_type": "A320",
}
times = pd.date_range(t0, t1, freq="1h")

# Google Contrail API URL and API Key
GOOGLE_URL = "https://contrails.googleapis.com/v1"
GOOGLE_API_KEY = "your_api_key"  # Replace with your actual API key

# Function to fetch and concatenate datasets
def get_ds(url, headers):
    ds_list = []

    for t in times:
        params["time"] = str(t)
        r = requests.get(url, params=params, headers=headers)
        print(f"HTTP Response Code: {r.status_code} {r.reason}")

        if r.status_code == 200:
            # Save request to disk, open with xarray, append grid to ds_list
            with tempfile.NamedTemporaryFile() as tmp, open(tmp.name, "wb") as file_obj:
                file_obj.write(r.content)
                ds = xr.load_dataset(tmp.name, engine="netcdf4", decode_timedelta=False)
            ds_list.append(ds)
        else:
            print(f"Failed to fetch data for time {t}")

    # Concatenate all grids into a single xr.Dataset
    return xr.concat(ds_list, dim="time")

# Set up headers with API key
headers = {
    "Authorization": f"Bearer {GOOGLE_API_KEY}"
}

# Fetch the dataset
ds = get_ds(f"{GOOGLE_URL}/data", headers)

# Print the dataset
print(ds)

# Plotting the data
def plot_data(ds):
    # Select a specific time and flight level for plotting
    ds_slice = ds.isel(time=0, flight_level=0)

    # Create a plot
    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)

    # Plot the data
    ds_slice['expected_effective_energy_forcing'].plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis')

    # Add gridlines and labels
    ax.gridlines(draw_labels=True)
    plt.title('Expected Effective Energy Forcing')
    plt.show()

# Plot the data
plot_data(ds)
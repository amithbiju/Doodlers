import requests
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import io

# Define time range and parameters
t0 = "2024-08-08T03:00:00Z"
t1 = "2024-08-08T06:00:00Z"
params = {
    "bbox": [-120,20,-70,65],  # Changed to string format
    "aircraft_type": "A320",
    "time": t0,  # initialize the time parameter
}
times = pd.date_range(t0, t1, freq="1h")

# Google Contrail API URL and API Key
GOOGLE_URL = "https://contrails.googleapis.com/v1/grid/ef"
GOOGLE_API_KEY = "AIzaSyBSMnTDLWdU79tlOva1QgzzV4YU0gWBAl4"  # Replace with your actual API key

# Function to fetch and concatenate datasets
def get_ds(url, params, headers):
    ds_list = []
    for t in times:
        # Update time parameter for this request
        current_params = params.copy()
        current_params["time"] = t.strftime("%Y-%m-%dT%H:%M:%SZ")  # Ensure proper format
        
        # Make the request
        r = requests.get(url, params=current_params, headers=headers)
        print(f"HTTP Response Code: {r.status_code} {r.reason}")
        print(f"Request URL: {r.url}")
        
        if r.status_code == 200:
            try:
                # Use BytesIO to create a file-like object from the response content
                ds = xr.open_dataset(io.BytesIO(r.content))
                ds_list.append(ds)
            except Exception as e:
                print(f"Error opening dataset for time {t}: {e}")
                print(f"Response content: {r.content[:200]}...")  # Print first 200 chars for debugging
        else:
            print(f"Failed to fetch data for time {t}")
            print(f"Response content: {r.content[:200]}...")  # Print first 200 chars for debugging
    
    if ds_list:
        return xr.concat(ds_list, dim="time")
    else:
        return None

# Set up headers with API key
headers = {
    'x-goog-api-key': GOOGLE_API_KEY
}

# Fetch the dataset
ds = get_ds(GOOGLE_URL, params, headers)

# Print the dataset
if ds is not None:
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
        if 'expected_effective_energy_forcing' in ds_slice:
            ds_slice['expected_effective_energy_forcing'].plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis')
            # Add gridlines and labels
            ax.gridlines(draw_labels=True)
            plt.title('Expected Effective Energy Forcing')
            plt.show()
        else:
            print("Variable 'expected_effective_energy_forcing' not found in dataset.")
            print("Available variables:", list(ds_slice.data_vars))

    # Plot the data
    plot_data(ds)
else:
    print("No data was returned from the API.")
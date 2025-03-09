import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class MapClick:
    def __init__(self):
        self.points = []
        self.fig, self.ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        self.ax.set_global()
        self.ax.add_feature(cfeature.COASTLINE)
        self.ax.add_feature(cfeature.BORDERS, linestyle=':')
        self.ax.add_feature(cfeature.RIVERS)
        self.ax.set_title("Click to select Start and Destination points")
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.points.append((event.xdata, event.ydata))
            self.ax.plot(event.xdata, event.ydata, 'ro' if len(self.points) == 1 else 'bo', 
                         markersize=8, transform=ccrs.PlateCarree())
            self.fig.canvas.draw()
            
            if len(self.points) == 2:
                plt.close()

    def get_points(self):
        plt.show()
        return self.points if len(self.points) == 2 else None


def get_bounding_box(start, dest):
    start_lon, start_lat = start
    dest_lon, dest_lat = dest
    return [round(min(start_lon, dest_lon), 6), round(min(start_lat, dest_lat), 6), 
            round(max(start_lon, dest_lon), 6), round(max(start_lat, dest_lat), 6)]


def display_map_and_get_bbox():
    map_click = MapClick()
    points = map_click.get_points()
    
    if points:
        start, dest = points
        bbox = get_bounding_box(start, dest)
        print(f"Bounding Box: {bbox}")
        return bbox
    else:
        print("No valid points selected.")
        return None

# Example usage
bbox_coords = display_map_and_get_bbox()
if bbox_coords:
    print(f"Westernmost Longitude: {bbox_coords[0]}")
    print(f"Southernmost Latitude: {bbox_coords[1]}")
    print(f"Easternmost Longitude: {bbox_coords[2]}")
    print(f"Northernmost Latitude: {bbox_coords[3]}")
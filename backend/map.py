import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.widgets import Button

class MapSelector:
    def __init__(self):
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        self.ax.stock_img()
        self.ax.coastlines()

        self.start_point = None
        self.end_point = None

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Button to confirm selection
        confirm_ax = plt.axes([0.81, 0.01, 0.1, 0.05])
        self.confirm_button = Button(confirm_ax, 'Confirm')
        self.confirm_button.on_clicked(self.on_confirm)

        plt.show()

    def on_click(self, event):
        if event.inaxes == self.ax:
            lon, lat = event.xdata, event.ydata  # ✅ Corrected: Directly using event.xdata, event.ydata
            if lon is None or lat is None:
                return  # Ignore clicks outside data range
            
            if self.start_point is None:
                self.start_point = (lon, lat)
                self.ax.plot(lon, lat, 'bo', markersize=5, label='Start')
            elif self.end_point is None:
                self.end_point = (lon, lat)
                self.ax.plot(lon, lat, 'ro', markersize=5, label='End')
            self.fig.canvas.draw()

    def on_confirm(self, event):
        if self.start_point and self.end_point:
            bbox = self.calculate_bbox(self.start_point, self.end_point)
            print(f'"bbox": [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]')  # Prints correct bbox coordinates
            plt.close(self.fig)

    def calculate_bbox(self, start, end):
        lon_min = min(start[0], end[0])
        lon_max = max(start[0], end[0])
        lat_min = min(start[1], end[1])
        lat_max = max(start[1], end[1])
        return [lon_min, lat_min, lon_max, lat_max]

# Run the map selector
if __name__ == "__main__":
    MapSelector()

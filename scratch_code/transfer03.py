import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class DraggablePoint:
    def __init__(self, ax, x, y):
        self.ax = ax
        self.point, = ax.plot(x, y, 'ro', markersize=8, picker=5)
        self.dragging = False
        self.offset = (0, 0)
        self.connect()

    def connect(self):
        self.cid_press = self.point.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.cid_release = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_pick(self, event):
        if event.artist == self.point:
            self.dragging = True
            self.offset = (self.point.get_xdata() - event.xdata, self.point.get_ydata() - event.ydata)

    def on_release(self, event):
        self.dragging = False

    def on_motion(self, event):
        if self.dragging:
            if event.xdata is not None and event.ydata is not None:
                self.point.set_xdata(event.xdata + self.offset[0])
                self.point.set_ydata(event.ydata + self.offset[1])
                self.point.figure.canvas.draw()

class PointManipulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Manipulation App")

        self.points_values = None

        self.create_widgets()

    def create_widgets(self):
        # Create Matplotlib figure and axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        # Create draggable points
        self.points = [DraggablePoint(self.ax, 0, 0.2),
                       DraggablePoint(self.ax, 0.3, 0.7),
                       DraggablePoint(self.ax, 0.6, 0.3),
                       DraggablePoint(self.ax, 0.9, 0.8)]

        # Embed Matplotlib figure in Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Apply button
        apply_button = ttk.Button(self.root, text="Apply", command=self.apply_changes)
        apply_button.pack()

    def apply_changes(self):
        # Get the values of the four points and store them in a list
        self.points_values = [(point.point.get_xdata().item(), point.point.get_ydata().item()) for point in self.points]
        self.root.destroy()  # Close the Tkinter window

    def get_points_values(self):
        return self.points_values

def main():
    root = tk.Tk()
    app = PointManipulationApp(root)
    root.mainloop()

    # Access points_values after Tkinter window is closed
    points_values = app.get_points_values()
    print("Points:", points_values)

if __name__ == "__main__":
    main()

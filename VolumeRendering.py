import numpy as np
import tkinter as tk
from tkinter import ttk
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-5000, 2600)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        self.points = [DraggablePoint(self.ax, 0, 0.2),
                       DraggablePoint(self.ax, 0.3, 0.7),
                       DraggablePoint(self.ax, 0.6, 0.3),
                       DraggablePoint(self.ax, 0.9, 0.8)]

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        apply_button = ttk.Button(self.root, text="Apply", command=self.apply_changes)
        apply_button.pack()

    def apply_changes(self):
        self.points_values = [(point.point.get_xdata().item(), point.point.get_ydata().item()) for point in self.points]
        print(self.points_values)
        self.root.destroy()

    def get_points_values(self):
        return self.points_values

def vtkToNumpy(data):
    temp = vtk_to_numpy(data.GetPointData().GetScalars())
    dims = data.GetDimensions()
    component = data.GetNumberOfScalarComponents()
    if component == 1:
        numpy_data = temp.reshape(dims[2], dims[1], dims[0])
        numpy_data = numpy_data.transpose(2,1,0)
    elif component == 3 or component == 4:
        if dims[2] == 1: # a 2D RGB image
            numpy_data = temp.reshape(dims[1], dims[0], component)
            numpy_data = numpy_data.transpose(0, 1, 2)
            numpy_data = np.flipud(numpy_data)
        else: 
            raise RuntimeError('unknown type')
    return numpy_data

def color_tf(x):
    r = 1.0*np.exp( -(x - 7.5)**2/1.0 ) +  0.8*np.exp( -(x - 2.0)**2/0.1 ) +  0.8*np.exp( -(x - -3.0)**2/0.5 )
    g = 0.10*np.exp( -(x - 9.0)**2/1.0 ) +  0.01*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 ) 
    b = 0.1*np.exp( -(x - 9.0)**2/1.0 ) +  1.0*np.exp( -(x - 3.0)**2/0.1 ) +  1.0*np.exp( -(x - -3.0)**2/0.5 )
    return r,g,b

def opacity_tf(x_values, opacitytf):
    a_values = []
    for x in x_values.flat:
        opacity = opacitytf.GetValue(x)
        a_values.append(opacity)

    return np.array(a_values).reshape(x_values.shape)

def transform_array(data, view_axis):
    if view_axis not in ('x', 'y', 'z', '-x', '-y', '-z'):
        raise ValueError("Invalid view_axis. Use 'x', 'y', or 'z'.")

    if view_axis == 'x':	# View along the x-axis
        transformed_data = np.transpose(data, (1, 2, 0))
    elif view_axis == 'y':	# View along the y-axis
        transformed_data = np.transpose(data, (0, 2, 1))
    else:	# View along the z-axis
        transformed_data = np.transpose(data, (2, 0, 1))
        transformed_data = np.flip(transformed_data, axis=0)
        transformed_data = np.flip(transformed_data, axis=1)
        transformed_data = np.rot90(transformed_data, k=1, axes=(1, 2))

    return transformed_data



def main():
    if rank == 0:
        root = tk.Tk()
        app = PointManipulationApp(root)
        root.mainloop()
        opacity_values = app.get_points_values()
        # temp = opacity_values
    else:
        opacity_values = [(-5000,1.0), (-2500,0.75), (1300, 0.25), (2600,0.0)]
        # opacity_values = [(-5000,0.0), (-2500,0.1), (1300, 0.25), (2600,1.0)]


    """ Volume Rendering """

    reader = vtk.vtkXMLImageDataReader() 
    reader.SetFileName('./datasets/Isabel_Pressure_Large.vti')  # dataset file path
    reader.Update()
    original_data = reader.GetOutput()
    original_array = vtkToNumpy(original_data)  # convert vti to numpy array
    camera_grid = transform_array(original_array, 'z')  # input orientation as x, y, z
    # print([np.max(camera_grid), np.min(camera_grid)]) #[2593.9722, -4930.2305]

    #  Split the data among n processes...
    a = camera_grid.shape
    sizee = a
    p1 = (int)(a[2] // size) * rank
    p2 = (int)(((a[2] // size) * (rank + 1)) - 1) 
    camera_grid = camera_grid[:, :, p1:p2]
    image = np.zeros((camera_grid.shape[1], camera_grid.shape[2], 3))

    #  Transfer Function...
    opacitytf = vtk.vtkPiecewiseFunction()
    opacitytf.AddPoint(opacity_values[0][0], opacity_values[0][1])
    opacitytf.AddPoint(opacity_values[1][0], opacity_values[1][1])
    opacitytf.AddPoint(opacity_values[2][0], opacity_values[2][1])
    opacitytf.AddPoint(opacity_values[3][0], opacity_values[3][1])

    #   Apply transfer function...
    for dataslice in camera_grid:
        r, g, b = color_tf(np.log(dataslice))
        a = opacity_tf(np.log(dataslice), opacitytf)
        image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
        image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
        image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]

    # comm.Gather(image, Comb, root=0)
    if rank != 0 :
        comm.Send(image, dest=0)
    else :
        total_data = image
        total_data = np.concatenate((total_data, image), 1)
        for k in range(1,size):
            comm.Recv(image, source=k)
            total_data = np.concatenate((total_data, image), 1)
        total_data = total_data[:,p2:(p2+sizee[2]),:]

    #   Display final image...
    if rank == 0 : 
        intensity = 1
        total_data = total_data * intensity
        plt.figure(figsize=(8,8), dpi=360)
        plt.imshow(total_data)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()
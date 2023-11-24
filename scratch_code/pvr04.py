import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
from mpi4py import MPI
from transfer03 import PointManipulationApp


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
    r = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
    g = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  1.0*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
    b = 0.1*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  1.0*np.exp( -(x - -3.0)**2/0.5 )
    # a = 0.6*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) + 0.01*np.exp( -(x - -3.0)**2/0.5 )
    return r,g,b

def opacity_tf(x_values, opacitytf):
    a_values = []
    for x in x_values.flat:
        opacity = opacitytf.GetValue(x)
        a_values.append(opacity)

    return np.array(a_values).reshape(x_values.shape)

def transform_array(data, view_axis):
    if view_axis not in ('x', 'y', 'z'):
        raise ValueError("Invalid view_axis. Use 'x', 'y', or 'z'.")

    if view_axis == 'x':	# View along the x-axis
        transformed_data = np.transpose(data, (1, 2, 0))
    elif view_axis == 'y':	# View along the y-axis
        transformed_data = np.transpose(data, (0, 2, 1))
    else:	# View along the z-axis
        transformed_data = np.transpose(data, (2 ,1, 0))

    return transformed_data



def main():
    root = tk.Tk()
    app = PointManipulationApp(root)
    root.mainloop()
    opacity_values = app.get_points_values()

    """ Volume Rendering """
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName('Isabel_Pressure_Large.vti')  # dataset file path
    reader.Update()
    original_data = reader.GetOutput()
    original_array = vtkToNumpy(original_data)  # convert vti to numpy array
    camera_grid = transform_array(original_array, 'z')  # input orientation as x, y, z
    # print([np.max(camera_grid), np.min(camera_grid)]) #[2593.9722, -4930.2305]

    #  Split the data among n processes..........
    a = camera_grid.shape
    sizee = a
    p1 = (int)(a[2] // size) * rank
    p2 = (int)(((a[2] // size) * (rank + 1)) - 1)     #  Saranya Pal
    camera_grid = camera_grid[:, :, p1:p2]
    image = np.zeros((camera_grid.shape[1], camera_grid.shape[2], 3))

    #  Transfer Function...
    opacitytf = vtk.vtkPiecewiseFunction()
    opacitytf.AddPoint(-5000.0, 1.0)
    opacitytf.AddPoint(2600.0, 0.0)

    #   apply transfer function...
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
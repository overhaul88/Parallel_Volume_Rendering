import numpy as np
import matplotlib.pyplot as plt
import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
from mpi4py import MPI


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
    # min_val = numpy_data.min()
    # max_val = numpy_data.max()
    # normalized_data = ((numpy_data - min_val) / (max_val - min_val) * 2 - 1)*50000
    # return normalized_data

def tF(x):
    r = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
    g = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  1.0*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
    b = 0.1*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  1.0*np.exp( -(x - -3.0)**2/0.5 )
    a = 0.6*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) + 0.01*np.exp( -(x - -3.0)**2/0.5 )
    return r,g,b,a

def tF_new(x_values):
    r_values, g_values, b_values, a_values = [], [], [], []
    # color_func = vtk.vtkColorTransferFunction()
    # color_func.SetColorSpaceToRGB()
    # color_func.SetRange(-5000, 2600)

    # color_func.AddRGBPoint(-5000, 0.01, 1.0, 0.0)
    # color_func.AddRGBPoint(-2500, 0.03, 1.0, 0.0)
    # color_func.AddRGBPoint(0, 0.0, 0.05, 0.0)
    # color_func.AddRGBPoint(1300, 0.08, 0.0, 0.0)
    # color_func.AddRGBPoint(2600, 0.1, 0.0, 0.0)

    colorTransferFunction = vtk.vtkColorTransferFunction()
    # colorTransferFunction.SetRange(-100.0,100.0)
    # colorTransferFunction.AddRGBPoint(-100.0, 0.0, 0.2, 0.0)
    # colorTransferFunction.AddRGBPoint(-50.0, 0.0, 0.0, 0.5)
    # colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
    # colorTransferFunction.AddRGBPoint(50.0, 0.5, 0.0, 0.0)
    # colorTransferFunction.AddRGBPoint(100.0, 1.0, 0.0, 0.0)

    scalar_min = -5000
    scalar_max = 2600

    # Add RGB points to the transfer function
    for x in np.linspace(scalar_min, scalar_max, 100):
        r = 1.0 * np.exp(-(x - 9.0)**2 / 1.0) + 0.1 * np.exp(-(x - 3.0)**2 / 0.1) + 0.1 * np.exp(-(x + 3.0)**2 / 0.5)
        g = 1.0 * np.exp(-(x - 9.0)**2 / 1.0) + 1.0 * np.exp(-(x - 3.0)**2 / 0.1) + 0.1 * np.exp(-(x + 3.0)**2 / 0.5)
        b = 0.1 * np.exp(-(x - 9.0)**2 / 1.0) + 0.1 * np.exp(-(x - 3.0)**2 / 0.1) + 1.0 * np.exp(-(x + 3.0)**2 / 0.5)
        a = 0.6 * np.exp(-(x - 9.0)**2 / 1.0) + 0.1 * np.exp(-(x - 3.0)**2 / 0.1) + 0.01 * np.exp(-(x + 3.0)**2 / 0.5)
        
        colorTransferFunction.AddRGBPoint(x, r, g, b)

    opacity_func = vtk.vtkPiecewiseFunction()
    opacity_func.AddPoint(-5000.0, 1.0)
    opacity_func.AddPoint(2600.0, 0.0)

    #   apply transfer function...
    for x in x_values.flat:
        col = [0, 0, 0]
        colorTransferFunction.GetColor(x, col)
        opacity = opacity_func.GetValue(x)
        r_values.append(col[0])
        g_values.append(col[1])
        b_values.append(col[2])
        a_values.append(opacity)

    return (
        np.array(r_values).reshape(x_values.shape),
        np.array(g_values).reshape(x_values.shape),
        np.array(b_values).reshape(x_values.shape),
        np.array(a_values).reshape(x_values.shape),
    )

def newtf(x_values, colortf, opacitytf):
    #   apply transfer function...
    r_values, g_values, b_values, a_values = [], [], [], []
    for x in x_values.flat:
        col = [0, 0, 0]
        colortf.GetColor(x, col)
        opacity = opacitytf.GetValue(x)
        r_values.append(col[0])
        g_values.append(col[1])
        b_values.append(col[2])
        a_values.append(opacity)

    return np.array(r_values).reshape(x_values.shape), np.array(g_values).reshape(x_values.shape), np.array(b_values).reshape(x_values.shape), np.array(a_values).reshape(x_values.shape)

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

    #  Transfer Function........
    colortf = vtk.vtkLookupTable()
    colortf.SetRange(-5000,2600)
    colortf.SetNumberOfColors(256)
    colortf.SetHueRange(0.33, 0.667)
    colortf.Addrgbpoint
    colortf.Build()
    # for x in np.linspace(-5000, 2600, 50):
    #     r = 1.0 * np.exp(-(x - 9.0)**2 / 1.0) + 0.1 * np.exp(-(x - 3.0)**2 / 0.1) + 0.1 * np.exp(-(x + 3.0)**2 / 0.5)
    #     g = 1.0 * np.exp(-(x - 9.0)**2 / 1.0) + 1.0 * np.exp(-(x - 3.0)**2 / 0.1) + 0.1 * np.exp(-(x + 3.0)**2 / 0.5)
    #     b = 0.1 * np.exp(-(x - 9.0)**2 / 1.0) + 0.1 * np.exp(-(x - 3.0)**2 / 0.1) + 1.0 * np.exp(-(x + 3.0)**2 / 0.5)
    #     a = 0.6 * np.exp(-(x - 9.0)**2 / 1.0) + 0.1 * np.exp(-(x - 3.0)**2 / 0.1) + 0.01 * np.exp(-(x + 3.0)**2 / 0.5)
    #     colortf.AddRGBPoint(x, r, g, b)

    opacitytf = vtk.vtkPiecewiseFunction()
    opacitytf.AddPoint(-5000.0, 0.8)
    opacitytf.AddPoint(2600.0, 0.2)


    for dataslice in camera_grid:
        r, g, b, a = newtf(np.log(dataslice), colortf, opacitytf)
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
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


def transferFunction(x):
	
	r = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
	g = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  1.0*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
	b = 0.1*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  1.0*np.exp( -(x - -3.0)**2/0.5 )
	a = 0.6*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) + 0.01*np.exp( -(x - -3.0)**2/0.5 )
	
	return r,g,b,a

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

    #  Split the data among n processes..........
    a = camera_grid.shape
    p1 = (int)(a[2] // size) * rank
    if size != rank+1 :
        p2 = (int)(a[2] // size) * (rank + 1)
    else :
        p2 = (int)(a[2])
    camera_grid = camera_grid[:, :, p1:p2]
    image = np.zeros((camera_grid.shape[1], camera_grid.shape[2], 3))

    #  Transfer Function........
    for dataslice in camera_grid:
        r, g, b, a = transferFunction(np.log(dataslice))
        image[:, :, 0] = a * r + (1 - a) * image[:, :, 0]
        image[:, :, 1] = a * g + (1 - a) * image[:, :, 1]
        image[:, :, 2] = a * b + (1 - a) * image[:, :, 2]

    # combined_image = np.empty((camera_grid.shape[1], camera_grid.shape[2], 3), dtype=np.float32)
    # # combined_image = None
    # if rank == 0:
    #     combined_image = np.empty((image.shape[0], image.shape[1], 3), dtype=np.float32)
    # comm.Gather(image, combined_image, root=0)
    all_images = comm.gather(image, root=0)
    # if rank == 0:
    # plt.figure(figsize=(4, 4), dpi=80)
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
    if rank == 0:
        combined_image = np.concatenate(all_images, axis=2)
        plt.figure(figsize=(4, 4), dpi=80)
        plt.imshow(combined_image)
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    main()
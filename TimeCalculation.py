import numpy as np
import matplotlib.pyplot as plt
import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
from mpi4py import MPI
import time



#   For fastest results, use either 3 or 4 or 5 processes........

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
    start_time = time.time()  # Start recording time

    """ Volume Rendering """
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName('C:/Users/91629/Desktop/c461c9be74570373fed4dc81358e566c5a0c62085c2fac8369b773e95a5d7d32_Isabel_Pressure_Largevti_/Isabel_Pressure_Large.vti')  # dataset file path
    reader.Update()
    original_data = reader.GetOutput()
    original_array = vtkToNumpy(original_data)  # convert vti to numpy array
    camera_grid = transform_array(original_array, 'z')  # input orientation as x, y, z


    #  Split the data among n processes..........
    a = camera_grid.shape
    # print(a)
    sizee = a
    p1 = (int)(a[2] // size) * rank
    p2 = (int)(((a[2] // size) * (rank + 1)) - 1)     #  Saranya Pal
    # print(p1,p2,rank)
    camera_grid = camera_grid[:, :, p1:p2]

    image = np.zeros((camera_grid.shape[1], camera_grid.shape[2], 3))


    #  Transfer Function........
    for dataslice in camera_grid:
        r, g, b, a = transferFunction(np.log(dataslice))
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
            # print(k)
            total_data = np.concatenate((total_data, image), 1)
        # print(p,sizee[2])
        total_data = total_data[:,p2:(p2+sizee[2]),:]

    if rank == 0: 
        intensity = 1
        total_data = total_data * intensity


        end_time = time.time()  # Stop recording time
        execution_time = end_time - start_time  # Calculate execution time
        print()
        print()
        print()
        print(f"The execution time required with {size} processes is: {execution_time} seconds")


        plt.figure(figsize=(4, 4), dpi=80)
        plt.imshow(total_data)
        plt.axis('off')
        plt.show()
       


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
# from scipy.interpolate import interpn
import vtkmodules.all as vtk
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.util.numpy_support import vtk_to_numpy

"""
Create Your Own Volume Rendering (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the Schrodinger-Poisson system with the Spectral method
"""

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
            raise RuntimeError('unknow type')
    return numpy_data


def transferFunction(x):
	
	r = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
	g = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  1.0*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
	b = 0.1*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  1.0*np.exp( -(x - -3.0)**2/0.5 )
	a = 0.6*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) + 0.01*np.exp( -(x - -3.0)**2/0.5 )
	
	return r,g,b,a


""" Volume Rendering """
reader = vtk.vtkXMLImageDataReader()
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName('foot.vti')  # dataset file 
reader.Update()
data = reader.GetOutput()
datacube = vtkToNumpy(data)

# Datacube Grid
Nx, Ny, Nz = datacube.shape
x = np.linspace(-Nx/2, Nx/2, Nx)
y = np.linspace(-Ny/2, Ny/2, Ny)
z = np.linspace(-Nz/2, Nz/2, Nz)
points = (x, y, z)

# Do Volume Rendering at Different Veiwing Angles
Nangles = 1
for i in range(Nangles):
	
	print('Rendering Scene ' + str(i+1) + ' of ' + str(Nangles) + '.\n')

	# Camera Grid / Query Points -- rotate camera view
	angle = np.pi/2 * i / Nangles
	N = 180
	c = np.linspace(-N/2, N/2, N)
	qx, qy, qz = np.meshgrid(c,c,c)
	qxR = qx
	qyR = qy * np.cos(angle) - qz * np.sin(angle) 
	qzR = qy * np.sin(angle) + qz * np.cos(angle)
	qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T
	
	# Interpolate onto Camera Grid
	# camera_grid = interpn(points, datacube, qi, method='linear').reshape((N,N,N))
	camera_grid = datacube
	
	# Do Volume Rendering
	image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))

	for dataslice in camera_grid:
		r,g,b,a = transferFunction(np.log(dataslice))
		image[:,:,0] = a*r + (1-a)*image[:,:,0]
		image[:,:,1] = a*g + (1-a)*image[:,:,1]
		image[:,:,2] = a*b + (1-a)*image[:,:,2]
	
	image = np.clip(image,0.0,1.0)
	
	# Plot Volume Rendering
	plt.figure(figsize=(4,4), dpi=80)
	plt.imshow(image)
	plt.axis('off')
	plt.show()
	
	# Save figure
	# plt.savefig('volumerender' + str(i) + '.png',dpi=240,  bbox_inches='tight', pad_inches = 0)



# Plot Simple Projection -- for Comparison
# plt.figure(figsize=(4,4), dpi=80)

# plt.imshow(np.log(np.mean(datacube,0)), cmap = 'viridis')
# plt.clim(-5, 5)
# plt.axis('off')

# Save figure
# plt.savefig('projection.png',dpi=240,  bbox_inches='tight', pad_inches = 0)
# plt.show()
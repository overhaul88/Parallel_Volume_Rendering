import vtkmodules.all as vtk
import numpy as np
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.util.numpy_support import vtk_to_numpy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName('foot.vti')  # dataset file 
reader.Update()

data = reader.GetOutput()
nparr = vtkToNumpy(data)
volume_data = nparr

# Define the camera parameters (orthographic projection)
width, height = 256, 256  # Image dimensions
near_plane = 0.0
far_plane = 1.0
fov = 90  # Field of view in degrees (orthographic projection)

# Define a simple transfer function (mapping scalar values to color and opacity)
# def transfer_function(scalar_value):
#     # Example: Map scalar values to grayscale color and constant opacity
    # return scalar_value, scalar_value, scalar_value, 0.5

def transfer_function(x):
  
  """Transfer Function returns r,g,b,a values as a function of density x"""
  r = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
  g = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  1.0*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
  b = 0.1*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  1.0*np.exp( -(x - -3.0)**2/0.5 )
  a = 0.6*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) + 0.01*np.exp( -(x - -3.0)**2/0.5 )
  return r,g,b,a

# Create an empty image
output_image = np.zeros((height, width, 4), dtype=np.uint8)

# Ray casting algorithm
for y in range(height):
    for x in range(width):
        # Compute the ray direction in world coordinates (orthographic projection)
        ray_dir = np.array([0, 0, 1])  # Along the positive Z-axis

        # Compute the ray origin in world coordinates (center of the near plane)
        ray_origin = np.array([x / width, y / height, near_plane])

        # Initialize color and opacity
        pixel_color = np.array([0.0, 0.0, 0.0, 0.0])

        # Perform ray traversal through the volume
        for z in np.linspace(near_plane, far_plane, volume_data.shape[2], endpoint=False):
            # Convert ray origin to voxel coordinates
            voxel_coord = ray_origin * volume_data.shape - 0.5  # Centered at (0.5, 0.5, 0.5)
            voxel_coord = np.floor(voxel_coord).astype(int)

            # Sample scalar value from the volume (no interpolation)
            scalar_value = volume_data[voxel_coord[0], voxel_coord[1], voxel_coord[2]]

            # Apply the transfer function
            color_and_opacity = transfer_function(scalar_value)

            # Composite color and opacity along the ray (simple over operator)
            # pixel_color[0:3] = pixel_color[0:3] + (1 - pixel_color[3]) * color_and_opacity[0:3] * color_and_opacity[3]
            pixel_color[0:3] = pixel_color[0:3] + (1 - pixel_color[3]) * np.array(color_and_opacity[0:3]) * color_and_opacity[3]

            pixel_color[3] = pixel_color[3] + (1 - pixel_color[3]) * color_and_opacity[3]

        # Store the computed color in the output image
        output_image[y, x] = (np.clip(pixel_color * 255, 0, 255)).astype(np.uint8)


# Display the rendered image using Matplotlib
plt.imshow(output_image[:, :, 0:3])  # Display RGB channels
plt.axis('off')
plt.show()


# Optionally, save the rendered image to a file using Pillow (PIL)
# output_pil_image = Image.fromarray(output_image[:, :, 0:3])
# output_pil_image.save('ray_casting_output.png')

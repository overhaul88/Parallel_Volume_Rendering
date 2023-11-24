import vtkmodules.all as vtk
import numpy as np
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.util.numpy_support import vtk_to_numpy

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
dataarray = data.GetPointData().GetArray(0)
# print(dataarray)

arr = vtk_to_numpy(dataarray)

print(arr.size)
# print(arr[84374999])

nparr = vtkToNumpy(data)
print(nparr.shape)
print(nparr[1][2])

# # Transfer function - iska GUI bnana hai
# opacityTransferFunction = vtkPiecewiseFunction()
# opacityTransferFunction.AddPoint(20, 0.0)
# opacityTransferFunction.AddPoint(255, 0.2)

# colorTransferFunction = vtkColorTransferFunction()
# colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
# colorTransferFunction.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
# colorTransferFunction.AddRGBPoint(128.0, 0.0, 0.0, 1.0)
# colorTransferFunction.AddRGBPoint(192.0, 0.0, 1.0, 0.0)
# colorTransferFunction.AddRGBPoint(255.0, 0.0, 0.2, 0.0)


# volume_mapper = vtk.vtkSmartVolumeMapper()
# volume_mapper.SetInputData(reader.GetOutput())


# volume_property = vtk.vtkVolumeProperty()
# volume_property.SetColor(colorTransferFunction)
# volume_property.SetScalarOpacity(opacityTransferFunction)
# volume_property.SetScalarOpacityUnitDistance(1)
# volume_property.ShadeOff()
# volume_property.SetInterpolationTypeToLinear()


# volume = vtk.vtkVolume()
# volume.SetMapper(volume_mapper)
# volume.SetProperty(volume_property)


# renderer = vtk.vtkRenderer()
# renderer.SetBackground(0.1, 0.1, 0.1)


# render_window = vtk.vtkRenderWindow()
# render_window.SetWindowName("Ray Casting Example")
# render_window.AddRenderer(renderer)


# render_window_interactor = vtk.vtkRenderWindowInteractor()
# render_window_interactor.SetRenderWindow(render_window)


# renderer.AddVolume(volume)
# renderer.ResetCamera()
# render_window.Render()
# render_window_interactor.Start()
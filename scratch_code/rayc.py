import vtkmodules.all as vtk

# Step 1: Create a VTK reader and read the dataset
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(r'C:\Users\adeel\Downloads\datavtk\Isabel_Pressure_Large.vti')  # Replace with your dataset file
reader.Update()

# Step 2: Create a volume mapper and actor
volume_mapper = vtk.vtkSmartVolumeMapper()
volume_mapper.SetInputData(reader.GetOutput())

volume_property = vtk.vtkVolumeProperty()
volume_property.SetScalarOpacityUnitDistance(1)
volume_property.ShadeOff()
volume_property.SetInterpolationTypeToLinear()

volume = vtk.vtkVolume()
volume.SetMapper(volume_mapper)
volume.SetProperty(volume_property)

# Step 3: Set up the renderer and render window
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.1, 0.1, 0.1)

render_window = vtk.vtkRenderWindow()
render_window.SetWindowName("Ray Casting Example")
render_window.AddRenderer(renderer)

# Step 4: Create a render window interactor
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# Step 5: Add the volume to the renderer and start rendering
renderer.AddVolume(volume)
renderer.ResetCamera()

# Customize ray casting properties if needed (e.g., transfer functions)

# Start rendering
render_window.Render()
render_window_interactor.Start()

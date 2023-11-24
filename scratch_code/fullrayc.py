import sys
import vtkmodules.all as vtk
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider, QLabel, QPushButton
from PyQt5.QtCore import Qt

class TransferFunctionGUI(QMainWindow):
    def __init__(self, volume_mapper):
        super().__init__()
        self.volume_mapper = volume_mapper
        self.initUI()

    def initUI(self):
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.volume_mapper)
        # Create the main window
        self.setGeometry(100, 100, 400, 200)
        self.setWindowTitle("Transfer Function GUI")

        # Create a central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create layout for the central widget
        layout = QVBoxLayout()

        # Create sliders for color and opacity
        self.color_slider = QSlider(Qt.Horizontal)
        self.color_slider.setRange(0, 255)
        self.color_slider.setValue(128)
        self.color_slider.valueChanged.connect(self.updateColorTransferFunction)
        layout.addWidget(QLabel("Color"))
        layout.addWidget(self.color_slider)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 255)
        self.opacity_slider.setValue(128)
        self.opacity_slider.valueChanged.connect(self.updateOpacityTransferFunction)
        layout.addWidget(QLabel("Opacity"))
        layout.addWidget(self.opacity_slider)

        # Create a button to apply changes
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.applyChanges)
        layout.addWidget(apply_button)

        central_widget.setLayout(layout)

    def updateColorTransferFunction(self):
        self.color_value = self.color_slider.value() / 255.0
        # Modify the color transfer function based on the slider value

    def updateOpacityTransferFunction(self):
        self.opacity_value = self.opacity_slider.value() / 255.0
        # Modify the opacity transfer function based on the slider value

    def applyChanges(self):
        print(self.opacity_value)
        print(self.color_value)
        # No need to apply changes in this integration; changes are applied immediately in updateColorTransferFunction and updateOpacityTransferFunction.
        ctf = self.volume.GetProperty().GetRGBTransferFunction()
        ctf.RemoveAllPoints()
        ctf.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        ctf.AddRGBPoint(1.0, self.color_value, self.color_value, self.color_value)
        self.volume_mapper.Update()

        otf = self.volume.GetProperty().GetScalarOpacity()
        otf.RemoveAllPoints()
        otf.AddPoint(0, 0.0)
        otf.AddPoint(255, self.opacity_value)
        self.volume_mapper.Update()



def main():
    # Create the VTK volume rendering components
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(r'C:\Users\adeel\Downloads\datavtk\Isabel_Pressure_Large.vti')  # Replace with your dataset file
    reader.Update()

    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(reader.GetOutput())

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetScalarOpacityUnitDistance(1)
    volume_property.ShadeOff()
    volume_property.SetInterpolationTypeToLinear()

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.1, 0.1)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Create the transfer function GUI and pass the volume mapper
    app = QApplication(sys.argv)
    window = TransferFunctionGUI(volume_mapper)
    window.show()

    # Add the volume to the renderer
    renderer.AddVolume(volume)
    renderer.ResetCamera()

    render_window.Render()
    render_window_interactor.Start()

if __name__ == '__main__':
    main()

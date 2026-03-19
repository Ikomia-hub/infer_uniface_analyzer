"""
Module that implements the UI widget of the algorithm.
"""
# PyQt GUI framework
from PyQt6.QtWidgets import *

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion

from infer_uniface_analyzer.infer_uniface_analyzer_process import InferUnifaceAnalyzerParam


class InferUnifaceAnalyzerWidget(core.CWorkflowTaskWidget):
    """
    Class that implements UI widget to adjust algorithm parameters.
    Inherits PyCore.CWorkflowTaskWidget from Ikomia API.
    """

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferUnifaceAnalyzerParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Detector selection
        self.combo_detector = pyqtutils.append_combo(self.grid_layout, "Detector")
        self.combo_detector.addItem("retinaface")
        self.combo_detector.addItem("yolov5face")
        self.combo_detector.addItem("scrfd")
        self.combo_detector.addItem("yolov8face")
        self.combo_detector.setCurrentText(self.parameters.detector_name)

        # Confidence threshold
        self.spin_conf_thres = pyqtutils.append_double_spin(
            self.grid_layout,
            "Confidence threshold",
            self.parameters.conf_thres,
            min=0.0,
            max=1.0,
            step=0.05,
            decimals=2
        )

        # NMS threshold
        self.spin_nms_thres = pyqtutils.append_double_spin(
            self.grid_layout,
            "NMS threshold",
            self.parameters.nms_thres,
            min=0.0,
            max=1.0,
            step=0.05,
            decimals=2
        )

        # Enable emotion detection checkbox
        self.check_enable_emotion = pyqtutils.append_check(
            self.grid_layout,
            "Enable emotion detection",
            self.parameters.enable_emotion
        )
        self.check_enable_emotion.stateChanged.connect(self.on_emotion_changed)

        # Emotion model selection
        self.combo_emotion = pyqtutils.append_combo(self.grid_layout, "Emotion model")
        self.combo_emotion.addItem("affecnet7")
        self.combo_emotion.addItem("affecnet8")
        self.combo_emotion.setCurrentText(self.parameters.emotion_model)
        self.combo_emotion.setEnabled(self.parameters.enable_emotion)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_emotion_changed(self, state):
        """Enable/disable emotion model combo based on checkbox state."""
        self.combo_emotion.setEnabled(state == 2)  # Qt.Checked = 2

    def on_apply(self):
        """QT slot called when users click the Apply button."""
        # Get parameters from widget
        self.parameters.detector_name = self.combo_detector.currentText()
        self.parameters.conf_thres = self.spin_conf_thres.value()
        self.parameters.nms_thres = self.spin_nms_thres.value()
        self.parameters.enable_emotion = self.check_enable_emotion.isChecked()
        self.parameters.emotion_model = self.combo_emotion.currentText()
        self.parameters.update = True

        # Send signal to launch the algorithm main function
        self.emit_apply(self.parameters)


class InferUnifaceAnalyzerWidgetFactory(dataprocess.CWidgetFactory):
    """
    Factory class to create algorithm widget object.
    Inherits PyDataProcess.CWidgetFactory from Ikomia API.
    """

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_uniface_analyzer"

    def create(self, param):
        """Instantiate widget object."""
        return InferUnifaceAnalyzerWidget(param, None)

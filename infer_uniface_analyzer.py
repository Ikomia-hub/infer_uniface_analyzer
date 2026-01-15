"""
Main Ikomia plugin module.
Ikomia Studio and Ikomia API use it to load algorithms dynamically.
"""
from ikomia import dataprocess
from infer_uniface_analyzer.infer_uniface_analyzer_process import InferUnifaceAnalyzerFactory
from infer_uniface_analyzer.infer_uniface_analyzer_process import InferUnifaceAnalyzerParamFactory


class IkomiaPlugin(dataprocess.CPluginProcessInterface):
    """
    Interface class to integrate the process with Ikomia application.
    Inherits PyDataProcess.CPluginProcessInterface from Ikomia API.
    """
    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        """Instantiate process object."""
        return InferUnifaceAnalyzerFactory()

    def get_widget_factory(self):
        """Instantiate associated widget object."""
        from infer_uniface_analyzer.infer_uniface_analyzer_widget import InferUnifaceAnalyzerWidgetFactory
        return InferUnifaceAnalyzerWidgetFactory()

    def get_param_factory(self):
        """Instantiate algorithm parameters object."""
        return InferUnifaceAnalyzerParamFactory()

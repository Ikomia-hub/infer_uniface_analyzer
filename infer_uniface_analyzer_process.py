"""
Module that implements the core logic of algorithm execution.
"""
import copy
import os

from ikomia import core, dataprocess
from ikomia.dataprocess.io.datadictIO import DataDictIO

from .models.model_loader import create_detector, create_age_gender, create_emotion


class InferUnifaceAnalyzerParam(core.CWorkflowTaskParam):
    """
    Class to handle the algorithm parameters.
    Inherits PyCore.CWorkflowTaskParam from Ikomia API.
    """

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Options for detector: "retinaface", "yolov5face", "scrfd", "yolov8face"
        self.detector_name = "retinaface"
        self.conf_thres = 0.5
        self.nms_thres = 0.4
        # Enable/disable emotion detection (requires PyTorch)
        self.enable_emotion = False
        # Options for emotion: "affecnet7" (7 emotions), "affecnet8" (8 emotions with Contempt)
        self.emotion_model = "affecnet7"
        self.update = False

    def set_values(self, params):
        """
        Set parameters values from Ikomia Studio or API.
        Parameters values are stored as string and accessible like a python dict.
        """
        self.detector_name = params.get("detector_name", "retinaface")
        self.conf_thres = float(params.get("conf_thres", 0.5))
        self.nms_thres = float(params.get("nms_thres", 0.4))
        self.enable_emotion = params.get("enable_emotion", "False") == "True"
        self.emotion_model = params.get("emotion_model", "affecnet7")
        self.update = True

    def get_values(self):
        """
        Send parameters values to Ikomia Studio or API.
        Create the specific dict structure (key-value as string).
        """
        params = {
            "detector_name": str(self.detector_name),
            "conf_thres": str(self.conf_thres),
            "nms_thres": str(self.nms_thres),
            "enable_emotion": str(self.enable_emotion),
            "emotion_model": str(self.emotion_model)
        }
        return params


class InferUnifaceAnalyzerParamFactory(dataprocess.CTaskParamFactory):
    """Factory class to create parameters object."""

    def __init__(self):
        dataprocess.CTaskParamFactory.__init__(self)
        self.name = "infer_uniface_analyzer"

    def create(self):
        """Instantiate parameters object."""
        return InferUnifaceAnalyzerParam()


class InferUnifaceAnalyzer(dataprocess.CObjectDetectionTask):
    """
    Class that implements the face analysis algorithm.
    Inherits from CObjectDetectionTask for face detection output.
    """

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)

        # Add dictionary output for age and gender results
        self.add_output(DataDictIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferUnifaceAnalyzerParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.names = ["face"]
        self.model_folder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "weights")
        self.detector = None
        self.age_gender = None
        self.emotion = None

    def _load_models(self):
        """Load the detector and age_gender predictor models."""
        param = self.get_param_object()

        # Create detector with model weights saved in self.model_folder
        self.detector = create_detector(
            model_type=param.detector_name,
            model_folder=self.model_folder,
            confidence_threshold=param.conf_thres,
            nms_threshold=param.nms_thres,
        )

        # Create age_gender predictor
        self.age_gender = create_age_gender(
            model_folder=self.model_folder,
        )

        # Create emotion predictor if enabled
        if param.enable_emotion:
            from uniface.constants import DDAMFNWeights
            emotion_weight = DDAMFNWeights.AFFECNET7 if param.emotion_model.lower(
            ) == "affecnet7" else DDAMFNWeights.AFFECNET8
            self.emotion = create_emotion(
                model_folder=self.model_folder,
                model_weights=emotion_weight
            )
            if self.emotion is None:
                print("Warning: PyTorch not available, emotion detection disabled")
        else:
            self.emotion = None

    def init_long_process(self):
        """Initialize long-running process."""
        self._load_models()
        super().init_long_process()

    def get_progress_steps(self):
        """
        Ikomia Studio only.
        Function returning the number of progress steps for this algorithm.
        This is handled by the main progress bar of Ikomia Studio.
        """
        return 1

    def run(self):
        """Main function and entry point for algorithm execution."""
        # Call begin_task_run() for initialization
        self.begin_task_run()
        param = self.get_param_object()

        # Get input
        img_input = self.get_input(0)
        src_image = img_input.get_image()

        # Convert RGBA to RGB if necessary
        if src_image.shape[-1] == 4:
            src_image = src_image[:, :, :3]

        # Load models if needed
        if self.detector is None or self.age_gender is None or param.update:
            self._load_models()
            param.update = False

        # Detect faces
        faces = self.detector.detect(src_image)

        # Get dictionary output for age and gender
        dict_output = self.get_output(2)

        # Initialize result dictionary
        result_dict = {}

        if faces is None or len(faces) == 0:
            # No faces detected
            result_dict["face_count"] = "0"
            dict_output.data = result_dict
            self.end_task_run()
            return

        # Set class names for object detection output
        self.set_names(self.names)

        # Process each detected face
        for i, face in enumerate(faces):
            # Extract bounding box coordinates
            bbox = face.bbox
            confidence = face.confidence

            x1, y1, x2, y2 = bbox
            w = float(x2 - x1)
            h = float(y2 - y1)

            # Add object detection output: (object_id, class_id, confidence, x, y, width, height)
            self.add_object(i+1, 0, float(confidence),
                            float(x1), float(y1), w, h)

            # Predict age and gender for this face
            try:
                age_gender_result = self.age_gender.predict(src_image, bbox)

                # Store results in dictionary for each face
                face_key = f"face_{i+1}"
                result_dict[f"{face_key}_age"] = str(age_gender_result.age)
                # "Male" or "Female"
                result_dict[f"{face_key}_gender"] = age_gender_result.sex
                result_dict[f"{face_key}_gender_id"] = str(
                    age_gender_result.gender)  # 0 or 1
                result_dict[f"{face_key}_confidence"] = str(confidence)

                # Predict emotion if enabled
                if self.emotion is not None and face.landmarks is not None:
                    try:
                        emotion_result = self.emotion.predict(
                            src_image, face.landmarks)
                        result_dict[f"{face_key}_emotion"] = emotion_result.emotion
                        result_dict[f"{face_key}_emotion_confidence"] = str(
                            emotion_result.confidence)
                    except Exception as e:
                        print(f"Error predicting emotion for face {i+1}: {e}")
                        result_dict[f"{face_key}_emotion_error"] = str(e)

            except Exception as e:
                print(f"Error predicting age/gender for face {i+1}: {e}")
                result_dict[f"face_{i+1}_error"] = str(e)

        # Add face count to dictionary
        result_dict["face_count"] = str(len(faces))

        print(result_dict)
        # Set the dictionary output
        dict_output.data = result_dict

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


class InferUnifaceAnalyzerFactory(dataprocess.CTaskFactory):
    """
    Factory class to create process object.
    Inherits PyDataProcess.CTaskFactory from Ikomia API.
    """

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_uniface_analyzer"
        self.info.short_description = "Face analysis with detection, age, gender, and emotion prediction using UniFace"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Yakhyokhuja Valikhujaev"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2024
        self.info.license = "MIT License"

        # Ikomia API compatibility
        self.info.min_ikomia_version = "0.16.0"

        # URL of documentation
        self.info.documentation_link = "https://yakhyo.github.io/uniface/"

        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_uniface_analyzer"
        self.info.original_repository = "https://github.com/yakhyo/uniface"

        # Keywords used for search
        self.info.keywords = "uniface, face, detection, age, gender, emotion, analysis, attribute"

        # General type: INFER, TRAIN, DATASET or OTHER
        self.info.algo_type = core.AlgoType.INFER

        # Algorithms tasks
        self.info.algo_tasks = "OBJECT_DETECTION"

        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 6

    def create(self, param=None):
        """Instantiate algorithm object."""
        return InferUnifaceAnalyzer(self.info.name, param)

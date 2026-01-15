<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_uniface_analyzer</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_uniface_analyzer">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_uniface_analyzer">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_uniface_analyzer/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_uniface_analyzer.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Comprehensive face analysis combining detection, age prediction, gender prediction, and optional emotion recognition using UniFace library. UniFace is a lightweight production-ready face analysis library built on ONNX Runtime, supporting multiple models for each task.

This algorithm detects all faces in an image and provides detailed analysis including:
- **Face Detection**: Bounding boxes with confidence scores
- **Age Prediction**: Estimated age in years
- **Gender Classification**: Male/Female classification
- **Emotion Recognition** (optional): 7 or 8 emotion categories (requires PyTorch)

![Example image](https://raw.githubusercontent.com/Ikomia-hub/infer_uniface_analyzer/main/images/output.jpg)

`Output` : {'face_1_age': '31', 'face_1_gender': 'Male', 'face_1_gender_id': '1', 'face_1_confidence': '0.9999094009399414', 'face_count': '1'}

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add face analyzer algorithm
analyzer = wf.add_task(name="infer_uniface_analyzer", auto_connect=True)

# Run the workflow on image
wf.run_on(url="https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_portrait_5.jpg?raw=true")

# Display result with bounding boxes
display(analyzer.get_image_with_graphics())

# Get analysis results
dict_output = analyzer.get_output(2)
results = dict_output.data

# Print details for each face
for i in range(1, int(results.get('face_count', 0)) + 1):
    print(f"\nFace {i}:")
    print(f"  Age: {results.get(f'face_{i}_age')} years")
    print(f"  Gender: {results.get(f'face_{i}_gender')}")
    print(f"  Confidence: {results.get(f'face_{i}_confidence')}")
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add face analyzer algorithm
analyzer = wf.add_task(name="infer_uniface_analyzer", auto_connect=True)

analyzer.set_parameters({
    "detector_name": "retinaface",
    "conf_thres": "0.5",
    "nms_thres": "0.4",
    "enable_emotion": "False",
    "emotion_model": "affecnet7"
})

# Run the workflow on image
wf.run_on(url="https://github.com/yakhyo/uniface/blob/main/assets/scientists.png?raw=true")

# Display result with bounding boxes
display(analyzer.get_image_with_graphics())

# Get analysis results
dict_output = analyzer.get_output(2)
results = dict_output.data
print(f"Number of faces detected: {results.get('face_count')}")
```

### Parameters

- **detector_name** (str, default="retinaface"): Face detection model to use. Options: "retinaface", "yolov5face", "scrfd", "yolov8face".
- **conf_thres** (float, default="0.5"): Object detection confidence threshold.
- **nms_thres** (float, default="0.4"): Non-maximum suppression threshold.
- **enable_emotion** (bool, default="False"): Enable emotion detection (requires PyTorch).
- **emotion_model** (str, default="affecnet7"): Emotion recognition model. Options: "affecnet7" (7 emotions), "affecnet8" (8 emotions including Contempt).

***Note***: parameter key and value should be in **string format** when added to the dictionary.

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add face analyzer algorithm
analyzer = wf.add_task(name="infer_uniface_analyzer", auto_connect=True)

# Run the workflow on image
wf.run_on(url="https://github.com/yakhyo/uniface/blob/main/assets/scientists.png?raw=true")

# Iterate over outputs
for output in analyzer.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

The algorithm produces two outputs:

1. **Object Detection Output** (index 0): Contains face bounding boxes with confidence scores
2. **Dictionary Output** (index 2): Contains detailed analysis for each detected face:
   - `face_count`: Total number of faces detected
   - `face_N_age`: Predicted age in years for face N
   - `face_N_gender`: Gender as text ("Male" or "Female") for face N
   - `face_N_gender_id`: Gender as ID (0=Female, 1=Male) for face N
   - `face_N_confidence`: Detection confidence score for face N
   - `face_N_emotion`: Predicted emotion (if emotion detection is enabled)
   - `face_N_emotion_confidence`: Emotion prediction confidence (if emotion detection is enabled)

## :computer: How it works

The algorithm works in these steps:

1. **Face Detection**: Detects all faces in the image using the selected detector model (RetinaFace, YOLOv5Face, SCRFD, or YOLOv8Face). Each detection includes bounding box coordinates, confidence score, and facial landmarks.

2. **Age & Gender Prediction**: For each detected face, the algorithm predicts:
   - Age in years (approximate, trained on CelebA dataset)
   - Gender classification (Male/Female)

3. **Emotion Recognition** (Optional): If enabled, predicts emotional state for each face using DDAMFN models trained on AffectNet dataset:
   - **affecnet7**: Neutral, Happy, Sad, Surprise, Fear, Disgust, Angry
   - **affecnet8**: Same as above + Contempt

4. **Output Generation**: Combines all analysis results into:
   - Visual output: Image with face bounding boxes
   - Data output: Dictionary with detailed attributes for each face

### Model Information

**Detectors** (ONNX, no special requirements):
- RetinaFace: Balanced accuracy and speed
- YOLOv5Face: Fast inference
- SCRFD: Lightweight and efficient
- YOLOv8Face: Latest YOLO architecture

**Age/Gender Model** (ONNX):
- Based on InsightFace models
- Trained on CelebA dataset
- Uses aligned face crops for prediction

**Emotion Models** (TorchScript, requires PyTorch):
- DDAMFN architecture
- Trained on AffectNet dataset
- Requires facial landmarks for alignment

***Note***: Emotion detection requires PyTorch to be installed. If PyTorch is not available, emotion detection will be automatically disabled with a warning message.

## :fast_forward: Advanced usage 

### Enabling Emotion Detection

To use emotion detection, you need to install PyTorch:

```sh
pip install torch
```

Then enable it in your workflow:

```python
analyzer.set_parameters({
    "enable_emotion": "True",
    "emotion_model": "affecnet7"  # or "affecnet8"
})
```

### Processing Multiple Images

```python
from ikomia.dataprocess.workflow import Workflow
import cv2

# Init your workflow
wf = Workflow()
analyzer = wf.add_task(name="infer_uniface_analyzer", auto_connect=True)

# Process multiple images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]

for image_path in image_paths:
    wf.run_on(path=image_path)
    
    # Get results
    dict_output = analyzer.get_output(2)
    results = dict_output.data
    
    print(f"\n{image_path}: {results.get('face_count')} faces detected")
    for i in range(1, int(results.get('face_count', 0)) + 1):
        print(f"  Face {i}: {results.get(f'face_{i}_gender')}, {results.get(f'face_{i}_age')}y")
```

### Accessing Face Bounding Boxes

```python
# Get object detection output
obj_output = analyzer.get_output(0)

# Access face bounding boxes
for obj in obj_output.get_objects():
    bbox = obj.box  # [x, y, width, height]
    confidence = obj.confidence
    print(f"Face at {bbox} with confidence {confidence}")
```

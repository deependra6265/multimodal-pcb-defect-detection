# PCB Defect Detection Using U-Net

## Project Overview

This project is a deep learning-based PCB defect detection system developed using a U-Net segmentation model.

The system takes a PCB image as input and automatically identifies potential defect regions using semantic segmentation.

## Features

* PCB image upload
* 256×256 patch-based processing
* U-Net based defect segmentation
* Full PCB probability-map reconstruction
* Defect area calculation
* Connected-component analysis
* Defect grouping
* Bounding-box visualization
* LOW / MEDIUM / HIGH severity classification
* CSV defect report generation
* Streamlit web application

## Model

**Architecture:** U-Net
**Input Patch Size:** 256 × 256 × 3
**Detection Threshold:** 0.45

## Sample Result

For the tested PCB image:

* Total image pixels: 1,502,280
* Defect pixels: 30,811
* Defect area: 2.05%
* Raw components: 101
* Filtered components: 44
* Merged defect groups: 24
* High severity: 0
* Medium severity: 3
* Low severity: 21

## Technology Stack

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* SciPy
* Pillow
* Matplotlib
* Streamlit

## Application Workflow

```text
PCB Image
    ↓
Patch Extraction
    ↓
U-Net Prediction
    ↓
Probability Map
    ↓
Thresholding
    ↓
Connected Components
    ↓
Defect Grouping
    ↓
Severity Assessment
    ↓
Visualization + CSV Report
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser.

## Project Structure

```text
PCB_Defect_Detection/
│
├── app.py
├── improved_unet.keras
├── requirements.txt
└── README.md
```

## Future Improvements

* Improve segmentation accuracy
* Add more PCB defect classes
* Add model confidence visualization
* Deploy the application online
* Add automated inspection reports

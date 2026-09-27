# Multimodal Traffic Management System

This project is a multimodal traffic management system that combines Computer Vision and Natural Language Processing (NLP) to make informed traffic management decisions.

## Overview
The system performs the following tasks:
1. **Computer Vision**: Estimates traffic density from a sequence of images/frames.
2. **NLP**: Classifies traffic incidents and determines their severity from text data (e.g., Twitter dataset).
3. **Fusion & Decision**: Fuses the density score and incident severity to output a traffic management decision.
4. **Visualization**: Overlays the density, incident type, and decision on the video frames in real-time.

## Prerequisites

- Python 3.8 or higher.
- Datasets for vision (image sequence) and NLP (CSV dataset).

## Installation

1. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Before running the application, you **must** configure the dataset paths.

1. Open `app.py` in your code editor.
2. Locate the `# ===================== CONFIG =====================` section at the top of the file.
3. Update the `VISION_SEQUENCE_PATH` and `NLP_DATASET_PATH` variables to point to your local dataset directories.
   
   ```python
   # Example:
   VISION_SEQUENCE_PATH = r"C:/path/to/your/vision/images"
   NLP_DATASET_PATH     = r"C:/path/to/your/nlp/TWITTER DATA SET.csv"
   ```

You can also adjust `MAX_FRAMES` to limit how many frames are processed and `TEXT_SAMPLE_INDEX` to test different rows of the NLP dataset.

## Running the Application

Execute the main application script:

```bash
python app.py
```

### Controls:
- The system will display a window showing the video frames with the overlayed traffic data and decisions.
- Press the **ESC** key to stop the application and close the window.

## Project Structure

- `app.py`: The main entry point of the application.
- `requirements.txt`: Python package dependencies.
- `vision/`: Contains computer vision modules (density estimation, image reading).
- `nlp/`: Contains NLP modules (incident classification, severity mapping).
- `fusion/`: Contains the decision engine that combines vision and NLP outputs.
- `analysis/`, `data/`, `decision/`, `sim/`: Supporting modules for system operation.

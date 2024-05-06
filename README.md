# Video Processing Application

This Video Processing Application utilizes OpenCV for handling video files and integrates the YOLOv5 model for object detection and the DeepSort algorithm for object tracking. The application provides functionalities such as tracking objects across video frames, displaying trajectories, and visualizing motion and density of tracked objects with a UI implemented.

## Features

- Multi-people tracking with optimized visualization:
  - YOLOv5 for detection
  - Multi-people tracking achieved using DeepSort
  - Trajectories visualization and optimization by moving-average
- Movement detection and real-time warning:
  - Movement detection using moving-average
  - Movement visualization
  - Real-time people significant motion warning
- Density estimation and visualization:
  - Density estimation using kernel density estimation (KDE)
  - Density heatmap visualization
- UI achievements


## Prerequisites

Before you can run this application, you need to have the following installed:
- Python 3.8 or higher
- OpenCV
- PyTorch
- NumPy
- SciPy
- Tkinter for the GUI

## Installation

1. Clone this repository to your local machine.
2. Install the required packages.

## Configuration

Modify the `deep_sort/configs/deep_sort.yaml` file according to your specific requirements for object tracking parameters such as `max_dist`, `min_confidence`, etc.

## Usage

To start the application, run the following command in your terminal:
```
python main.py
```

Follow the GUI prompts to select an input video file and an output file location. Check the option for density visualization if needed, and click "Start Processing" to begin the video analysis.

## Screenshots
![截屏2024-05-06 12.05.44.png](..%2F%E6%88%AA%E5%B1%8F2024-05-06%2012.05.44.png)
![截屏2024-05-06 18.22.30.png](..%2F..%2F..%2F..%2Fvar%2Ffolders%2Flj%2F43b65nkj7g16wy2_v549q56m0000gn%2FT%2FTemporaryItems%2FNSIRD_screencaptureui_Fx5b0c%2F%E6%88%AA%E5%B1%8F2024-05-06%2018.22.30.png)
![截屏2024-05-06 18.22.57.png](..%2F..%2F..%2F..%2Fvar%2Ffolders%2Flj%2F43b65nkj7g16wy2_v549q56m0000gn%2FT%2FTemporaryItems%2FNSIRD_screencaptureui_e4nNyM%2F%E6%88%AA%E5%B1%8F2024-05-06%2018.22.57.png)

## Acknowledgments

- YOLOv5 model provided by Ultralytics
- DeepSort algorithm for robust tracking

# Infrared Hand Image Research Project

Welcome to the **Infrared Hand Image Research Project** repository. This project explores novel techniques for enhancing hand landmark detection in infrared images, focusing on tasks like colorization, evaluation, and visualization. By leveraging advanced pipelines and metrics, the project aims to address challenges in computer vision applications for medical and diagnostic use cases.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Features](#features)
4. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Usage](#usage)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Pipeline Overview](#pipeline-overview)


---

## Project Overview

This research investigates how to improve landmark detection in infrared hand images using techniques like:

- **Colorization**: Converting grayscale infrared images to resemble real hand images.
- **Evaluation**: Measuring performance through metrics like Percentage of Correct Keypoints (PCK).
- **Visualization**: Overlaying landmarks and results for better interpretability.

---

## Repository Structure

```
├── src
│   ├── evaluation
│   │   ├── LandmarkMerger.py
│   │   ├── PCKCalculator.py
│   │   └── Recognizer.py
│   ├── colorization
│   │   └── [Richard Zhang's colorization repository files]
│   ├── annotations
│   │   └── adding_landmarks.py
│   ├── visualisations
│   │   └── drawing_landmarks.py
│   ├── pipelines
│   │   └── PipelineManager.py
├── requirements.txt
└── main.py
```

### Key Components

- **`evaluation/`**: Contains modules to compute metrics and analyze results.
  - `LandmarkMerger.py`: Combines annotated and predicted landmarks for comparison.
  - `PCKCalculator.py`: Calculates the PCK metric at various thresholds.
  - `Recognizer.py`: Aids in landmark detection evaluations.

- **`colorization/`**: Adapted from [Richard Zhang's colorization model](https://richzhang.github.io/).

- **`annotations/`**: Tools for annotating infrared hand images.
  - `adding_landmarks.py`: A script for adding landmark annotations manually.

- **`visualisations/`**: Visualization tools for interpreting results.
  - `drawing_landmarks.py`: Visualizes landmarks on hand images.

- **`pipelines/`**: Manages the flow of transformations and evaluations.
  - `PipelineManager.py`: Centralizes pipeline management.

- **`main.py`**: Runs evaluations, calculates upper and lower bounds, and scores the evaluation dataset.

---

## Features

- **Robust Evaluation Framework**:
  - Calculate upper and lower bounds for landmark detection accuracy.
  - Score evaluation datasets with precise metrics.

- **Colorization Module**:
  - Enhance infrared images to mimic real-hand images for improved detection.

- **Visualization Tools**:
  - Clearly display landmarks on images for analysis and debugging.

- **Custom Pipelines**:
  - Flexible pipeline manager for seamless processing.

---

## Getting Started

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/username/repository.git
   cd repository
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run the main script to perform evaluations and generate results:

```bash
python main.py
```

---

## Evaluation Metrics

This project evaluates landmark detection using:

1. **Upper Bound**:
   - Accuracy of MediaPipe predictions on RGB images (ground truth: manually annotated landmarks).

2. **Lower Bound**:
   - Accuracy of MediaPipe predictions on unprocessed infrared images.

3. **PCK (Percentage of Correct Keypoints)**:
   - Thresholds: 0.025, 0.05, 0.07 (relative to image dimensions).

---

## Pipeline Overview

Pipelines in this project allow for:

- Infrared image transformations (e.g., colorization, CLAHE).
- Landmark annotation and merging.
- Comprehensive evaluation with metrics.

The `PipelineManager` class ensures flexibility and modularity for experimentation.

---



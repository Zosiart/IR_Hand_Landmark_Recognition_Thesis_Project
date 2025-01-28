# Infrared Hand Image Research Project

Welcome to the **Infrared Hand Image Research Project** repository. This project explores novel techniques for enhancing hand landmark detection in infrared images, focusing on tasks like colorization, evaluation, and visualization. By leveraging advanced pipelines and metrics, the project aims to address challenges in computer vision applications for medical and diagnostic use cases.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Usage](#usage)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Pipeline Overview](#pipeline-overview)
6. [Affiliations](#affiliations)


---

## Project Overview

This research investigates how to improve landmark detection in infrared hand images using:

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

- **`colorization/`**: Adapted from [Richard Zhang's colorization model](https://github.com/richzhang/colorization).

- **`annotations/`**: Tools for annotating infrared hand images.
  - `adding_landmarks.py`: A script for adding landmark annotations manually.

- **`visualisations/`**: Visualization tools for interpreting results.
  - `drawing_landmarks.py`: Visualizes landmarks on hand images.

- **`pipelines/`**: Manages the flow of transformations and evaluations.
  - `PipelineManager.py`: Centralizes pipeline management.

- **`main.py`**: Runs evaluations, calculates upper and lower bounds, and scores the evaluation dataset.

---

## Getting Started

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Zosiart/IR_Hand_Landmark_Recognition_Thesis_Project.git
   cd repository
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run the main script to perform evaluations and generate results

---

## Evaluation Metrics

This project evaluates landmark detection using:

1. **Upper Bound**:
   - Accuracy of MediaPipe predictions on RGB images (ground truth: manually annotated landmarks).

2. **Lower Bound**:
   - Accuracy of MediaPipe predictions on unprocessed infrared images.

3. **PCK (Percentage of Correct Keypoints)**:
   - Thresholds: 0.01 - 0.15.
   - Acceptable distance computed as: `threshold * length of the hand`.

---

## Pipeline Overview

Pipelines in this project allow for:

- Infrared image transformations (e.g., colorization, CLAHE).
- Landmark annotation and merging.
- Comprehensive evaluation with metrics.

The `PipelineManager` class ensures flexibility and modularity for experimentation.

---

## Affiliations

This repository is part of the **Research Project 2025** at **TU Delft**.


---



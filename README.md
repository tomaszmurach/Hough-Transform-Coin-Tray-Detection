# Coin and Tray Detection with Hough Transform

A computer vision project in Python and OpenCV for detecting a tray and coins in test images, classifying coins as placed **on the tray** or **outside the tray**, recognizing their nominal value, and calculating the total value for both groups.

## Project Overview

The program processes a set of images (`tray1.jpg` to `tray8.jpg`) and performs the following steps:

1. image preprocessing,
2. tray detection using the Hough transform for lines,
3. coin detection using the Hough transform for circles,
4. classification of coins by position relative to the tray,
5. classification of coin nominal values based on detected radius,
6. counting coins and summing their values,
7. visualization of final results.

The project was created as part of a computer vision / image processing laboratory assignment.

## Main Idea

The tray is detected first, because its position is later used to decide whether a coin lies inside or outside the tray area.

The solution uses two separate Hough-based detection stages:

- **HoughLinesP** for tray boundary detection,
- **HoughCircles** for coin detection.

To improve robustness, the program applies:
- grayscale conversion,
- local contrast enhancement with **CLAHE**,
- Gaussian blur,
- Canny edge detection,
- morphological closing,
- filtering of false circular detections near rounded tray corners,
- adaptive threshold selection for coin denomination classification.

## Features

- Detects the tray using line segments
- Detects coins using circular Hough transform
- Separates coins into:
  - **on_tray**
  - **outside_tray**
- Classifies coin denomination into:
  - **5 gr**
  - **5 zl**
- Calculates:
  - number of coins on the tray,
  - number of coins outside the tray,
  - total value on the tray,
  - total value outside the tray
- Displays annotated output images with:
  - tray rectangle,
  - detected coins,
  - coin labels,
  - summary text

## Technologies

- Python
- OpenCV
- NumPy

## Project Structure

Example repository structure:

```text
.
├── lab2_transformata_hougha.py
├── tray1.jpg
├── tray2.jpg
├── tray3.jpg
├── tray4.jpg
├── tray5.jpg
├── tray6.jpg
├── tray7.jpg
├── tray8.jpg
├── results/
│   ├── trays_1-3.png
│   ├── trays_3-4.png
│   └── ...
└── README.md
```

## How It Works

### 1. Preprocessing

The input image is converted to grayscale and enhanced with CLAHE to improve local contrast.

Two slightly different preprocessed versions are created:
- one for **coin detection**,
- one for **tray detection**.

For tray detection, the program additionally uses:
- Gaussian blur,
- Canny edge detection,
- morphological closing to connect fragmented tray edges.

### 2. Tray Detection

The tray is detected using **probabilistic Hough transform** (`cv2.HoughLinesP`).

The algorithm:
- extracts vertical and horizontal line candidates,
- clusters vertical lines,
- selects the best left/right tray boundaries,
- estimates top and bottom boundaries from horizontal lines,
- uses a fallback strategy based on inner vertical segments when horizontal lines are insufficient.

This produces a rectangular tray area used later for coin classification.

### 3. Coin Detection

Coins are detected with `cv2.HoughCircles`.

The detector is tuned for the given image set by controlling:
- minimum distance between circles,
- radius range,
- edge threshold,
- accumulator threshold.

To reduce false positives, circles detected in rounded tray corner regions are discarded.

### 4. Position Classification

Each detected coin is classified according to the position of its center:
- if the center lies inside the detected tray rectangle → `on_tray`,
- otherwise → `outside_tray`.

### 5. Nominal Classification

The project distinguishes two nominal values:
- **5 gr**
- **5 zl**

Classification is based on the detected circle radius.

Instead of relying only on a single fixed threshold, the program computes an **adaptive radius threshold** for each image by finding the largest gap between sorted radii.  
If adaptive thresholding is not possible, a fallback threshold is used.

### 6. Counting and Summation

The program counts coins and computes total value separately for:
- coins on the tray,
- coins outside the tray.

### 7. Visualization

The final image contains:
- detected tray rectangle,
- circles around coins,
- nominal labels,
- summary information for both groups.

Color convention:
- **green** – coin on the tray,
- **red** – coin outside the tray,
- **yellow** – detected tray boundary.

## Running the Project

### Requirements

Install dependencies:

```bash
pip install opencv-python numpy
```

### Run

Make sure the image files `tray1.jpg` to `tray8.jpg` are in the same directory as the Python script.

Then run:

```bash
python lab2_transformata_hougha.py
```

## Input

The program expects image files named:

```text
tray1.jpg
tray2.jpg
...
tray8.jpg
```

## Output

For each processed image, the program:
- opens a result window with annotations,
- prints a short summary in the console.

Example summary:
- number of coins on the tray,
- total value of coins on the tray,
- number of coins outside the tray,
- total value of coins outside the tray.

## Notes

- The solution is tuned for the provided tray image set.
- Tray detection quality directly affects coin position classification.
- The main practical difficulty in this project was handling false detections near tray corners and making tray boundary estimation more stable across multiple images.

## Possible Improvements

- more robust tray shape modeling,
- better handling of perspective changes,
- automatic visualization export to files,
- more advanced denomination classification using additional visual features,
- support for larger and more varied datasets.

## Screenshots

Add example results here, for example:

```md
![Result 1](screenshots/result_1.png)
![Result 2](screenshots/result_2.png)
```

## Author

Tomasz Murach

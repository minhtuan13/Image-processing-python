# Introduction
- This project is a comprehensive exploration of fundamental image processing techniques implemented in Python. It includes various scripts that demonstrate color transformations, edge detection using both theoretical approaches and OpenCV's built-in functions, image filtering, and geometric transformations like scaling and rotation.

- Each script is designed to provide hands-on experience with different image processing algorithms, allowing users to understand both the underlying theory and practical implementation.

# Usage
Each script in the source/ directory focuses on different image processing tasks. Below is an overview of each script and how to run them.

# 1. Color Transformations
Script: Lab01_Color.py

- Description: Demonstrates linear and nonlinear color transformations, including brightness and contrast adjustments, logarithmic and exponential transformations, histogram equalization, and histogram specification.

- Key Features:

- Linear Transformations:
Brightness Adjustment
Contrast Adjustment
Combined Brightness & Contrast Adjustment

- Nonlinear Transformations:
Logarithmic Transformation
Exponential Transformation

- Probability-Based Transformations:
Histogram Equalization
Histogram Specification

# 2. Edge Detection (Theoretical Implementations)
Script: Lab01_Edge_Detection_LT.py

- Description: Implements edge detection algorithms from scratch, including Roberts, Sobel, Prewitt, Frei-Chen, Laplacian (4 and 8 masks), and Canny edge detection with non-maximal suppression.

- Key Features:
Roberts Edge Detection
Sobel Edge Detection
Prewitt Edge Detection
Frei-Chen Edge Detection
Laplacian Edge Detection (4 and 8 Masks)
Canny Edge Detection

# 3. Edge Detection (OpenCV Implementations)
Script: Lab01_Edge_Detection_OpenCV.py

- Description: Utilizes OpenCV's built-in functions to perform edge detection using Roberts, Sobel, Prewitt, Frei-Chen, Laplacian, Laplacian of Gaussian, and Canny operators.

- Key Features:
Edge detection using optimized OpenCV functions for better performance.
Comparison between custom implementations and OpenCV's methods.

# 4. Image Filtering
Script: Lab01_Filtering.py

- Description: Demonstrates various image smoothing techniques including Average (Mean) Filter, Median Filter, and Gaussian Filter both through manual implementations and OpenCV's built-in functions.

- Key Features:

Average Filter: Blurs the image by averaging the pixel values.
Median Filter: Reduces noise by replacing each pixel with the median of neighboring pixels.
Gaussian Filter: Applies Gaussian blur to smooth the image while preserving edges.
Comparison between custom filters and OpenCV's optimized functions.

# 5. Scaling and Rotation
Script: Lab01_Scale_Rotation.py

- Description: Implements image scaling and rotation manually without using OpenCV's built-in functions. Demonstrates the effects of these transformations on the image.

- Key Features:

Scaling: Enlarges or reduces the image based on a scaling factor.
Rotation: Rotates the image by a specified angle around its center.
Visualization of the transformed images.


- Note: Ensure that all necessary images (house.png, ex.png, cold.png) are placed in the images/ directory before running the scripts. Modify the image paths in the scripts if you choose a different directory structure.

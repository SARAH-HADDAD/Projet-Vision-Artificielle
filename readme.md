# About this Project:
This code performs a 3D reconstruction using stereo vision with two images taken from different perspectives. 
#### Here are the main steps:
- The camera is calibrated by loading the intrinsic parameters from a file.
- Two images are acquired and resized to a common size.
- SIFT features are detected and matched between the two images using the Brute-Force Matcher.
- RANSAC algorithm is applied to remove outliers and refine the matches.
- The 3D coordinates of the matched points are computed using the disparity and the camera intrinsic parameters.
- The 3D points are visualized in a 3D scatter plot using Matplotlib.

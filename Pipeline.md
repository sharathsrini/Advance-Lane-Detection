## Project Pipeline
1. Calibrate Camera  
2. Undistort Frame
3. Gradient Threshold
4. Color Threshold.
5. Combine THreshold
6. Perspective Transform
7. Find Lines
8. Measure Curvature
9. Measure Position.
10.

#### Calibrate Camera : 
1. BGR Color Space if it is OpenCV library or RGB if Matplotlib.
2. Chessboard image, given.
3. Grzayscale should be applied.
4. Output Parameters : Camera Matrix, Distort Coefficient.

#### Undistrot Frame
1. Image BGR is fed in.
1. One Function which could give us undistorted images.
2. Output is undistorted frame in BGR.

#### Gradient Threshold.
1. What king of Gradient Threshold?
2. X,Y
3. Magnitude and Direction of the Gradient. 
4. Combine Everything.
5. Magnitude and Direction finds its application in much more advnced application.

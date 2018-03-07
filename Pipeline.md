## Project Pipeline
1. Calibrate Camera  
2. Undistort Frame
3. Gradient Threshold
4. Color Threshold.
5. Combine Threshold
6. Perspective Transform
7. Find Lines
8. Measure Curvature
9. Measure Position.


#### Calibrate Camera : 
1. BGR Color Space if it is OpenCV library or RGB if Matplotlib.
2. Chessboard image, given.
3. Grzayscale should be applied. : Grayscale Images are single channel Binary Images.
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
6. Output is a Binary Image.

#### Color Threshold
1. Change the Color Space.
2. Apply various color spaces such as HSV, HLS, RGB, LAB, etc.
3. Output : Binary Image.

#### Combine Thresholds.
1. Apply combination of the various applied color threshold, or gradient threshold to obtain an image with granular level detail.
2. Output : Binary Image

#### Perspective Transform
1. Define the Source and the Destination points, which would help in transforming the image and could help oin the transformed image as the destination points plottted on to the transformed image.
2. Output: Warped Image, Binary, Different azShape.

#### Find Lines
1. Get the parameters useful to calculate the curvature.
2. Find and place the Left and Right Lanes , filter for noise.

#### Calculate Curvature
1. A methoed to calculate the Curvature using the equation could help us define a curvature with which  the lane is turning.
2. Output: Curvature of the Left and the Right lane

#### Calculate Position
1. This Would return a positional value the car on the frame.

#### Average
1. This method inputs the calculated curvature, and the recogonized lines, it will output a a perfect Curvature which will help us adding it tothre Position Pavem.

## Final Image
1. Multiple Parameters in a Sequential list.
2. For every frame, the Functions are called again and again. 
3. Suggestions by the mentor : Create Global variables or use Classes to provide acces throughout.

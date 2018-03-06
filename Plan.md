#### 05-03-2018
1. Advance deep learning in determistic approach 
2. Calibrating Camera 
     1.  Type of lens
     2. Position of mounting 
     3. Camera Distortion 
     4. Types of camera distortion
     5. Various loss
3. Deep learning on camera distortion 
4. Chess Board 
5. Chessboard processing 
6. Objects Points 
7. Shape of the object point
8. Perspective transformation
9. Inverse Perspective Transformation 
10. Warp Perspective transformation 
11. Source and destination Points 
 
1. IMAGE CALLIBRATION in drones.... With fish eye lens
DEADLINE : 1ST - 18TH  VIDEOS 


#### 06/03/2018
1. Why Sobel over Canny
2. Gradient Threshold - Sobel Operator
     1. Sobel X and Y
     2. cv2.Sobel - Takes in Image, Ksize(a int value that defines the size, and the more bigger, the more smoother ), 
     3. Magnitude and Direction of Gradient
     4. Gradient of X and Y axis are combined.
     5. Color Space Expirement - HLS, HSV, YUV
     6. Warp and Sobel 
     7. Histogram To visualize the Sobel image  to understand the bright-pixel distribution
     8. Windowing technique to find the lines : https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
     9. Apply Convolution.
     10. Measuring the Curvature.
     11. Measuring the Position.
     12. Where else can we use the xm_per_pix and ym_per_pix values?
     13. http://colorizer.org/
     
     
### Notes on Sobel and Canny edge Detector
Sobel detection refers to computing the gradient magnitude of an image using 3x3 filters. Where "gradient magnitude" is, for each a pixel, a number giving the greatest rate of change in light intensity in the direction where intensity is changing fastest.
Canny edge detection goes a bit further by removing speckle noise with a low pass filter first, then applying a Sobel filter, and then doing non-maximum suppression to pick out the best pixel for edges when there are multiple possibilities in a local neighborhood. That's a simplification, but basically its smarter than just applying a threshold to a Sobel filter, but it is still fairly low level processing.
"Edge detection" could refer to either of the above, or to many modern edge detection algorithms that are much more sophisticated than either of the above. For example there are edge detectors that have some success at finding edges between two textured regions while ignoring the edges in the textures themselves. There are edge detectors that are more global in scope in that they try to find edges between regions of homogeneous color or texture. Likewise, another global algorithm looks for edges that follow smooth contours even when parts of those contours are weak or obscured. A recent paper based edge detection on the statistics of color co-occurrence between adjacent pixels. I am too lazy to write out all the references. 

Edge detection (known as contour detection in more modern parlance) is an active area of research.

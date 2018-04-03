
## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

### My Approach :


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



## Final Image
1. Multiple Parameters in a Sequential list.
2. For every frame, the Functions are called again and again. 
3. Suggestions by the mentor : Create Global variables or use Classes to provide acces throughout.


## First, I'll compute the camera calibration using chessboard images


```python
import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
print("Packages loaded Successfully")
```

    Packages loaded Successfully
    


```python
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)


objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
#images_folder = []
i =0
figure = plt.figure(figsize=(30, 30))
images = glob.glob('camera_cal/calibration*.jpg')


for idx,fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)    
    if ret == True:
        i+=1
              
        objpoints.append(objp)
        imgpoints.append(corners) 
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        
        figure.add_subplot(5,4,i)
        plt.imshow(img)
```


![png](output_2_0.png)



```python
img = cv2.imread('camera_cal/calibration5.jpg')
img_size = (img.shape[1], img.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)
dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('test_undist.jpg',dst)

fig,(axis1, axis2) = plt.subplots(1, 2, figsize=(20,10))
axis1.imshow(img)
axis1.set_title('Original Image', fontsize=15)
axis2.imshow(dst)
axis2.set_title('Undistorted Image', fontsize=15)
plt.show()
```


![png](output_3_0.png)


#### Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, objp is just a replicated array of coordinates, and objpoints will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. imgpoints will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort() function 


```python
img = cv2.imread('camera_cal/calibration1.jpg')
dst = cv2.undistort(img, mtx, dist, None, mtx)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10), )
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=20)
plt.show()
```


![png](output_5_0.png)



```python
image = cv2.imread('test_images/straight_lines1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Undistort it and show the result
image_dst = cv2.undistort(image, mtx, dist, None, mtx)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=20)
ax2.imshow(image_dst)
ax2.set_title('Undistorted Image', fontsize=20)
plt.show()
```


![png](output_6_0.png)



```python
def getCalibrationParams(images, nx, ny):
    objPoints = []
    imgPoints = []

    objP = np.zeros((ny*nx,3), np.float32)
    objP[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)
        
        if ret == True:
            imgPoints.append(corners)
            objPoints.append(objP)

    return imgPoints, objPoints

# Calibrate Image
def calibrateImage(calibPath, calibImg):
    nx = 9
    ny = 6
    images = glob.glob(calibPath)
    imgPoints, objPoints = getCalibrationParams(images, nx, ny)

    img = cv2.imread(calibImg)
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, img_size, None, None)

    return mtx, dist


def hls_threshold(img, channel, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel == 'h':
        channelImg = hls[:,:,0]
    elif channel == 'l':
        channelImg = hls[:,:,1]
    elif channel == 's':
        channelImg = hls[:,:,2]
    hlsBinary = np.zeros_like(channelImg)
    hlsBinary[(channelImg > thresh[0]) & (channelImg <= thresh[1])] = 1
    return hlsBinary
```


```python
def hls_select(img, thresh=(0, 255)):   
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)    
    s_channel = hls[:,:,2]   
    binary_output = np.zeros_like(s_channel)      
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output
```


```python
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):    
   
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    gradBinary = np.zeros_like(scaled_sobel)
    gradBinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return gradBinary

```


```python
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):     
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)   
    gradmag = np.sqrt(sobelx**2 + sobely**2)   
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8)     
    magBinary = np.zeros_like(gradmag)
    magBinary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    
    return magBinary
```


```python
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    
    return binary_output
```

#### Methods to Generate Images at various Color-Spaces


```python
def blur(img, k=5):
    kernel_size = (k, k)
    return cv2.GaussianBlur(img, kernel_size, 0)

def channel(img, ch):
    return img[:, :, ch]

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def rgb2lab(img, ch=-1):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    if ch < 0:
        return img
    else:
        return img[:, :, ch]

def rgb2hls(img, ch=-1):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if ch < 0:
        return img
    else:
        return img[:, :, ch]

def rgb2hsv(img, ch=-1):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if ch < 0:
        return img
    else:
        return img[:, :, ch]

def rgb2yuv(img, ch=-1):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if ch < 0:
        return img
    else:
        return img[:, :, ch]
```

## Color Spaces

The Various Color Spaces that exists in our 

 [https://docs.opencv.org/3.2.0/df/d9d/tutorial_py_colorspaces.html]


```python
img = cv2.imread('test_images/straight_lines1.jpg')
p = bgr2rgb(img)
r = rgb2hsv(img)
x = rgb2yuv(img)
t = rgb2hls(img)
u = rgb2lab(img)
b = blur(img)
fig,(axis1, axis2) = plt.subplots(1, 2, figsize=(16,9))
axis1.imshow(img)
axis1.set_title('Original Image', fontsize=15)
axis2.imshow(p)
axis2.set_title('RGB Image', fontsize=15)

fig,(axis1, axis2) = plt.subplots(1, 2, figsize=(16,9))
axis1.imshow(img)
axis1.set_title('Original Image', fontsize=15)
axis2.imshow(x)
axis2.set_title('YUV Image', fontsize=15)

fig,(axis1, axis2) = plt.subplots(1, 2, figsize=(16,9))
axis1.imshow(img)
axis1.set_title('Original Image', fontsize=15)
axis2.imshow(t)
axis2.set_title('HLS Image', fontsize=15)

fig,(axis1, axis2) = plt.subplots(1, 2, figsize=(16,9))
axis1.imshow(img)
axis1.set_title('Original Image', fontsize=15 )
axis2.imshow(u)
axis2.set_title('LAB Image', fontsize=15)

fig,(axis1, axis2) = plt.subplots(1, 2, figsize=(16,9))
axis1.imshow(img)
axis1.set_title('Original Image', fontsize=15)
axis2.imshow(b)
axis2.set_title('Blur Image', fontsize=15)

fig,(axis1, axis2) = plt.subplots(1, 2, figsize=(16,9))
axis1.imshow(img)
axis1.set_title('Original Image', fontsize=15)
axis2.imshow(r)
axis2.set_title('HSV Image', fontsize=15)
plt.show()
```


![png](output_15_0.png)



![png](output_15_1.png)



![png](output_15_2.png)



![png](output_15_3.png)



![png](output_15_4.png)



![png](output_15_5.png)



```python
img = cv2.imread('test_images/straight_lines1.jpg')
gray = rgb2gray(img)
mag_threshold = mag_thresh(img, sobel_kernel=3, mag_thresh=(20, 100))
Sobel = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 100))
direction = dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3)) 
hls = hls_select(img,  thresh=(10, 150))
at = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,2)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=25)
ax2.imshow(mag_threshold, cmap='gray')
ax2.set_title('Thresholded Magnitude', fontsize=25)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=25)
ax2.imshow(Sobel, cmap='gray')
ax2.set_title('Sobel', fontsize=25)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=25)
ax2.imshow(direction, cmap='gray')
ax2.set_title('Direction', fontsize=25)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=25)
ax2.imshow(hls, cmap='gray')
ax2.set_title('HLS', fontsize=25)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=25)
ax2.imshow(at, cmap='gray')
ax2.set_title('Adaptive Threhold', fontsize=25)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```


![png](output_16_0.png)



![png](output_16_1.png)



![png](output_16_2.png)



![png](output_16_3.png)



![png](output_16_4.png)



```python
img = cv2.imread('test_images/straight_lines2.jpg') 


# HLS H-channel Threshold
hlsBinary_h = hls_threshold(img, channel='h', thresh=(10, 40))
# HLS L-channel Threshold
hlsBinary_l = hls_threshold(img, channel='l', thresh=(200, 255))
# HLS S-channel Threshold
hlsBinary_s = hls_threshold(img, channel='s', thresh=(100, 255)) 

f, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(16, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=25)
ax2.imshow(hlsBinary_h, cmap='gray')
ax2.set_title('H Threhold', fontsize=25)
ax3.imshow(hlsBinary_l, cmap='gray')
ax3.set_title('L Threhold', fontsize=25)
ax4.imshow(hlsBinary_s, cmap='gray')
ax4.set_title('S Threhold', fontsize=25)
plt.savefig('./HLS.jpg')
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


```


![png](output_17_0.png)


## A Note on Combining Thresholds

####  Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result
Sobel detection refers to computing the gradient magnitude of an image using 3x3 filters. Where "gradient magnitude" is, for each a pixel, a number giving the greatest rate of change in light intensity in the direction where intensity is changing fastest. Even though using Sobel, a lot of Noise had picked up, thus I tried combining the Hue, Lightness and the Satturation to detect the yellow and the white lines. I tried Using various Color Spaces Such as YUV, HSV, LAB. All the color spaces where helpful, but I ended up choosing th HLS.

Instead of edge detection, I used adaptive thresholding to further boost the line detection since the videos, especially the challenge videos, have lighting variations. Below the overpass in the challenge video, and constant switch between tree shadows and sunlight in the hard challenge video are all examples of lighting variations. Most edge detection approaches and color scheme based filters require a static threshold, which makes line detection harder.



###### The Threhold Applied Where: 
* Hue Channel : 10, 40
* Light Channel : 200, 255
* Saturation Channel : 100, 255





```python
img = cv2.imread('test_images/straight_lines2.jpg') 
gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(5, 100))
grady = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(10, 120))
# Sobel Magnitude
magbinary = mag_thresh(img, mag_thresh=(10, 150))

# Sobel Direction
dirbinary = dir_threshold(img, thresh=(0.8, 1.0))

combined2 = np.zeros_like(dirbinary)
combined2[((gradx == 1) & (grady == 1)) | ((magbinary == 1) & (dirbinary == 1))] = 1


gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
adaptive_Thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,2)
# HLS S-channel Threshold
hlsBinary_s = hls_threshold(img, channel='s', thresh=(100, 255)) 
# HLS H-channel Threshold
hlsBinary_h = hls_threshold(img, channel='h', thresh=(10, 40))
# HLS L-channel Threshold
hlsBinary_l = hls_threshold(img, channel='l', thresh=(200, 255))
# Combine channel thresholds
combined = np.zeros_like(hlsBinary_s)
combined[((hlsBinary_h == 1) & (hlsBinary_s == 1)) | (hlsBinary_l == 1) | (adaptive_Thresh == 1)] = 1

f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(16, 9))
    
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=15)
ax2.imshow(combined, cmap='gray')
ax2.set_title('Combined-HLS Threshold', fontsize=15)
ax3.imshow(combined2, cmap='gray')
ax3.set_title('Combined-Sobel and Gradient', fontsize=15)
plt.savefig('./Combined.jpg')

#adaptive_Thresh.copyTo(gray(cv::Rect(x,y,adaptive_Thresh.cols, adaptive_Thresh.rows)));
```


![png](output_19_0.png)



```python
def region_of_interest(img):
    mask = np.zeros_like(img)   
    
    imshape = img.shape    
    if len(imshape) > 2:
        channel_count = img.shape[2]  
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    vertices = np.array([[(0,imshape[0]),(imshape[1]*.48, imshape[0]*.58), (imshape[1]*.52, imshape[0]*.58), (imshape[1],imshape[0])]], dtype=np.int32)              
    cv2.fillPoly(mask, vertices, ignore_mask_color)   
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def warpPerspective(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  
    Minv = cv2.getPerspectiveTransform(dst, src)
    return warped, M, Minv

```


```python
def fitPolynomial(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = []
    right_fit = []
    if ((lefty.size > 0) and (leftx.size > 0)):
        left_fit = np.polyfit(lefty, leftx, 2)
    if((righty.size > 0) and (rightx.size > 0)):
        right_fit = np.polyfit(righty, rightx, 2)
    
    
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]



    return left_lane_inds, right_lane_inds, left_fit, right_fit, nonzerox, nonzeroy
```


```python
def getCurvature(img, ploty, left_fit, right_fit, left_fitx, right_fitx):

    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters

    h = img.shape[0]
    left_fit_pos = left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2]
    right_fitx_pos = right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2]
    lane_center = (left_fit_pos + right_fitx_pos)//2 #(left_fitx[0] + right_fitx[0])//2
    image_center = img.shape[1]//2
    vehicle_pos = (image_center - lane_center) * xm_per_pix

    return left_curverad, right_curverad, vehicle_pos
```

### Draw - Lanes

I tried implementing a two point solution, where there would be two lane regions which detects the left lane from the center, and the other one detects the right lane from the center of the lane. I also wanted to implement a locking mechanism which would lock the lane region once the road is seen with divider on both sides


```python
def drawLanes(image, undist, binary_warped, left_fitx, right_fitx, ploty, Minv):
    # Create an image to draw the lines on    
    
    left_lane_inds, right_lane_inds, left_fit, right_fit, nonzerox, nonzeroy = fitPolynomial(binary_warped)
    img_x = binary_warped.shape[1]
    img_y = binary_warped.shape[0]
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_y-1, img_y )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    center_fit = (left_fit + right_fit)/2
    center_fitx = center_fit[0]*ploty**2 + center_fit[1]*ploty + center_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_center = np.array([np.flipud(np.transpose(np.vstack([center_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_center))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    ## Recast the x and y points into usable format for cv2.fillPoly()
    pts_center = np.array([np.transpose(np.vstack([center_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_center, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (255,0, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)
   

    return result
```


```python
def visualize_Lanes(binary_warped, left_lane_inds, right_lane_inds, left_fit, right_fit, nonzerox, nonzeroy):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    return ploty, left_fitx, right_fitx, out_img
```

#### Get-Lanes
This method helps in combining the mathematical formulaes to display on top of the video. The Vehicle position along with the direction of the car. On my opinion, I think this could be used to calculate and predict the state of the motion model(Vehicle) by fusing this information with an IMU. This could possibly help in the sensor fusion along with the heading and the Yaw rate  


```python
def getLanes(img, imgUndistort, warpedImg, Minv):
    left_lane_inds, right_lane_inds, left_fit, right_fit, nonzerox, nonzeroy = fitPolynomial(warpedImg)

    if len(right_fit) <= 0 or len(left_fit) <= 0:
        return img

    ploty, left_fitx, right_fitx, out_img = visualize_Lanes(warpedImg, left_lane_inds, right_lane_inds, left_fit, right_fit, nonzerox, nonzeroy)
        
    output = drawLanes(img, imgUndistort, warpedImg, left_fitx, right_fitx, ploty, Minv)
   

    direction = 'straight'
    if (left_fitx[0] - left_fitx[len(left_fitx)-1] < -20):
        direction = 'left'
    if (left_fitx[0] - left_fitx[len(left_fitx)-1] > 20):
        direction = 'right'

    left_curverad, right_curverad, vehicle_pos = getCurvature(imgUndistort, ploty, left_fit, right_fit, left_fitx, right_fitx)
    
    avgText = 'Radius of Curvature : ' + '{:04.2f}'.format((left_curverad + right_curverad)/2) + 'm'
    cv2.putText(output, avgText, (300,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,255,155), 2, cv2.LINE_AA)

    closest = ' To the Left of centre'
    if vehicle_pos > 0:
        closest = '  To the Right  of centre'

    avgText = 'Vehicle Position : ' + '{:04.2f}'.format(abs(vehicle_pos)) + closest
    cv2.putText(output, avgText, (300,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,255,155), 2, cv2.LINE_AA)
    
    avgText = 'Direction of Heading : ' + direction +  '{:04.5f}'.format(left_fitx[0]) + '{:04.5f}'.format(left_fitx[len(left_fitx)-1])
    cv2.putText(output, avgText, (300,150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,255,155), 2, cv2.LINE_AA)

    return output
```

#### Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

* The code to compute the radius of curvature from the course notes was obtained. Using the lanes, we can compute the position of the vehicle.

* In addition, I also tried to check if the vechicle is going straight, turning left or turning right based on the lines. The images below show the direction of turn.




```python
def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids



```

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



```python
def pipeline(img):  
    
    imgUndistort = cv2.undistort(img, mtx, dist, None, mtx)  
    height,width = imgUndistort.shape[:2]
    width_Offset = 450
    height_Offset = 0
    src = np.float32([(575, 464), (707, 464), (258, 682), (1049, 682)])
    dst = np.float32([(width_Offset, height_Offset), (width-width_Offset, height_Offset), (width_Offset, height-height_Offset), (width-width_Offset, height-height_Offset)])
    gray = cv2.cvtColor(imgUndistort, cv2.COLOR_RGB2GRAY)
    adaptive_Thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,2)
    hlsBinary_h = hls_threshold(imgUndistort, channel='h', thresh=(10, 40))# HLS H-channel Threshold
    hlsBinary_l = hls_threshold(imgUndistort, channel='l', thresh=(200, 255))# HLS L-channel Threshold
    hlsBinary_s = hls_threshold(imgUndistort, channel='s', thresh=(100, 255))# HLS S-channel Threshold  
    combined = np.zeros_like(hlsBinary_s) # Combine channel thresholds
    combined[((hlsBinary_h == 1) & (hlsBinary_s == 1)) | (hlsBinary_l == 1) | (adaptive_Thresh == 1)] = 1
    roiImg = region_of_interest(combined)
    resize_roi = cv2.resize(np.dstack((roiImg, roiImg, roiImg))*255,(192,154))#Resize the image to fit the picture in the video
    warpedImg, M, Minv = warpPerspective(roiImg, src, dst)
    resize_warped = cv2.resize(np.dstack((warpedImg, warpedImg, warpedImg))*255,(192,154))
    output = getLanes(img, imgUndistort, warpedImg, Minv)# Fit polynomial and get Lanes
    x_offset = 50
    y_offset = 50
    output[y_offset:y_offset+resize_roi.shape[0],x_offset:x_offset+resize_roi.shape[1]] = resize_roi
    x_offset = 50
    y_offset = 250
    output[y_offset:y_offset+resize_warped.shape[0],x_offset:x_offset+resize_warped.shape[1]] = resize_warped
    return output

```


```python
images = glob.glob('./test_images/*.jpg')
for image in images:
    print(image)
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgUndistort = cv2.undistort(img, mtx, dist, None, mtx)  
    height,width = imgUndistort.shape[:2]
    width_Offset = 450
    height_Offset = 0
    src = np.float32([(575, 464), (707, 464), (258, 682), (1049, 682)])
    dst = np.float32([(width_Offset, height_Offset), (width-width_Offset, height_Offset), (width_Offset, height-height_Offset), (width-width_Offset, height-height_Offset)])
    gray = cv2.cvtColor(imgUndistort, cv2.COLOR_RGB2GRAY)
    adaptive_Thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,2)
    hlsBinary_h = hls_threshold(imgUndistort, channel='h', thresh=(10, 40))# HLS H-channel Threshold
    hlsBinary_l = hls_threshold(imgUndistort, channel='l', thresh=(200, 255))# HLS L-channel Threshold
    hlsBinary_s = hls_threshold(imgUndistort, channel='s', thresh=(100, 255))# HLS S-channel Threshold  
    combined = np.zeros_like(hlsBinary_s) # Combine channel thresholds
    combined[((hlsBinary_h == 1) & (hlsBinary_s == 1)) | (hlsBinary_l == 1) | (adaptive_Thresh == 1)] = 1
    roiImg = region_of_interest(combined)
    
    resize_roi = cv2.resize(np.dstack((roiImg, roiImg, roiImg))*255,(192,154))
    #print(resize_roi.shape)
    warpedImg, M, Minv = warpPerspective(roiImg, src, dst)  
    resize_warped = cv2.resize(np.dstack((warpedImg, warpedImg, warpedImg))*255,(192,154))
    x_offset = 50
    y_offset = 50
    
    corner_offset = 0
    #print((resize_roi.dtype))
    output = getLanes(img, imgUndistort, warpedImg, Minv)# Fit polynomial and get Lanes 
    #print(output.shape)
    #print((output.dtype))
    output[y_offset:y_offset+resize_roi.shape[0],x_offset:x_offset+resize_roi.shape[1]] = resize_roi
    
    x_offset = 50
    y_offset = 250
    output[y_offset:y_offset+resize_warped.shape[0],x_offset:x_offset+resize_warped.shape[1]] = resize_warped
    
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8)  = plt.subplots(1, 8, figsize=(16, 9))

    ax1.imshow(imgUndistort)
    ax1.set_title('Original Image', fontsize=15)
    ax2.imshow(gray, cmap='gray')
    ax2.set_title('Gray', fontsize=15)
    ax3.imshow(adaptive_Thresh, cmap='gray')
    ax3.set_title('Adaptive Threshold', fontsize=15)
    ax4.imshow(hlsBinary_h, cmap='gray')
    ax4.set_title('H', fontsize=15)
    ax5.imshow(hlsBinary_l, cmap='gray')
    ax5.set_title('L', fontsize=15)
    ax6.imshow(hlsBinary_s, cmap='gray')
    ax6.set_title('S', fontsize=15)
    ax7.imshow(combined, cmap='gray')
    ax7.set_title('Combined', fontsize=15)
    ax8.imshow(warpedImg, cmap='gray')
    ax8.set_title('Warped', fontsize=15)    
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)



    f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(16, 9))
    f.tight_layout()
    ax1.imshow(imgUndistort)
    ax1.set_title('Original Image', fontsize=25)
    ax2.imshow(roiImg, cmap='gray')
    ax2.set_title('Region of Interest', fontsize=25)
    ax3.imshow(output, cmap='gray')
    ax3.set_title('Pipeline Output', fontsize=25)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)        
    plt.savefig('./histogram_2.jpg')

```

    ./test_images\straight_lines1.jpg
    ./test_images\straight_lines2.jpg
    ./test_images\test1.jpg
    ./test_images\test2.jpg
    ./test_images\test3.jpg
    ./test_images\test4.jpg
    ./test_images\test5.jpg
    ./test_images\test6.jpg
    


![png](output_32_1.png)



![png](output_32_2.png)



![png](output_32_3.png)



![png](output_32_4.png)



![png](output_32_5.png)



![png](output_32_6.png)



![png](output_32_7.png)



![png](output_32_8.png)



![png](output_32_9.png)



![png](output_32_10.png)



![png](output_32_11.png)



![png](output_32_12.png)



![png](output_32_13.png)



![png](output_32_14.png)



![png](output_32_15.png)



![png](output_32_16.png)



```python
vid_output = 'Project_video_Submit.mp4'
clip2 = VideoFileClip('project_video.mp4') #harder_challenge_video
vid_clip = clip2.fl_image(pipeline)
vid_clip.write_videofile(vid_output, audio=False)
```

    [MoviePy] >>>> Building video Project_video_Submit.mp4
    [MoviePy] Writing video Project_video_Submit.mp4
    

    100%|█████████████████████████████████████████████████████████████████████████████▉| 1260/1261 [04:18<00:00,  5.64it/s]
    

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: Project_video_Submit.mp4 
    
    

#### Further Improvements :

* Adaptive thresholding when combined with other image processing techniques such as histogram equalization, contrast adjustments or brightness adjustments could yield better results. I tried histogram equalization, but that had little to no effect.

* While the region of interest was both useful in general, it cut off parts of the lane in several challenging cases. One improvement would be to adjust the bird eye view to account for thisIt is possible for us to learn the presence of lane lines, and predict if a region is a lane or not. That might help mitigate some of the effects of lighting.
 
* Improved Detection of lanes Can help in exact a calculation of state estimate using a Kalman or an Extended Kalman Filter that may Update and Predict our State using Sensors and various other computer Vision technique.

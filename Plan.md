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


#### 07/03/2018

1. Generative Adversial Networks 
2. Statistics - PDF
3. Gaussian Distribution, Random Sampple Function
4. PDF for each Pixel in  the image.
5. High Dimensional PDF
6. Input is a vector of Random numbers
7. THese numbers are sampled from a uniform Distribution
8.  The way you generate images is by mapping a random number with a PDF,  the distribution is learnt by the GAN.
9. Collection of input and output may lead to a deterministic i.e a pattern can be understood, and thus a image creation pattern could be seen , but cannot be controlled.
10. github/vdumoulin/conv_arthimetic
11. Discriminator : Convolution network - 64x64x3 is taken in and a single output is give.
12. This tries to tell us if the image is fke or not. This is called  DCGAN -Combined Architecture.
13. After a Equilibrium, the Discriminator would be confused and will output 0.5 as PDF. The Discriminator is being fooled.
14. DCGAN ----- Convolution is used. Deep Convolution 
15. If x is is the real world dataser, z is the random noise, oytput of the GAN is G(z). if discriminator has a function of D, the optimization objective would be to maximize D(x), but miniminze D(G(z)).
16. Generator would maximize the output i.e. D(G(z)) to fool the discriminator.
17. pz = The known Distribution
18. pdata = the unknown distribution of the images.
19. pz = The probability of generator images.
20. We would want pz = pdata.
21. LSUN Bedrooms Dataset.
22. Variations of GANs : github:eriklindernoren/Keras-GAN , github:junyanz/CycleGAN
23. Data-Augmentation using GANs. 
24. Image Super Resolution: SR GAN
25. Image Completion.
26. Semi Supervised Learning with GANs, we have a huge unlabelled dataset, train the GAN on a unlabelled Data, modify the Discriminiator to produce an additional output indicatiiong the label of the input.
27. Train the Discriminator on a label dataset.
28. OpenAI 's impleentation.
29. SIM-GAN , images synthetically generated from images from games like GTA V.

#### 08/03/2018
1. Claroom Sliding Window Code Discussion.
2. 

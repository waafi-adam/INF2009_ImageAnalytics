**Image Analytics with Raspberry Pi using Web Camera**

**Objective:** By the end of this session, participants will understand how to set up a web camera with the Raspberry Pi, capture images, and perform basic and advanced image analytics.

---

**Prerequisites:**
1. Raspberry Pi with Raspbian OS installed.
2. MicroSD card (16GB or more recommended).
3. Web camera compatible with Raspberry Pi (Will be using USB Webcam for this experiment).
4. Internet connectivity (Wi-Fi).
5. Basic knowledge of Python and Linux commands.

---

**1. Introduction (10 minutes)**
Computer vision has been a very popular field since the advent of digital systems. However computer vision on the edge devices such as Raspberry Pi is challenging due to resource contraints. Edge Computer Vision (ECV) has emerged as a transformative technology, with [Gartner](https://www.linkedin.com/pulse/what-edge-computer-vision-how-get-started-deep-block-net) recognizing it as one of the top emerging technologies of 2023. ECV offers several benefits such as 1) they can operate in real-time or near-real-time, providing instant insights and enabling immediate actions, 2) they offer enhanced privacy and security and 3) It reduces dependency on network connectivity or relaxes the bandwidth requirements as some processing will be done within. 
In this lab, few basic and advanced image processing tasks on edge devices is introduced. An overview of the experiments/setup is as follows:
![image](https://github.com/drfuzzi/INF2009_VideoAnalytics/assets/52023898/882c84dc-1989-4039-807d-554a079e3776)

**2. Setting up the Raspberry Pi (15 minutes)**
- Booting up the Raspberry Pi.
- Setting up Wi-Fi/Ethernet.
- System updates:
  ```bash
  sudo apt update
  sudo apt upgrade
  ```
- **[Important!] Set up and activate a virtual environment named "image" for this experiment (to avoid conflicts in libraries) as below**
  ```bash
  sudo apt install python3-venv
  python3 -m venv image
  source image/bin/activate

**3. Connecting and Testing the Web Camera (5 minutes)**
- Physically connect the web camera to the Raspberry Pi.
  
**. Introduction to Real-time Image Processing with Python (25 minutes)**
- Installing OpenCV:
  ```bash
  pip install opencv-python  
  ```
- The [sample code](Codes/image_capture_display.py) shows the code to read frames from a webcam and then based on the intensity range for each colour channel (RGB), how to segment the image into red green and blue images. A sample image and the colour segmentation is as shown below:
  ![image](https://github.com/drfuzzi/INF2009_ImageAnalytics/assets/52023898/fd7c115d-0301-0d2-b2c1-7966dce3fec)
- Expand the code to segment another colour (say yellow)

```python
import cv2
import numpy as np

# Define color boundaries in HSV space
# HSV is better for color segmentation than RGB
boundaries = {
    "Red": ([0, 120, 70], [10, 255, 255]),       # Lower range of red
    "Red2": ([170, 120, 70], [180, 255, 255]),   # Upper range of red
    "Green": ([36, 25, 25], [86, 255, 255]),     # Green range
    "Blue": ([94, 80, 2], [126, 255, 255]),      # Blue range
    "Yellow": ([15, 150, 150], [35, 255, 255])   # Yellow range
}

# Normalize image for display (scales pixel values to 0-255)
def normalizeImg(Img):
    Img = np.float64(Img)
    norm_img = (Img - np.min(Img)) / (np.max(Img) - np.min(Img))
    norm_img = np.uint8(norm_img * 255.0)
    return norm_img

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    output = []

    # Apply color masks and extract regions for each color
    for color_name, (lower, upper) in boundaries.items():
        lower_bound = np.array(lower, dtype="uint8")
        upper_bound = np.array(upper, dtype="uint8")
        
        # Create mask for current color range
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        
        # Apply the mask to the original frame
        segmented = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Normalize segmented image and store it
        output.append(normalizeImg(segmented))

    # Combine both red ranges into one
    red_combined = cv2.add(output[0], output[1])  # Red + Red2

    green_img = output[2]
    blue_img = output[3]
    yellow_img = output[4]

    # Concatenate original and processed images side by side
    catImg = cv2.hconcat([frame, red_combined, green_img, blue_img, yellow_img])
    
    # Show the result
    cv2.imshow("Original | Red | Green | Blue | Yellow", catImg)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
```

**5. Real-time Image Analysis (25 minutes)**
- Installing scikit-image:
  ```bash
  pip install scikit-image  
  ```
- Computer vision employs feature extraction from images. Some important image features include edges and textures. In this section we will employ a feature named histogram of gradients (HoG) which is widely employed for face recognition and other tasks. HoG involves gradient operation (basically extracting edges) on various image patches (by dividing the image into blocks). A [sample code](Codes/image_hog_feature.py) involving scikit-image is employed for the same. The code displays the dominant HoG image for each image patch overlaid on the actual image. It has to be noted that OpenCV can also be employed for the same task, but the visualization using scikit-image is better compared to that from OpenCV. A sample image for the HoG feature is as shown below:
![image](https://github.com/drfuzzi/INF2009_ImageAnalytics/assets/52023898/94e7d597-c259-4634-a3dc-433c79e8533b)
  -  Note the usage of colour (RGB) to gray scale converion employed before HoG feature extraction.
  - Run the code with and without resizing the image and observe the resultant frame rate. It is important to note that for edge computing, downsizing the image will speed up the compute and many such informed decisions are critical.
  - Change the patch size in line 25 (feature.hog) and observe the changes in the results.

# EXPLAIN START

Based on the instructions shown in your image and the code you provided, here’s a breakdown of what **changes or experiments** you can try in the code to match the suggestions:

---

### ✅ **Instruction 1: RGB to grayscale**
Already handled:
```python
image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```
✅ No change needed. This converts the color image to grayscale before applying HoG.

---

### ✅ **Instruction 2: Resize the image to test speed**
To test how resizing improves performance, uncomment the following line:
```python
#frame = cv2.resize(frame, (256, 256))  # Uncomment and see the speed up
```
⬇️ Change to:
```python
frame = cv2.resize(frame, (256, 256))  # Resize to reduce processing time
```
📌 You can even test different sizes like `(128, 128)` or `(320, 240)` to see performance differences.

---

### ✅ **Instruction 3: Change the patch size (`pixels_per_cell`)**
Locate this part:
```python
(H, hogImage) = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
	cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
	visualize=True)
```

🔁 Try changing `pixels_per_cell=(8, 8)` to something like:
```python
pixels_per_cell=(16, 16)
```
or
```python
pixels_per_cell=(4, 4)
```

This affects the **patch size**, which changes the detail level in the HoG visualization.

---

### ✨ Example: Apply All Three Suggestions

Here’s a version of your code with all instructions applied:

```python
import cv2
import numpy as np
from skimage import feature
from skimage import exposure

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    try:
        ret, frame = cap.read()

        # ✅ Resize for faster processing
        frame = cv2.resize(frame, (256, 256))

        # ✅ Convert to grayscale for HoG
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ✅ Change patch size for HoG feature extraction
        (H, hogImage) = feature.hog(
            image,
            orientations=9,
            pixels_per_cell=(16, 16),  # Try different values like (4, 4), (8, 8), etc.
            cells_per_block=(2, 2),
            transform_sqrt=True,
            block_norm="L1",
            visualize=True
        )

        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")
        hogImg = cv2.cvtColor(hogImage, cv2.COLOR_GRAY2RGB)

        catImg = cv2.hconcat([frame, hogImg])
        cv2.imshow("HOG Image", catImg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        break

cap.release()
cv2.destroyAllWindows()
```

Let me know if you’d like sliders or a GUI to change patch size or resize live!

# EXPLAIN END

- The HoG features can be employed to identify the presence of face. An [example using OpenCV](Codes/image_human_capture.py) is available for experimenting with. A multiscale HoG feature extraction is employed in this case. This involves extracting HoG features at multiple scales (resolutions) of the given image. 

**6. Real-time Image Feature Analysis for Face Capture and Facial Landmark Extraction (20 minutes)**
- In this work, a light weight opensource library named *"Mediapipe"* for tasks such as face landmark detection, pose estimation, hand landmark detection, hand gesture recognition and object detection using pretrained neural network models.
- [MediaPipe](https://developers.google.com/mediapipe) is a on-device (*embedded machine learning*) framework for building cross platform multimodal applied ML pipelines that consist of fast ML inference, classic computer vision, and media processing (e.g. video decoding). MediaPipe was open sourced at CVPR in June 2019 as v0.5.0 and has various lightweight models developed with Tensorflow lite available for usage.
- Installing media pipe:
  ```bash  
  pip install mediapipe
  ```
- Try the [sample code](Codes/image_face_capture.py) to detect the face based on Mediapipe's approach which is very light weight when compared to the approach employed in above section. Observe the speed up. - A sample image with face landmarks is as shown below:
![Mediapipe Face Mesh_screenshot_18 01 2025](https://github.com/user-attachments/assets/3e952cbb-72df-4258-9d96-83f05c741096)

- [Optional] An opencv alternative (no dependence on mediapipe) of the face detection is available in the [sample code](Codes/image_human_capture_opencv.py). If you are using this code, make sure you download the [Haar cascade model](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml) manually and save it as 'haarcascade_frontalface_alt2.xml' in the same folder as the code. 
---

**[Optional] Homework/Extended Activities:**
1. Explore more advanced OpenCV functionalities like SIFT, SURF, and ORB for feature detection. These features alongside HoG could be used for image matching (e.g. face recognition)
2. Build an eye blink detection system for drowsiness detection.  

---

**Resources:**
1. Raspberry Pi official documentation.
2. OpenCV documentation and tutorials.
3. Relevant Python libraries documentation for image processing (e.g., `opencv`, `scikit-image`, `mediapipe`).

---


# Image-Handling-and-Pixel-Transformations-Using-OpenCV 

## AIM:
Write a Python program using OpenCV that performs the following tasks:

1) Read and Display an Image.  
2) Adjust the brightness of an image.  
3) Modify the image contrast.  
4) Generate a third image using bitwise operations.

## Software Required:
- Anaconda - Python 3.7
- Jupyter Notebook (for interactive development and execution)

## Algorithm:
### Step 1:
Load an image from your local directory and display it.

### Step 2:
Create a matrix of ones (with data type float64) to adjust brightness.

### Step 3:
Create brighter and darker images by adding and subtracting the matrix from the original image.  
Display the original, brighter, and darker images.

### Step 4:
Modify the image contrast by creating two higher contrast images using scaling factors of 1.1 and 1.2 (without overflow fix).  
Display the original, lower contrast, and higher contrast images.

### Step 5:
Split the image (boy.jpg) into B, G, R components and display the channels

## Program Developed By:
- **Name:** [Your Name Here]  
- **Register Number:** [Your Register Number Here]

  ### Ex. No. 01

#### 1. Read the image ('Eagle_in_Flight.jpg') using OpenCV imread() as a grayscale image.
```
import cv2
img = cv2.imread('Eagle_in_Flight.jpg', cv2.IMREAD_GRAYSCALE)
```

#### 2. Print the image width, height & Channel.
```
image = cv2.imread('Eagle_in_Flight.jpg')
print("Height, Width and Channel:", image.shape)
```

#### 3. Display the image using matplotlib imshow().
```
import matplotlib.pyplot as plt
plt.imshow(img)
```

#### 4. Save the image as a PNG file using OpenCV imwrite().
```
cv2.imwrite('Eagle_in_Flight.png', image)
```

#### 5. Read the saved image above as a color image using cv2.cvtColor().
```
img = cv2.imread('Eagle_in_Flight.png')
color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(color_img)
```

#### 6. Display the Colour image using matplotlib imshow() & Print the image width, height & channel.
```
plt.imshow(color_img)
color_img.shape
```

#### 7. Crop the image to extract any specific (Eagle alone) object from the image.
```
cropped = color_img[10:450, 150:570]
plt.imshow(cropped)
plt.axis("off")
```

#### 8. Resize the image up by a factor of 2x.
```
height, width = image.shape[:2]
resized_image = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
plt.imshow(resized_image)
```

#### 9. Flip the cropped/resized image horizontally.
```
flipped = cv2.flip(resized_image, 1)
plt.imshow(flipped)
```

#### 10. Read in the image ('Apollo-11-launch.jpg').
```
img_apollo = cv2.imread('Apollo-11-launch.jpg')
```

#### 11. Add the following text to the dark area at the bottom of the image (centered on the image):
```python
text = 'Apollo 11 Saturn V Launch, July 16, 1969'
font_face = cv2.FONT_HERSHEY_PLAIN
cv2.putText(img_apollo, 'Apollo 11 Saturn V Launch, July 16, 1969', (50, img_apollo.shape[0] - 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
plt.imshow(img_apollo)
```

#### 12. Draw a magenta rectangle that encompasses the launch tower and the rocket.
```python
rect_color = magenta
cv2.rectangle(img_apollo, (400, 30), (750, 600), (255, 0, 255), 3)
```

#### 13. Display the final annotated image.
```
plt.imshow(img_apollo)
```

#### 14. Read the image ('Boy.jpg').
```
boy_img = cv2.imread('Boy.jpg')
```

#### 15. Adjust the brightness of the image.
```
# Create a matrix of ones (with data type float64)
# matrix_ones = 
import numpy as np
matrix= np.ones(boy_img.shape, dtype='uint8') * 50

```

#### 16. Create brighter and darker images.
```
img_brighter = cv2.add(img, matrix)
img_darker = cv2.subtract(img, matrix)
# YOUR CODE HERE
```

#### 17. Display the images (Original Image, Darker Image, Brighter Image).
```
plt.figure(figsize=(10, 3))
for i, img in enumerate([boy_img, img_darker, img_brighter]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```

#### 18. Modify the image contrast.
```python
# Create two higher contrast images using the 'scale' option with factors of 1.1 and 1.2 (without overflow fix)
matrix1 = np.ones(boy_img.shape, dtype='uint8') * 25
matrix2 = np.ones(boy_img.shape, dtype='uint8') * 50
img_higher1 = cv2.addWeighted(boy_img, 1.1, matrix1, 0, 0)
img_higher2 = cv2.addWeighted(boy_img, 1.2, matrix2, 0, 0)
```

#### 19. Display the images (Original, Lower Contrast, Higher Contrast).
```
plt.figure(figsize=(10, 3))
for i, img in enumerate([boy_img, img_higher1, img_higher2]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```

#### 20. Split the image (boy.jpg) into the B,G,R components & Display the channels.
```
b, g, r = cv2.split(boy_img)
plt.figure(figsize=(10, 3))
for i, channel in enumerate([b, g, r]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(channel, cmap='gray')
plt.show()
```

#### 21. Merged the R, G, B , displays along with the original image
```
merged = cv2.merge([b, g, r])
plt.imshow(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))
plt.show()
```

#### 22. Split the image into the H, S, V components & Display the channels.
```
hsv = cv2.cvtColor(boy_img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
plt.figure(figsize=(10, 3))
for i, channel in enumerate([h, s, v]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(channel, cmap='gray')
plt.show()
```
#### 23. Merged the H, S, V, displays along with original image.
```
merged_hsv = cv2.merge([h, s, v])
plt.imshow(cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2RGB))
plt.show()
```

## Output:
- **i)** Read and Display an Image.
![Screenshot 2025-03-12 105241](https://github.com/user-attachments/assets/1fab8abf-ed73-4ed5-ba7d-ed9eef59e20c)

   
- **ii)** Adjust Image Brightness.

-  
- **iii)** Modify Image Contrast.  
- **iv)** Generate Third Image Using Bitwise Operations.

## Result:
Thus, the images were read, displayed, brightness and contrast adjustments were made, and bitwise operations were performed successfully using the Python program.


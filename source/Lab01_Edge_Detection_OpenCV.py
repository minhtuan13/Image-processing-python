import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


#Sử dụng các hàm trong OpenCV để phát hiện biên cạnh

# Đọc ảnh
img = cv2.imread('house.png', cv2.IMREAD_GRAYSCALE)

imgColor = cv2.imread('house.png', cv2.IMREAD_COLOR)
imgColor = cv2.cvtColor(imgColor, cv2.COLOR_BGR2RGB)


# Toán tử Gradient dùng mặt nạ Roberts
roberts_x = cv2.filter2D(img, cv2.CV_64F, np.array([[1, 0], [0, -1]])) #X-axis
roberts_y = cv2.filter2D(img, cv2.CV_64F, np.array([[0, 1], [-1, 0]])) #Y-axis
edge_image_roberts = np.sqrt(roberts_x**2 + roberts_y**2) # Tính độ lớn của đạo hàm
# Chuyển đổi giá trị độ lớn về định dạng uint8 để có thể hiển thị
edge_image_roberts = np.uint8(edge_image_roberts)


# Toán tử Gradient dùng mặt nạ Sobel
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) #X-axis
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) #Y-axis
edge_image_sobel = np.sqrt(sobelx**2 + sobely**2)# Tính độ lớn của đạo hàm
# Chuyển đổi giá trị độ lớn về định dạng uint8 để có thể hiển thị
edge_image_sobel = np.uint8(edge_image_sobel)


# Toán tử Gradient dùng mặt nạ Prewitt 
prewittx = cv2.filter2D(img, cv2.CV_64F, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])) #X-axis
prewitty = cv2.filter2D(img, cv2.CV_64F, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])) #Y-axis
edge_image_prewitt = np.sqrt(prewittx**2 + prewitty**2)# Tính độ lớn của đạo hàm
# Chuyển đổi giá trị độ lớn về định dạng uint8 để có thể hiển thị
edge_image_prewitt = np.uint8(edge_image_prewitt)

# Tạo figure thứ 1

plt.figure(figsize = (12, 12))
plt.subplot(2, 2, 1)
plt.imshow(imgColor, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(edge_image_roberts, cmap='gray')
plt.title('Roberts Edge Image')

plt.subplot(2, 2, 3)
plt.imshow(edge_image_sobel, cmap='gray')
plt.title('Sobel Edge image')

plt.subplot(2, 2, 4)
plt.imshow(edge_image_prewitt, cmap='gray')
plt.title('Prewitt Edge Image')

plt.suptitle('Figure 1: Roberts, Sobel, Prewitt')
plt.show()

# Toán tử Gradient dùng mặt nạ Frei-Chen
# Định nghĩa mặt nạ Frei-Chen
frei_chen_mask_x = np.array([[-1, -np.sqrt(2), -1], [0, 0, 0], [1, np.sqrt(2), 1]]) #Cho X-axis
frei_chen_mask_y = np.array([[-1, 0, 1], [-np.sqrt(2), 0, np.sqrt(2)], [-1, 0, 1]]) #Cho Y-axis
# Dùng mặt nạ Frei-Chen theo chiều ngang và dọc
frei_chen_x = cv2.filter2D(img, cv2.CV_64F, frei_chen_mask_x) #X-axis
frei_chen_y = cv2.filter2D(img, cv2.CV_64F, frei_chen_mask_y) #Y-axis
edge_image_frei_chen = np.sqrt(frei_chen_x**2 + frei_chen_y**2)# Tính độ lớn của đạo hàm
# Chuyển đổi giá trị độ lớn về định dạng uint8 để có thể hiển thị
edge_image_frei_chen = np.uint8(edge_image_frei_chen)


# Toán tử Laplace
laplace = cv2.Laplacian(img, cv2.CV_64F)# Dùng mặt nạ Laplace
# Chuyển đổi giá trị độ lớn về định dạng uint8 để có thể hiển thị
laplace = np.uint8(np.abs(laplace))

# Toán tử Laplace of Gaussian
img_smoothed = cv2.GaussianBlur(img, (5, 5), 0) # Dùng bộ lọc Gaussian
laplacian = cv2.Laplacian(img_smoothed, cv2.CV_64F) # Dùng mặt nạ Laplace
# Chuyển đổi giá trị độ lớn về định dạng uint8 để có thể hiển thị
laplacian = np.uint8(np.abs(laplacian))





plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.imshow(imgColor, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(edge_image_frei_chen, cmap='gray')
plt.title('Frei-chen Edge Image')

plt.subplot(2, 2, 3)
plt.imshow(laplace, cmap='gray')
plt.title('Laplace Edge Image')

plt.subplot(2, 2, 4)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplace of Gaussian Edge Image')

plt.suptitle('Figure 2: Frei-chen, Laplace, Laplace of Gaussian')
plt.show()


# Áp dụng phương pháp Canny
edge_image_canny = cv2.Canny(img, 50, 150)

plt.subplot(1, 2, 1)
plt.imshow(imgColor, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(edge_image_canny, cmap='gray')
plt.title('Canny Edge Image')

plt.suptitle('Figure 3: Original & Canny')
plt.show()
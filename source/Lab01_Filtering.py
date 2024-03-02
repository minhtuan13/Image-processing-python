import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
#Làm trơn ảnh dựa trên miền không gian 

img = cv2.imread('ex.png', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#1 Toán tử trung bình(average filter)


def average_filter(image, kernel_size):
    # Lấy kích thước ảnh
    height, width, channels = image.shape

    # Khởi tạo ảnh mới
    blurred_image = np.zeros_like(image, dtype=np.float32)

    # Tạo kernel 3x3 với giá trị trung bình
    kernel_height, kernel_width = kernel_size, kernel_size 
    kernel = np.ones((kernel_height, kernel_width), np.float32) / (kernel_height * kernel_width)

    # Áp dụng tích chập để làm trơn ảnh
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]
            # Tích chập giá trị trung bình vào pixel tại vị trí (i, j) sử dụng kernel
            blurred_image[i, j] = np.sum(region * kernel, axis=(0, 1)).astype(np.uint8)
    # Chuẩn hóa ảnh trơn về kiểu uint8 để hiển thị
    blurred_image = np.uint8(blurred_image)
    return blurred_image

# img_dst = average_blur_temp(img, 3)
# plt.imshow(img_dst)
# plt.title("Average Filter")
# plt.show()
# cv2.waitKey(0)


#2 Toán tử trung vị (Median) 3x3

# def median_filter1(image, kernel_size):
#     height, width, channels = image.shape
#     kernel_height, kernel_width = kernel_size, kernel_size
#     result_image = np.zeros_like(image, dtype=np.uint8)

#     for c in range(channels):
#         for i in range(1, height - 1):
#             for j in range(1, width - 1):
#                 # Lấy phần ảnh tương ứng với kernel
#                 neighbors = [image[i - 1, j - 1, c], 
#                              image[i - 1, j, c], 
#                              image[i - 1, j + 1, c],
#                              image[i, j - 1, c], 
#                              image[i, j, c], 
#                              image[i, j + 1, c],
#                              image[i + 1, j - 1, c], 
#                              image[i + 1, j, c], 
#                              image[i + 1, j + 1, c]]

#                 # Lấy giá trị trung vị
#                 #result_image[i, j, c] = np.median(neighbors).astype(np.uint8)
#                 sorted_array = np.sort(neighbors)
#                 median_value = sorted_array[len(sorted_array) // 2]
#                 result_image[i, j, c] = median_value
#     return result_image



#2 Toán tử trung vị (Median)
def median_filter(image, kernel_size):
    height, width, channels = image.shape
    kernel_height = kernel_width = kernel_size
    result_image = np.zeros_like(image, dtype=np.uint8)

    for c in range(channels):
        for i in range(kernel_height // 2, height - kernel_height // 2):
            for j in range(kernel_width // 2, width - kernel_width // 2):
                # Lấy phần ảnh tương ứng với kernel
                neighbors = []
                for m in range(-kernel_height // 2, kernel_height // 2 + 1):
                    for n in range(-kernel_width // 2, kernel_width // 2 + 1):
                        neighbors.append(image[i + m, j + n, c])

                # Lấy giá trị trung vị
                 #result_image[i, j, c] = np.median(neighbors).astype(np.uint8)
                sorted_array = np.sort(neighbors)
                median_value = sorted_array[len(sorted_array) // 2]
                result_image[i, j, c] = median_value

    return result_image


#3 Toán tử Gaussian

#Tạo kernel Gaussian.
def gaussian_kernel(size, sigma=1.0):
    
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    return kernel / np.sum(kernel)

#Hàm tích chập 2D
def convolve2D(image, kernel):
 
    image_height, image_width, channels = image.shape
    kernel_height, kernel_width = kernel.shape
    output = np.zeros_like(image)

    # Lật kernel để chuẩn bị cho phép tích chập
    kernel = np.flipud(np.fliplr(kernel))

    # Padding ảnh với zeros
    padded_image = np.pad(image, ((kernel_height//2, kernel_height//2), (kernel_width//2, kernel_width//2), (0, 0)), mode='constant')

    # Thực hiện phép tích chập
    for i in range(image_height):
        for j in range(image_width):
            for c in range(channels):
                output[i, j, c] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width, c] * kernel)

    return output

def gaussian_filter(image, kernel_size, sigma):
    # Tạo kernel Gaussian
    kernel = gaussian_kernel(kernel_size, sigma)

    # Áp dụng phép tích chập cho từng kênh màu
    blurred_image = convolve2D(image, kernel)

    return blurred_image

img_dst = average_filter(img, kernel_size = 3)
img_dst2 = median_filter(img, kernel_size = 5)
img_dst3 = gaussian_filter(img, kernel_size = 5, sigma = 1.0)


plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Ảnh nguồn')

plt.subplot(1, 4, 2)
plt.imshow(img_dst, cmap='gray')
plt.title("Average Filter")

plt.subplot(1, 4, 3)
plt.imshow(img_dst2, cmap='gray')
plt.title("Median Filter")

plt.subplot(1, 4, 4)
plt.imshow(img_dst3, cmap='gray')
plt.title("Gaussian Filter")

plt.suptitle('Figure 1: Làm trơn ảnh dùng giải thuật theo lý thuyết')

plt.show()



#Làm trơn ảnh dùng hàm trong OpenCV
# Mean blur
meanBlur = cv2.blur(img,(3,3))  
# Median Blur
median_blur = cv2.medianBlur(img,5) 
# Guassian Blur
guassian_blur = cv2.GaussianBlur(img,(5,5), 0, 0) 
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.title('Ảnh nguồn')

plt.subplot(1, 4, 2)
plt.imshow(meanBlur, cmap='gray')
plt.title('Average Blur')

plt.subplot(1, 4, 3)
plt.imshow(median_blur, cmap='gray')
plt.title('Median Blur')

plt.subplot(1, 4, 4)
plt.imshow(guassian_blur, cmap='gray')
plt.title('Guassian Blur')

plt.suptitle('Figure 2: Làm trơn ảnh dùng hàm trong OpenCV')

plt.show()
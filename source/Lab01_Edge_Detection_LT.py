import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

img = cv2.imread('house.png', cv2.IMREAD_GRAYSCALE)

imgColor = cv2.imread('house.png', cv2.IMREAD_COLOR)
imgColor = cv2.cvtColor(imgColor, cv2.COLOR_BGR2RGB)


def convolve(img, kernel):
     # Lấy kích thước ảnh và kernel
    img_row, img_col = img.shape
    kernel_row, kernel_col = kernel.shape

    # Tạo một ma trận kết quả với kích thước giống với ảnh đầu vào
    output = np.zeros(img.shape)

    # Thiết lập padding cho ảnh để tích chập với kernel
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_img = np.zeros((img_row + (2 * pad_height), img_col + (2 * pad_width)))
    padded_img[pad_height:padded_img.shape[0] - pad_height, pad_width:padded_img.shape[1] - pad_width] = img

    # # Áp dụng phép tích chập
    for row in range(img_row):
        for col in range(img_col):
            output[row, col] = np.sum(kernel * padded_img[row:row + kernel_row, col:col + kernel_col]) / (kernel.shape[0] * kernel.shape[1])
    
    return output




def roberts_edge_detection(image):

    roberts_mask_x = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
    roberts_mask_y = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])

    # Áp dụng mặt nạ Roberts theo chiều ngang và dọc bằng hàm convolve
    roberts_x = convolve(image, roberts_mask_x)
    roberts_y = convolve(image, roberts_mask_y)

    # Tính độ lớn của đạo hàm
    edge_roberts = np.sqrt(roberts_x**2 + roberts_y**2)

    # Chuyển đổi giá trị độ lớn về định dạng uint8 để có thể hiển thị
    edge_roberts = np.uint8(edge_roberts)

    return edge_roberts


def sobel_edge_detection(image):
    sobel_mask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_mask_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Áp dụng mặt nạ Sobel theo chiều ngang và dọc bằng hàm convolve
    sobel_x = convolve(image, sobel_mask_x)
    sobel_y = convolve(image, sobel_mask_y)

    # Tính độ lớn của đạo hàm
    edge_sobel = np.sqrt(sobel_x**2 + sobel_y**2)

    # Chuyển đổi giá trị độ lớn về định dạng uint8 để có thể hiển thị
    edge_sobel = np.uint8(edge_sobel)

    return edge_sobel


def prewitt_edge_detection(image):
    prewitt_mask_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_mask_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # Áp dụng mặt nạ Prewitt theo chiều ngang và dọc bằng hàm convolve
    prewitt_x = convolve(image, prewitt_mask_x)
    prewitt_y = convolve(image, prewitt_mask_y)

    # Tính độ lớn của đạo hàm
    edge_prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)

    # Chuyển đổi giá trị độ lớn về định dạng uint8 để có thể hiển thị
    edge_prewitt = np.uint8(edge_prewitt)

    return edge_prewitt



def freichen_edge_detection(image, threshold=50):
    # Mặt nạ Frei-Chen cho X-axis và Y-axis
    frei_chen_mask_x = np.array([[-1, -np.sqrt(2), -1], [0, 0, 0], [1, np.sqrt(2), 1]])
    frei_chen_mask_y = np.array([[-1, 0, 1], [-np.sqrt(2), 0, np.sqrt(2)], [-1, 0, 1]])

    # Áp dụng convolve cho X-axis và Y-axis
    result_x = convolve(image, frei_chen_mask_x)
    result_y = convolve(image, frei_chen_mask_y)

    # Tính toán biên cạnh cuối cùng
    final_edge = np.sqrt(result_x**2 + result_y**2)

    # Ngưỡng hóa để xác định biên cạnh
    edge_freichen = (final_edge > threshold).astype(np.uint8) * 255
    return edge_freichen



edge_image_roberts = roberts_edge_detection(img)
edge_image_sobel = sobel_edge_detection(img)
edge_image_prewitt = prewitt_edge_detection(img)


plt.subplot(2, 2, 1)
plt.imshow(imgColor, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(edge_image_roberts, cmap='gray')
plt.title('Roberts Edge Image')

plt.subplot(2, 2, 3)
plt.imshow(edge_image_sobel, cmap='gray')
plt.title('Sobel Edge Image')

plt.subplot(2, 2, 4)
plt.imshow(edge_image_prewitt, cmap='gray')
plt.title('Prewitt Edge Image')

plt.suptitle('Figure 1: Roberts, Sobel, Prewitt (Theo Lý Thuyết)')
plt.show()



def laplace4_edge_detection(image):
    # Bộ lọc toán tử Laplace 3x3
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Áp dụng toán tử Laplace sử dụng hàm convolve
    laplacian_result = convolve(image, laplacian_kernel)

    # Chuyển đổi giá trị độ lớn về định dạng uint8 để có thể hiển thị
    laplacian_result = np.uint8(np.abs(laplacian_result))

    return laplacian_result

def laplace8_edge_detection(image):
    # Bộ lọc toán tử Laplace mở rộng 3x3
    laplacian_kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    # Áp dụng toán tử Laplace sử dụng hàm convolve
    laplacian_result = convolve(image, laplacian_kernel)

    # Chuyển đổi giá trị độ lớn về định dạng uint8 để có thể hiển thị
    laplacian_result = np.uint8(np.abs(laplacian_result))

    return laplacian_result

laplace4 = laplace4_edge_detection(img)
laplace8 = laplace8_edge_detection(img)
edge_freichen = freichen_edge_detection(img, threshold=50)

plt.subplot(2, 2, 1)
plt.imshow(imgColor, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(edge_freichen, cmap='gray')
plt.title('Frei-chen Image')

plt.subplot(2, 2, 3)
plt.imshow(laplace4, cmap='gray')
plt.title('Laplace -4 Edge Image')

plt.subplot(2, 2, 4)
plt.imshow(laplace8, cmap='gray')
plt.title('Laplace -8 Edge Image')

plt.suptitle('Figure 2: Frei-chen & Laplace 2 bộ lọc (Theo Lý Thuyết)')
plt.show()

def non_maximal_suppression(edges, gradient_direction):
    rows, cols = edges.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = gradient_direction[i, j]

            # Chuyển đổi góc về giá trị dương
            if angle < 0:
                angle += np.pi

            # Chuyển đổi góc về giá trị nguyên tố nhỏ nhất (0, 45, 90, 135 độ)
            angle = (angle * 180 / np.pi) % 180
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [edges[i, j-1], edges[i, j+1]]
            elif (22.5 <= angle < 67.5):
                neighbors = [edges[i-1, j-1], edges[i+1, j+1]]
            elif (67.5 <= angle < 112.5):
                neighbors = [edges[i-1, j], edges[i+1, j]]
            elif (112.5 <= angle < 157.5):
                neighbors = [edges[i-1, j+1], edges[i+1, j-1]]

            if edges[i, j] < max(neighbors):
                edges[i, j] = 0

    return edges


def canny_edge_detection(image, low_threshold, high_threshold, kernel_size=3, sigma=1.0):
   
    # Bước 1: Làm mờ ảnh bằng bộ lọc gaussian
    gaussian_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Bước 2: Tính toán độ lớn và hướng gradient
    gradient_x = cv2.Sobel(gaussian_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gaussian_image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # Bước 3: Thực hiện ngưỡng hai giai đoạn và Non-maximal suppression
    gradient_magnitude_normalized = (gradient_magnitude / np.max(gradient_magnitude) * 255).astype(np.uint8)

    edges = np.zeros_like(gradient_magnitude_normalized)

    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            if gradient_magnitude_normalized[i, j] > high_threshold:
                edges[i, j] = 255
            elif gradient_magnitude_normalized[i, j] > low_threshold and any(
                gradient_magnitude_normalized[i-1:i+2, j-1:j+2].ravel() > high_threshold
            ):
                edges[i, j] = 255

    # Bước 4: Non-maximal suppression
    edges = non_maximal_suppression(edges, gradient_direction)

    return edges



canny_edge = canny_edge_detection(img, low_threshold = 20, high_threshold = 50, kernel_size=3, sigma=1.0)
plt.subplot(1, 2, 1)
plt.imshow(imgColor, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(canny_edge, cmap='gray')
plt.title('Canny Edge Image')
plt.suptitle('Figure 3: Original & Canny (Theo Lý Thuyết)')
plt.show()
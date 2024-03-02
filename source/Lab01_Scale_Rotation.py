import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

srcImg = cv2.imread('house.png', cv2.IMREAD_COLOR)
srcImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2RGB)

#Scale ảnh 
def scale_image(img, scale_factor):
    # Đọc ảnh từ đường dẫn đầu vào
    original_image = np.asarray(img, dtype = int)

    # Lấy kích thước ban đầu của ảnh
    original_height, original_width = original_image.shape[:2]

    # Tính toán kích thước mới dựa trên tỷ lệ scale
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Tạo ảnh mới với kích thước mới
    scaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Scale từng pixel một
    for y in range(new_height):
        for x in range(new_width):
            # Tính toán vị trí tương ứng trên ảnh gốc
            original_x = int(x / scale_factor)
            original_y = int(y / scale_factor)

            # Kiểm tra xem vị trí original_x, original_y có nằm trong phạm vi hợp lý không
            if (0 <= original_x < original_width) and (0 <= original_y < original_height):
                # Lấy giá trị pixel từ ảnh gốc và gán vào ảnh mới
                scaled_image[y, x] = original_image[original_y, original_x]

    return scaled_image


def rotate_image(img, alpha):
    # Đọc ảnh từ đường dẫn đầu vào
    original_image = np.asarray(img, dtype = int)

    # Lấy kích thước ban đầu của ảnh
    original_height, original_width = original_image.shape[:2]

    # Tính toán trung tâm của ảnh để sử dụng làm trung tâm xoay
    center = (original_width // 2, original_height // 2)

    # Chuyển đổi góc alpha từ độ sang radian
    alpha_rad = np.radians(alpha)

    # Tạo ma trận xoay
    rotation_matrix = np.array([[np.cos(alpha_rad), -np.sin(alpha_rad), 0],
                                [np.sin(alpha_rad), np.cos(alpha_rad), 0],
                                [0, 0, 1]])

    # Tạo ảnh mới với kích thước giữ nguyên
    rotated_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    # Xoay từng pixel một
    for y in range(original_height):
        for x in range(original_width):
            # Tính toán vị trí pixel sau khi xoay
            rotated_x, rotated_y, _ = np.dot(rotation_matrix, [x - center[0], y - center[1], 1]) + [center[0], center[1], 0]

            # Lấy giá trị pixel từ ảnh gốc và gán vào ảnh mới
            if 0 <= rotated_x < original_width and 0 <= rotated_y < original_height:
                rotated_image[int(rotated_y), int(rotated_x)] = original_image[y, x]
    

    return rotated_image

scaled_image = scale_image(srcImg, 5)
rotation_image = rotate_image(srcImg, 45)

plt.subplot(1,3,1)
plt.imshow(srcImg)
plt.title("Original Image")

plt.subplot(1,3,2)
plt.imshow(scaled_image)
plt.title("Scaled Image")

plt.subplot(1,3,3)
plt.imshow(rotation_image)
plt.title("Rotation Image")

plt.show()
cv2.waitKey(0)

cv2.destroyAllWindows








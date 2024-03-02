import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

img = cv2.imread('house.png', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2.imshow("Read image ",img)
# cv2.waitKey(0)

#Kiểm tra xem giá trị màu có nằm trong khoảng 0 đến 255 hay không. Nếu vượt ra ngoài thì lấy giá trị giới hạn
def inZone(value):
    if value < 0:
        return 0
    if value > 255:
        return 255
    return value


#Thông số ảnh gồm chiều cao (h), chiều rộng(w), độ sâu (d)
#Lấy kích thước của ảnh để duyệt qua từng pixel khi thực hiện biến đổi
height,width,depth = img.shape
print ("Height:", height) #Chiều Cao
print ("Width:", width) #Chiều Rộng
print ("Depth:", depth) #Đại diện cho số lượng kênh màu của ảnh


#Giải thuật biến đổi màu

#1 Phép biến đổi tuyến tính

#1.1 Thay đổi độ sáng (Brighrness)
def brightness_algo(img):
    h,w,d = img.shape
    brightened_image = np.asarray(img, dtype = int)
    brightness = 50
    # Duyệt qua từng pixel
    for y in range(h):
        for x in range(w):
            for channel in range(d):
                # Tăng độ sáng bằng cách cộng giá trị b vào giá trị pixel
                brightened_image[y, x, channel] = inZone(brightened_image[y, x, channel] + brightness)
    
    return brightened_image


#1.1.2 Thay đổi độ tương phản (Contrast)
def contrast_algo(img):
    h,w,d = img.shape
    contrast = 3
    contrast_image = np.copy(img)
    # Duyệt qua từng pixel 
    for y in range(h):
        for x in range(w):
            for channel in range(d):
                # Tăng độ sáng bằng cách cộng giá trị b vào giá trị pixel
                contrast_image[y, x, channel] = inZone(contrast_image[y, x, channel]*contrast)
    return contrast_image

#1.3 Thay đổi độ tương phản (Brightness + Contrast)
def brightness_contrast_algo(img):
    # Tạo một bản sao của ảnh để không ảnh hưởng đến ảnh gốc
    h,w,d = img.shape
    bn_ct_image = np.asarray(img, dtype = int)

    brightness = 50
    contrast = 3

    # Duyệt qua từng pixel 
    for y in range(h):
        for x in range(w):
            for channel in range(d):
                # Tăng độ sáng bằng cách cộng giá trị b vào giá trị pixel
                bn_ct_image[y, x, channel] = inZone(bn_ct_image[y, x, channel]*contrast + brightness)
    return bn_ct_image

linear_image1 = brightness_algo(img)
linear_image2 = contrast_algo(img)
linear_image3 = brightness_contrast_algo(img)

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Ảnh nguồn')

plt.subplot(2, 2, 2)
plt.imshow(linear_image1, cmap='gray')
plt.title("Brightness")

plt.subplot(2, 2, 3)
plt.imshow(linear_image2, cmap='gray')
plt.title("Contrast")

plt.subplot(2, 2, 4)
plt.imshow(linear_image3, cmap='gray')
plt.title("Brightness & Contrast")

plt.suptitle('Figure 1: Phép biến đổi tuyến tính (Linear)')
plt.show()


#2 Phép biến đổi phi tuyến

img1 = cv2.imread('ex.png', cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

#2.1 Biến đổi theo hàm logarithm
def logarithm_algo(img):
    # Lấy kích thước của ảnh
    h, w, d = img.shape
    
    # Tạo một bản sao của ảnh để không ảnh hưởng đến ảnh gốc
    transformed_image = np.asarray(img, dtype=int)

    # Duyệt qua từng pixel và kênh màu
    for y in range(h):
        for x in range(w):
            for channel in range(d):
                c = 255 / np.log(1 + np.max(img))
                transformed_image[y, x, channel] = inZone(c * (np.log(img[y, x, channel] + 1)))

    # Chuyển đổi độ phân giải của ảnh trở lại thành 8-bit
    final_image = np.array(transformed_image, dtype=np.uint8)
    
    return final_image

#2.2 Biến đổi theo hàm e mũ

def exponential_algo(image):
    scale = 0.05 #Chỉnh mức độ biến đổi

    # Tạo một bản sao của ảnh để không ảnh hưởng đến ảnh gốc
    transformed_image = np.asarray(image, dtype=int)

    # Lấy kích thước của ảnh
    height, width, channels = image.shape

    # Duyệt qua từng pixel và kênh màu
    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                pixel_value = image[y, x, channel]
                transformed_pixel = inZone(np.exp(scale * pixel_value))
                transformed_image[y, x, channel] = transformed_pixel

    return transformed_image

nonlinear_image1 = logarithm_algo(img1)
nonlinear_image2 = exponential_algo(img1)

plt.subplot(1, 3, 1)
plt.imshow(img1, cmap='gray')
#plt.imshow(image, cmap='gray')
plt.title('Ảnh gốc')

plt.subplot(1, 3, 2)
plt.imshow(nonlinear_image1, cmap='gray')
#plt.imshow(image, cmap='gray')
plt.title('Biến đổi theo hàm Logarithm')

plt.subplot(1, 3, 3)
plt.imshow(nonlinear_image2, cmap='gray')
#plt.imshow(image, cmap='gray')
plt.title('Biến đổi theo hàm Exponential')
plt.suptitle('Figure 2: Phép biến đổi phi tuyến (Nonlinear)')

plt.show()



#3 Phép biến đổi dựa trên phân bố xác suất 
#3.1 Cân bằng lược đồ xám (Histogram Equalization)

img2 = cv2.imread('cold.png', cv2.IMREAD_GRAYSCALE)

def histogram_equalization(image):
    # B1: Khởi tạo mảng H chiều dài nG với giá trị 0
    nG = 256
    H = np.zeros(nG, dtype=int)

    # B2: Tính lược đồ độ xám của ảnh f, lưu vào H H[f(x, y)] += 1
    height, width = image.shape[:2]
    for y in range(height):
        for x in range(width):
            pixel_value = image[y, x]
            H[pixel_value] += 1

    # B3: Tính lược đồ độ xám tích lũy của f, lưu vào T
    T = np.cumsum(H)

    # B4: Chuẩn hóa T về đoạn [0, nG-1]
    normalized_T = np.round((nG - 1) * T / (height * width)).astype(int)

    # B5: Tạo ảnh kết quả g(x, y) = T(f(x, y))
    equalized_image = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            pixel_value = image[y, x]
            equalized_image[y, x] = normalized_T[pixel_value]

    return equalized_image

#dùng cho ảnh màu
# def histogram_equalization_color(image):
#     equalized_channels = [histogram_equalization(channel) for channel in cv2.split(image)]
#     equalized_image = cv2.merge(equalized_channels)
#     return equalized_image


# Áp dụng hàm cân bằng histogram
equalized_image = histogram_equalization(img2)

# Hiển thị ảnh gốc và ảnh sau khi cân bằng histogram
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(img2, cmap='gray')
plt.title('Ảnh gốc')

plt.subplot(2, 2, 2)
plt.hist(img2.flatten(), bins=256, range=[0, 256], color='black', alpha=0.7)
plt.title('Histogram gốc')


plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Ảnh sau khi cân bằng histogram')

plt.subplot(2, 2, 4)
plt.hist(equalized_image.flatten(), bins=256, range=[0, 256], color='black', alpha=0.7)
plt.title('Histogram sau cân bằng')
plt.suptitle('Figure 3: Cân bằng lược đồ xám (Histogram Equalization)')


plt.tight_layout()
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
#Đặc tả lược đồ xám (Histogram Specification)

def initialize_histogram(nG):
    return np.zeros(nG, dtype=int)

def calculate_histogram(image, Hf):
    height, width = image.shape[:2]

    for y in range(height):
        for x in range(width):
            pixel_value = image[y, x]
            Hf[pixel_value] += 1

def cumulative_histogram(Hf):
    T = np.zeros_like(Hf)
    T[0] = Hf[0]

    for p in range(1, len(Hf)):
        T[p] = T[p-1] + Hf[p]

    return T

def normalize_histogram_T(T, height, width, nG):
    normalized_T = np.round((nG - 1) * T / (height * width)).astype(int)
    return normalized_T

def cumulative_histogram_Hg(Hg):
    G = np.zeros_like(Hg)
    G[0] = Hg[0]

    for p in range(1, len(Hg)):
        G[p] = G[p-1] + Hg[p]

    return G

def normalize_histogram_G(G, height, width, nG):
    normalized_G = np.round((nG - 1) * G / (height * width)).astype(int)
    return normalized_G

def create_result_image(image, normalized_T, normalized_G):
    result_image = np.zeros_like(image)

    height, width = image.shape[:2]

    for y in range(height):
        for x in range(width):
            pixel_value = image[y, x]
            result_image[y, x] = normalized_G[normalized_T[pixel_value]]

    return result_image



image_f = cv2.imread("house.png", cv2.IMREAD_GRAYSCALE)
image_g = cv2.imread("cold.png", cv2.IMREAD_GRAYSCALE)

def histogram_specification(image_f, image_g):
    nG = 256  # Số mức xám thường là 256 (8-bit ảnh)

    # B1: Khởi tạo mảng Hf chiều dài nG với giá trị 0
    Hf = initialize_histogram(nG)

    # B2: Tính lược đồ độ xám của ảnh f, lưu vào Hf
    calculate_histogram(image_f, Hf)

    # B3: Tính lược đồ xám tích lũy của f, lưu vào T
    T = cumulative_histogram(Hf)

    # B4: Chuẩn hóa T về đoạn [0, nG-1]
    normalized_T = normalize_histogram_T(T, image_f.shape[0], image_f.shape[1], nG)

    # B5: Tính lược đồ xám tích lũy của g, lưu vào Hg
    Hg = initialize_histogram(nG)
    calculate_histogram(image_g, Hg)

    # B6: Tính lược đồ xám tích lũy của g, lưu vào G
    G = cumulative_histogram_Hg(Hg)

    # B7: Chuẩn hóa G về đoạn [0, nG-1]
    normalized_G = normalize_histogram_G(G, image_g.shape[0], image_g.shape[1], nG)

    # B8: Tạo ảnh kết quả g
    result_image = create_result_image(image_f, normalized_T, normalized_G)
    return result_image


result_image = histogram_specification(image_f, image_g)
plt.subplot(2, 3, 1)
plt.imshow(image_f, cmap='gray')
plt.title('Ảnh f')

plt.subplot(2, 3, 4)
plt.hist(image_f.flatten(), bins=256, range=[0, 256], color='black', alpha=0.7)
plt.title('Lược đồ ảnh f')

plt.subplot(2, 3, 2)
plt.imshow(image_g, cmap='gray')
plt.title('Ảnh g')

plt.subplot(2, 3, 5)
plt.hist(image_g.flatten(), bins=256, range=[0, 256], color='black', alpha=0.7)
plt.title('Lược đồ ảnh g')

plt.subplot(2, 3, 3)
plt.imshow(result_image, cmap='gray')
plt.title('Ảnh kết quả đặc tả')

plt.subplot(2, 3, 6)
plt.hist(result_image.flatten(), bins=256, range=[0, 256], color='black', alpha=0.7)
plt.title('Lược đồ ảnh kết quả')

plt.suptitle('Figure 4: Đặc tả lược đồ xám (Histogram Specification)')
plt.show()
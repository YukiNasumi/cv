import cv2
import numpy as np

def compute_optical_flow(image1_path, image2_path):
    # 读取两张图片
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    img1 = cv2.resize(img1,(448,448))
    img2 = cv2.resize(img2,(448,448))
    cv2.imshow('img1',img1)
    cv2.waitKey(1000)
    #cv2.destroyAllWindows()
    cv2.imshow('img2',img2)
    cv2.waitKey(1000)
    #cv2.destroyAllWindows()
    
    # 将图片转换为灰度图像
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 计算光流的幅度和角度
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 创建HSV图像来表示光流
    hsv = np.zeros_like(img1)
    hsv[..., 1] = 255

    # 设置色调（H）为角度
    hsv[..., 0] = ang * 180 / np.pi / 2

    # 设置亮度（V）为光流的幅度
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # 将HSV图像转换为BGR图像以便显示
    bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr_flow

# 输入图片路径
image1_path = 'clip1/image/00001.png'
image2_path = 'clip1/image/00002.png'

# 计算光流图
flow_image = compute_optical_flow(image1_path, image2_path)

# 显示光流图
cv2.imshow('Optical Flow', flow_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存光流图
cv2.imwrite('optical_flow_output.jpg', flow_image)

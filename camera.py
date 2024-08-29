import cv2
import tools
# 打开摄像头（设备索引通常为 0）
cap = cv2.VideoCapture(0)
import torch
model = torch.hub.load('ultralytics/yolov5','yolov5s')
def frame_process(frame):
    img = tools.toRGB(frame)
    result = model(img)
    return tools.toRGB(result.render()[0])
while True:
    # 从摄像头捕获一帧图像
    ret, frame = cap.read()

    # 如果成功捕获图像，则显示图像
    if ret:
        
        cv2.imshow('Video Feed', frame_process(frame))

    # 等待用户按键
    key = cv2.waitKey(1) & 0xFF

    # 如果按下 'q' 键，退出循环
    if key == ord('q'):
        break

# 释放摄像头资源
cap.release()
# 关闭所有窗口
cv2.destroyAllWindows()

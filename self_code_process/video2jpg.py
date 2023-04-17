import cv2

# 打开视频文件
video = cv2.VideoCapture('/home/chenzhen/data/video/hangzhou.mp4')

# 读取第一帧
success, image = video.read()
count = 0

# 循环读取每一帧并保存为图像
while success:
    # 保存图像
    cv2.imwrite('/home/chenzhen/data/image/hangzhoou/frame{}.jpg'.format(count), image)

    # 读取下一帧
    success, image = video.read()
    count += 1

# 释放资源
video.release()

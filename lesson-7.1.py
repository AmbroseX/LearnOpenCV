import cv2
import numpy as np

if __name__ == '__main__':
    video = cv2.VideoCapture('G:/Data/Video/flay.mp4')
    video_save = cv2.VideoWriter(filename= 'gray.mp4',
                                 fourcc = cv2.VideoWriter_fourcc(*'MP4V'), # mp4格式
                                 fps=24, # 帧率
                                 frameSize=(1920,1080)
                                 )
    
    face_detector = cv2.CascadeClassifier('./data/haarcascade_frontalface_alt2.xml')
    
    while True:
        retval,image = video.read()
        image = cv2.resize(image,(1920,1080))
        gray = cv2.cvtColor(image,code=cv2.COLOR_RGB2GRAY)
        faces = face_detector.detectMultiScale(gray,
                                               scaleFactor = 1.1,
                                               minNeighbors=3
                                               ) # 扫描增长图片
        
        if retval == False:  # 最后一张
            print('视频读取完成,没有图片!')
            break
        for x,y,w,h in faces:
            cv2.rectangle(image,
                    pt1=(x,y),
                    pt2=(x+w,y+h),
                    color = [0,0,255],
                    thickness=2)  # 绘制矩形
        video_save.write(image)
        cv2.imshow('face',image)
        key = cv2.waitKey(10) # 等待1ms
        if key ==ord('q'):
            print('退出')
            break
    video.release()  # 释放内存
    video_save.release()
    cv2.destroyAllWindows()
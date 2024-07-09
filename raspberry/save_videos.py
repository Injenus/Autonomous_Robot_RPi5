
from picamera2 import Picamera2
import time
import cv2


wh_sign, wh_road = (1280, 720), (640, 360)
cam_road = Picamera2(1)
cam_sign = Picamera2(0)
camera_config = cam_road.create_preview_configuration(main={"format": "BGR888", "size": wh_road})
cam_road.configure(camera_config)
camera_config = cam_sign.create_preview_configuration(main={"format": "BGR888", "size": wh_sign})
cam_sign.configure(camera_config)
cam_road.set_controls({
        "AwbEnable": 0,  # Отключение автоматического баланса белого
        "ColourGains": (1.4, 1.5),  # Установка коэффициентов цветовой коррекции
        "AeEnable": 0,  # Отключение автоматической экспозиции
        "AnalogueGain": 2.0,  # Установка аналогового усиления (чувствительности)
        "ExposureTime": 35000  # Установка выдержки в микросекундах (например, 20000 = 20мс)
    })
cam_sign.set_controls({
        "AwbEnable": 0,  # Отключение автоматического баланса белого
        "ColourGains": (1.4, 1.5),  # Установка коэффициентов цветовой коррекции
        "AeEnable": 0,  # Отключение автоматической экспозиции
        "AnalogueGain": 2.0,  # Установка аналогового усиления (чувствительности)
        "ExposureTime": 35000  # Установка выдержки в микросекундах (например, 20000 = 20мс)
    })
cam_sign.start()
cam_road.start()


fps = 20
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для записи видео
out_road = cv2.VideoWriter('out_road.mp4', fourcc, fps, wh_road)
out_sign = cv2.VideoWriter('out_sign.mp4', fourcc, fps, wh_sign)


prev_time_road = time.time()
fps_road = 0
prev_time_sign = time.time()
fps_sign = 0


while True:
    frame_road = cam_road.capture_array()
    frame_road = cv2.rotate(frame_road, cv2.ROTATE_180)
    out_road.write(frame_road)
    
    current_time_road = time.time()
    elapsed_time_road = current_time_road - prev_time_road
    prev_time_road = current_time_road
    fps_road = int(1 / elapsed_time_road)
    print(f'ROAD {fps_road} fps')
    cv2.imshow('Road', frame_road)
#######
    frame_sign = cam_sign.capture_array()
    frame_sign = cv2.rotate(frame_sign, cv2.ROTATE_180)
    out_sign.write(frame_sign)
    
    current_time_sign = time.time()
    elapsed_time_sign = current_time_sign - prev_time_sign
    prev_time_sign = current_time_sign
    fps_sign = int(1 / elapsed_time_sign)
    print(f'SIGN {fps_sign} fps')
    cv2.imshow('Sign', frame_sign)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_road.stop()
cam_sign.stop()
out_road.release()
out_sign.release()
cv2.destroyAllWindows()









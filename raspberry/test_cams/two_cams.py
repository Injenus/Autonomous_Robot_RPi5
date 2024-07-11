from picamera2 import Picamera2
from time import sleep

# Функция для захвата изображения с заданной камеры
def capture_image(camera_num, file_name):
    cam = Picamera2(camera_num=camera_num)
    cam.start()
    sleep(0.1)  # Небольшая задержка для настройки экспозиции
    cam.capture_file(file_name)
    cam.stop()

# Захват и сохранение изображения с первой камеры
capture_image(0, "camera0.jpg")

# Захват и сохранение изображения со второй камеры
capture_image(1, "camera1.jpg")

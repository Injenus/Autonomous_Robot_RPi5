import cv2
import numpy as np

# Исходные данные
frame_width = 2592
frame_height = 1944
focal_length_mm = 0.87  # фокусное расстояние в мм
sensor_diagonal_mm = 4  # диагональ сенсора в мм
aspect_ratio = 4/3  # соотношение сторон кадра

# Расчеты размеров сенсора
sensor_width_mm = 3.39
sensor_height_mm = 2.54

# Размеры пикселей на сенсоре
pixel_width_mm = sensor_width_mm / frame_width
pixel_height_mm = sensor_height_mm / frame_height

print(pixel_width_mm, pixel_height_mm)

# Фокусное расстояние в пикселях
f_x = focal_length_mm / pixel_width_mm
f_y = focal_length_mm / pixel_height_mm

# Оптический центр
c_x = 1080 / 2 
c_y = 720 / 2
#c_y +=20
# Матрица камеры
camera_matrix = np.array([[f_x, 0, c_x],
                          [0, f_y, c_y],
                          [0, 0, 1]], dtype=np.float32)

# Коэффициенты дисторсии (искусственные для примера)
print(camera_matrix)
k1 = -0.3  # Параметр радиального искажения
dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)

# Чтение изображения с камеры
cap = cv2.VideoCapture('1080.mp4')
ret, frame = cap.read()

# Коррекция дисторсии
h, w = frame.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

# Отображение результата
cv2.imshow('Original', frame)
cv2.imshow('Undistorted', undistorted_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()

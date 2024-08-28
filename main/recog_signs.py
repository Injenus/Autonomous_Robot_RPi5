import cv2
import numpy as np

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

# Фокусное расстояние в пикселях
f_x = focal_length_mm / pixel_width_mm
f_y = focal_length_mm / pixel_height_mm

# Оптический центр
c_x = 1080 / 2
c_y = 720 / 2
camera_matrix = np.array([[f_x, 0, c_x],
                          [0, f_y, c_y],
                          [0, 0, 1]], dtype=np.float32)

# Коэффициенты дисторсии (искусственные для примера)
k1 = -0.3  # Параметр радиального искажения
dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)


def distor_corr(frame):
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(
        frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undistorted_frame


def nothing(x):
    pass


# Open the video file
cap = cv2.VideoCapture('1080.mp4')

# Variable to control the pause state
is_paused = False
current_frame_index = 0


def read_frame(frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    return ret, frame


start_frame = 550
end_frame = 3000
# Read the first frame at the start_frame index
current_frame_index = start_frame
ret, frame = read_frame(current_frame_index)

while True:
    if not is_paused:
        # Read a frame from the video capture
        ret, frame = read_frame(current_frame_index)
        if not ret or current_frame_index > end_frame:
            break

        current_frame_index += 1
    else:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):  # Pause/resume on 'p' key
            is_paused = not is_paused
        continue

    frame = distor_corr(frame)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1.5, minDist=30,
                               param1=50, param2=0.96,
                               minRadius=18, maxRadius=200)

    # Create a mask to highlight the detected circles
    mask = np.zeros_like(frame)
    blured_zones = frame.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:

            side = int(i[2]/2**0.5*1)
            mean_values = np.mean(
                blured_zones[i[1]-side:i[1]+side, i[0]-side:i[0]+side], axis=(0, 1))

            if mean_values[0] > 105 and mean_values[0] > mean_values[1] and mean_values[0] > mean_values[2] and mean_values[1] < 100 and mean_values[2] < 90:
                print('c', mean_values)
                # Draw the outer circle
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)

                sign_type = 'None'
                sub_side = int(side * 0.85)

                roi = blured_zones[max(
                    1, i[1]-sub_side):i[1]+sub_side, max(1, i[0]-sub_side):i[0]+sub_side]
                roi_g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_g = cv2.medianBlur(roi_g, 3)

                roi_bin = np.where(roi_g >= 100, 255, 0).astype(np.uint8)
                roi_bin = cv2.dilate(roi_bin, np.ones(
                    (5, 5), np.uint8), iterations=1)

                white_pixels = np.argwhere(roi_bin == 255)
                # print(white_pixels)

                # Вычисляем центр масс белых пикселей
                if len(white_pixels) > 0:
                    center_of_mass = white_pixels.mean(axis=0)
                    x, y = int(center_of_mass[0]), int(center_of_mass[1])
                else:
                    pass

                roi_c = np.stack((roi_bin,) * 3, axis=-1)
                blured_zones[max(1, i[1]-sub_side):i[1]+sub_side,
                             max(1, i[0]-sub_side):i[0]+sub_side] = roi_c

                left_p = (i[0]-3*sub_side//4, i[1]+3*sub_side//4)
                middle_p = (i[0], i[1]+3*sub_side//4)
                right_p = (i[0]+3*sub_side//4, i[1]+3*sub_side//4)

                # print(blured_zones[left_p[1]][left_p[0]][0], blured_zones[middle_p[1]][middle_p[0]][0], blured_zones[right_p[1]][right_p[0]][0])
                l, m, r = blured_zones[left_p[1]][left_p[0]][0], blured_zones[middle_p[1]
                                                                              ][middle_p[0]][0], blured_zones[right_p[1]][right_p[0]][0]
                if l == 255 and m == 0 and r == 0:
                    sign_type = 'Right'
                elif l == 0 and m == 255 and r == 0:
                    sign_type = 'Straight'
                elif l == 0 and m == 0 and r == 255:
                    sign_type = 'Left'

                cv2.circle(blured_zones, left_p, 2, (0, 0, 255), -1)
                cv2.circle(blured_zones, middle_p, 2, (0, 0, 255), -1)
                cv2.circle(blured_zones, right_p, 2, (0, 0, 255), -1)

                # x, y, r = i[0], i[1], i[2]
                cv2.putText(blured_zones, f'{sign_type}', (
                    i[0]-i[2], i[1]-i[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                print('fuck', mean_values)
                pass

    cv2.imshow('Original', blured_zones)
    cv2.imshow('Signs', cv2.bitwise_and(frame, mask))

    # Handle key events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):  # Pause/resume on 'p' key
        is_paused = not is_paused
    elif is_paused:
        if key == ord('n'):  # Move to next frame
            current_frame_index += 1
            ret, frame = read_frame(current_frame_index)
            if not ret:
                current_frame_index -= 1  # Stay at the last frame if end is reached
        elif key == ord('b'):  # Move to previous frame
            current_frame_index -= 1
            if current_frame_index < 0:
                current_frame_index = 0  # Stay at the first frame if beginning is reached
            ret, frame = read_frame(current_frame_index)

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

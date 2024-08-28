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
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undistorted_frame

def nothing(x):
    pass

# Open the video file
cap = cv2.VideoCapture('1080.mp4')

# Variable to control the pause state
is_paused = False
current_frame_index = 0

# Set the range of frames to process
start_frame = 120
end_frame = 3000

# Function to read a specific frame
def read_frame(frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    return ret, frame

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
    frame = distor_corr(frame)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=30,
                               param1=150, param2=0.96,
                               minRadius=15, maxRadius=200)

    # Create a mask to highlight the detected circles
    mask = np.zeros_like(frame)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)

    # Show the original frame and the frame with detected circles and ovals
    masked = cv2.bitwise_and(frame, mask)
    blured = cv2.medianBlur(masked, 7)

    cv2.imshow('Original', frame)
    cv2.imshow('Detected Objects', cv2.bitwise_and(blured, mask))

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
            if not ret or current_frame_index > end_frame:
                current_frame_index -= 1  # Stay at the last frame if end is reached
        elif key == ord('b'):  # Move to previous frame
            current_frame_index -= 1
            if current_frame_index < start_frame:
                current_frame_index = start_frame  # Stay at the first frame if beginning is reached
            ret, frame = read_frame(current_frame_index)

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()


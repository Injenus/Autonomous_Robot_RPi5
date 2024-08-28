import cv2
import numpy as np
import time

crop_top, crop_bott, crop_w = 120, 400, 150
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
c_x = 1080 / 2 - crop_w
c_y = 720 / 2 - crop_top
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

def av_col_3x3(image, point):
    x, y = point

    height, width, _ = image.shape
    x_min = max(0, x - 2)
    x_max = min(width - 2, x + 2)
    y_min = max(0, y - 2)
    y_max = min(height - 2, y + 2)

    square_3x3 = image[y_min:y_max + 2, x_min:x_max + 2][0]
    mean_color = np.mean(square_3x3)
    
    return mean_color

def average_colors_bottom_strip(image):
 
    height, width, _ = image.shape
    strip_height = 3
    

    bottom_strip = image[height - strip_height:height, :, 0]
    
    part_width = width // 3
    
    mean_colors = []
    for i in range(3):
        part = bottom_strip[:, i * part_width: (i + 1) * part_width]
        mean_color = np.mean(part)
        mean_colors.append(mean_color)

    return mean_colors

def nothing(x):
    pass
# Create a window for the trackbars
cv2.namedWindow('HSV Adjustments')

# Create trackbars for color change with initial values
cv2.createTrackbar('H Lower', 'HSV Adjustments', 72, 179, nothing)
cv2.createTrackbar('S Lower', 'HSV Adjustments', 0, 255, nothing)
cv2.createTrackbar('V Lower', 'HSV Adjustments', 98, 255, nothing)
cv2.createTrackbar('H Upper', 'HSV Adjustments', 90, 179, nothing)
cv2.createTrackbar('S Upper', 'HSV Adjustments', 192, 255, nothing)
cv2.createTrackbar('V Upper', 'HSV Adjustments', 255, 255, nothing)


# Open the video file
cap = cv2.VideoCapture('1080.mp4')

# Variable to control the pause state
is_paused = False
current_frame_index = 0


def read_frame(frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    return ret, frame


start_frame = 900
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

    frame = frame[crop_top:720-crop_bott, crop_w:1080-crop_w]

    frame = distor_corr(frame)
    
    top_, bott_, left_, right_ = 90, 180, 430, 310
    frame = frame[top_:720-crop_bott-bott_, left_:1080-crop_w-right_]
    #print(frame.shape)
    '''
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 1.0)
    equal = cv2.equalizeHist(gray)
    equal = np.where(equal >= 240, 255, 0).astype(np.uint8)
    cv2.imshow('eq', equal)
    '''
    '''
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(equal, cv2.HOUGH_GRADIENT_ALT, 1.5, minDist=1,
                               param1=50, param2=0.9,
                               minRadius=0, maxRadius=30)

    # Create a mask to highlight the detected circles
    mask = np.zeros_like(frame)
    blured_zones = frame.copy()
    '''
    '''
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:

            side = int(i[2]/2**0.5*1)
            mean_values = np.mean(
                blured_zones[i[1]-side:i[1]+side, i[0]-side:i[0]+side], axis=(0, 1))

            if 1 or mean_values[0] > 90 and mean_values[0] > mean_values[1] and mean_values[0] > mean_values[2] and mean_values[1] < 110 and mean_values[2] < 90:
                print('c', mean_values)
                # Draw the outer circle
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)

                sign_type = 'None'
                sub_side = int(side * 0.8)

                roi = blured_zones[max(
                    1, i[1]-sub_side):i[1]+sub_side, max(1, i[0]-sub_side):i[0]+sub_side]
                roi_g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_g = cv2.resize(roi_g, (roi_g.shape[0]*2,roi_g.shape[0]*2))
                sub_side *=2
                roi_g = cv2.medianBlur(roi_g, 3)

                roi_bin = np.where(roi_g >= 80, 255, 0).astype(np.uint8)
                roi_bin = cv2.dilate(roi_bin, np.ones(
                    (4, 4), np.uint8), iterations=1)

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
                # l, m, r = blured_zones[left_p[1]][left_p[0]][0], blured_zones[middle_p[1]
                #                                                               ][middle_p[0]][0], blured_zones[right_p[1]][right_p[0]][0]
                # if l == 255 and m == 0 and r == 0:
                #     sign_type = 'Right'
                # elif l == 0 and m == 255 and r == 0:
                #     sign_type = 'Straight'
                # elif l == 0 and m == 0 and r == 255:
                #     sign_type = 'Left'
                # h_th, l_th = 144, 112
                # if av_col_3x3(blured_zones, left_p) > h_th and av_col_3x3(blured_zones, middle_p) < l_th and av_col_3x3(blured_zones, right_p) < l_th:
                #     sign_type = 'Right'
                # if av_col_3x3(blured_zones, left_p) < l_th and av_col_3x3(blured_zones, middle_p) > h_th and av_col_3x3(blured_zones, right_p) < l_th:
                #     sign_type = 'Straight'
                # if av_col_3x3(blured_zones, left_p) < l_th and av_col_3x3(blured_zones, middle_p) < l_th and av_col_3x3(blured_zones, right_p) > h_th:
                #     sign_type = 'Left'
                l,m,r = av_col_3x3(blured_zones, left_p), av_col_3x3(blured_zones, middle_p), av_col_3x3(blured_zones, right_p)
                print(l,m,r)
                if l > m and l > r:
                    sign_type = 'Right'
                elif m > l and m > r:
                    sign_type = 'Straight'
                elif r > l and r > m:
                    sign_type = 'Left'

                # l,m,r = average_colors_bottom_strip(blured_zones)
                # print(l,m,r)
                # if l > m and l > r:
                #     sign_type = 'Right'
                # elif m > l and m > r:
                #     sign_type = 'Straight'
                # elif r > l and r > m:
                #     sign_type = 'Left'
                

                cv2.circle(blured_zones, left_p, 2, (0, 0, 255), -1)
                cv2.circle(blured_zones, middle_p, 2, (0, 0, 255), -1)
                cv2.circle(blured_zones, right_p, 2, (0, 0, 255), -1)

                # x, y, r = i[0], i[1], i[2]
                cv2.putText(blured_zones, f'{sign_type}', (
                    i[0]-i[2], i[1]-i[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                print(sign_type)

            else:
                print('fuck', mean_values)
                pass
    '''
    
    #cv2.imshow('Original', blured_zones)
    #cv2.imshow('Signs', cv2.bitwise_and(frame, mask))
    #time.sleep(0.1)


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Get current positions of the trackbars
    h_lower = cv2.getTrackbarPos('H Lower', 'HSV Adjustments')
    s_lower = cv2.getTrackbarPos('S Lower', 'HSV Adjustments')
    v_lower = cv2.getTrackbarPos('V Lower', 'HSV Adjustments')
    h_upper = cv2.getTrackbarPos('H Upper', 'HSV Adjustments')
    s_upper = cv2.getTrackbarPos('S Upper', 'HSV Adjustments')
    v_upper = cv2.getTrackbarPos('V Upper', 'HSV Adjustments')

    # Define range of blue color in HSV
    lower_blue = np.array([h_lower, s_lower, v_lower])
    upper_blue = np.array([h_upper, s_upper, v_upper])

    # Create a mask that identifies blue objects
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red rectangles around blue objects
    for contour in contours:
        if 15< cv2.contourArea(contour) < 80:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            if 0.85 < w/h < 1.15: 
                if 6 <= w <= 11 and  6 <= h <= 11: 
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                    print(w,h)
        
        # x, y, w, h = cv2.boundingRect(contour)
        # if 6 <= w <= 11 and  6 <= h <= 11: 
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)

    # Show the original frame and the frame with blue objects highlighted
    
    cv2.imshow('Green Objects', cv2.bitwise_and(frame, frame, mask=mask))

    cv2.imshow('Original', frame)

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

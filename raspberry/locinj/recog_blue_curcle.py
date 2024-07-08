import cv2
import numpy as np

def nothing(x):
    pass


# Open the video file
cap = cv2.VideoCapture('test.mp4')

# Variable to control the pause state
is_paused = False
current_frame_index = 0

def read_frame(frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    return ret, frame


# Setup color settings
# Define range of blue color in HSV
h_lower = 80
s_lower = 42
v_lower = 42
h_upper = 125
s_upper = 255
v_upper = 255
lower_blue = np.array([h_lower, s_lower, v_lower])
upper_blue = np.array([h_upper, s_upper, v_upper])

# Read the first frame
ret, frame = read_frame(current_frame_index)

while True:
    if not is_paused:
        # Read a frame from the video capture
        ret, frame = read_frame(current_frame_index)
        if not ret:
            break

        current_frame_index += 1

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 1.5)


    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1.5, minDist=50,
                               param1=50, param2=0.95,
                               minRadius=20, maxRadius=200)

    # Create a mask to highlight the detected circles
    mask = np.zeros_like(frame)
    blured_zones = frame.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:

            side = int(i[2]/2**0.5*1)
            mean_values = np.mean(blured_zones[i[1]-side:i[1]+side, i[0]-side:i[0]+side], axis=(0, 1))
            
            if mean_values[0] > 100 and mean_values[0] > mean_values[1] and mean_values[0] > mean_values[2] and mean_values[1] < 135 and mean_values[2] < 135:
                print(mean_values)
                # Draw the outer circle
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)
                
                sign_type = 'None'
                sub_side = int(side * 0.8)
                
                roi = blured_zones[max(1, i[1]-sub_side):i[1]+sub_side, max(1, i[0]-sub_side):i[0]+sub_side]
                roi_g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_g = cv2.medianBlur(roi_g, 3)
                
                
                roi_bin = np.where(roi_g >= 128, 255, 0).astype(np.uint8)
                roi_bin = cv2.dilate(roi_bin, np.ones((5, 5), np.uint8), iterations=1) 

                white_pixels = np.argwhere(roi_bin == 255)
                #print(white_pixels)

                # Вычисляем центр масс белых пикселей
                if len(white_pixels) > 0:
                    center_of_mass = white_pixels.mean(axis=0)
                    x, y = int(center_of_mass[0]), int(center_of_mass[1])
                else:
                    pass

                roi_c = np.stack((roi_bin,) * 3, axis=-1)
                blured_zones[max(1, i[1]-sub_side):i[1]+sub_side, max(1, i[0]-sub_side):i[0]+sub_side] = roi_c

                left_p = (i[0]-3*sub_side//4, i[1]+3*sub_side//4)
                middle_p = (i[0], i[1]+3*sub_side//4)
                right_p = (i[0]+3*sub_side//4, i[1]+3*sub_side//4)
                
                #print(blured_zones[left_p[1]][left_p[0]][0], blured_zones[middle_p[1]][middle_p[0]][0], blured_zones[right_p[1]][right_p[0]][0])
                if blured_zones[left_p[1]][left_p[0]][0] == 255:
                    sign_type = 'Right'
                elif blured_zones[middle_p[1]][middle_p[0]][0] == 255:
                    sign_type = 'Straight'
                elif blured_zones[right_p[1]][right_p[0]][0] == 255:
                    sign_type = 'Left'

                cv2.circle(blured_zones, left_p, 2, (0,0,255), -1)
                cv2.circle(blured_zones, middle_p, 2, (0,0,255), -1)
                cv2.circle(blured_zones, right_p, 2, (0,0,255), -1)


                cv2.putText(blured_zones, f'{sign_type}', (i[0]-i[2], i[1]-i[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  #x, y, r = i[0], i[1], i[2]
                

            else:
                print(mean_values)
                
     
    cv2.imshow('Original', blured_zones)
    cv2.imshow('Detected Objects', cv2.bitwise_and(frame, mask))

    
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

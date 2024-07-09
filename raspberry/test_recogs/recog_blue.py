import cv2
import numpy as np
import time

def nothing(x):
    pass

# Create a window for the trackbars
cv2.namedWindow('HSV Adjustments')

# Create trackbars for color change with initial values
cv2.createTrackbar('H Lower', 'HSV Adjustments', 90, 179, nothing)
cv2.createTrackbar('S Lower', 'HSV Adjustments', 128, 255, nothing)
cv2.createTrackbar('V Lower', 'HSV Adjustments', 128, 255, nothing)
cv2.createTrackbar('H Upper', 'HSV Adjustments', 115, 179, nothing)
cv2.createTrackbar('S Upper', 'HSV Adjustments', 255, 255, nothing)
cv2.createTrackbar('V Upper', 'HSV Adjustments', 255, 255, nothing)

# Open the video file
cap = cv2.VideoCapture('test.mp4')

# Variable to control the pause state
is_paused = False
current_frame_index = 0

def read_frame(frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    return ret, frame

# Read the first frame
ret, frame = read_frame(current_frame_index)

while True:
    if not is_paused:
        time.sleep(0.08)
        # Read a frame from the video capture
        ret, frame = read_frame(current_frame_index)
        if not ret:
            break

        current_frame_index += 1

    # Convert the frame from BGR to HSV
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
        if cv2.contourArea(contour) > 500:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Show the original frame and the frame with blue objects highlighted
    cv2.imshow('Original', frame)
    cv2.imshow('Blue Objects', cv2.bitwise_and(frame, frame, mask=mask))

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

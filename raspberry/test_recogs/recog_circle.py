import cv2
import numpy as np

def nothing(x):
    pass

# Create a window for the trackbars
cv2.namedWindow('Circle Adjustments')

# Create trackbars for HoughCircles parameters
cv2.createTrackbar('dp', 'Circle Adjustments', 1, 10, nothing)
cv2.createTrackbar('minDist', 'Circle Adjustments', 20, 100, nothing)
cv2.createTrackbar('param1', 'Circle Adjustments', 255, 255, nothing)
cv2.createTrackbar('param2', 'Circle Adjustments', 30, 255, nothing)
cv2.createTrackbar('minRadius', 'Circle Adjustments', 10, 100, nothing)
cv2.createTrackbar('maxRadius', 'Circle Adjustments', 200, 1000, nothing)

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
        # Read a frame from the video capture
        ret, frame = read_frame(current_frame_index)
        if not ret:
            break

        current_frame_index += 1

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 1.5)

    # Get current positions of the trackbars
    dp = cv2.getTrackbarPos('dp', 'Circle Adjustments')
    minDist = cv2.getTrackbarPos('minDist', 'Circle Adjustments')
    param1 = cv2.getTrackbarPos('param1', 'Circle Adjustments')
    param2 = cv2.getTrackbarPos('param2', 'Circle Adjustments')
    minRadius = cv2.getTrackbarPos('minRadius', 'Circle Adjustments')
    maxRadius = cv2.getTrackbarPos('maxRadius', 'Circle Adjustments')

    dp = 1.5
    minDist = 10

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1.5, minDist=50,
                               param1=50, param2=0.96,
                               minRadius=5, maxRadius=200)

    # Create a mask to highlight the detected circles
    mask = np.zeros_like(frame)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)

    # # Detect contours to find ovals
    # contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     if len(contour) >= 5:
    #         ellipse = cv2.fitEllipse(contour)
    #         cv2.ellipse(frame, ellipse, (0, 0, 255), 2)
    #         cv2.ellipse(mask, ellipse, (255, 255, 255), -1)

    # Show the original frame and the frame with detected circles and ovals
    
    masked = cv2.bitwise_and(frame, mask)

    blured = cv2.medianBlur(masked, 21)

    cv2.imshow('Original', frame)
    
    cv2.imshow('Detected Objects', cv2.bitwise_and(blured, mask)
)

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

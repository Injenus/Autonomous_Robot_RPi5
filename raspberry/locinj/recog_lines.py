import cv2
import numpy as np

def process_frame(frame, edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Фильтрация горизонтальных и вертикальных линий
            if abs(y2 - y1) < 10 or abs(x2 - x1) < 10 or 1:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return frame

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    
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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
        bin = np.where(gray >= 200, 255, 0).astype(np.uint8)
        edges = cv2.Canny(blur, 149, 150)

        processed_frame = process_frame(frame, edges)

        cv2.imshow('Bin', bin)
        cv2.imshow('Canny', edges)
        cv2.imshow('Processed Frame', processed_frame)
        
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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "test.mp4"
    main(video_path)

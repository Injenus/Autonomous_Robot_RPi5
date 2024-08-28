import cv2
import numpy as np
import time

frame_width = 2592
frame_height = 1944
focal_length_mm = 0.87  # фокусное расстояние в мм
sensor_diagonal_mm = 4  # диагональ сенсора в мм
aspect_ratio = 4/3  # соотношение сторон кадра
sensor_width_mm = 3.39
sensor_height_mm = 2.54
pixel_width_mm = sensor_width_mm / frame_width
pixel_height_mm = sensor_height_mm / frame_height
f_x = focal_length_mm / pixel_width_mm
f_y = focal_length_mm / pixel_height_mm
c_x = 1080 / 2
c_y = 720 / 2
camera_matrix = np.array([[f_x, 0, c_x],
                          [0, f_y, c_y],
                          [0, 0, 1]], dtype=np.float32)
k1 = -0.3  # Параметр радиального искажения
dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)


def distor_corr(frame):
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(
        frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undistorted_frame


# Open the video file
cap = cv2.VideoCapture('1080.mp4')

# Variable to control the pause state
is_paused = False
current_frame_index = 0


def read_frame(frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    return ret, frame

#######################


def process_frame(frame, edges):

    lines = cv2.HoughLinesP(edges, 1, 1*np.pi / 180,
                            threshold=30, minLineLength=50, maxLineGap=100)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return frame


def order_points(pts):
    # Сортировка по Y-координате (pts[:, 1]) сначала, затем по X-координате (pts[:, 0])
    sorted_pts = pts[np.lexsort((pts[:, 0], pts[:, 1]))]

    # Упорядочивание точек для образования прямоугольника
    rect = np.zeros((4, 2), dtype=pts.dtype)

    if sorted_pts[0, 0] < sorted_pts[1, 0]:
        rect[0] = sorted_pts[0]
        rect[1] = sorted_pts[1]
    else:
        rect[0] = sorted_pts[1]
        rect[1] = sorted_pts[0]

    if sorted_pts[2, 0] < sorted_pts[3, 0]:
        rect[2] = sorted_pts[3]
        rect[3] = sorted_pts[2]
    else:
        rect[2] = sorted_pts[2]
        rect[3] = sorted_pts[3]

    return rect

#########################


def crop_triangle(image, base_angle_deg):
    height, width = image.shape[:2]

    # Define the triangle vertices
    base_length = width  # Full width as the base of the triangle
    base_angle_rad = np.deg2rad(base_angle_deg)

    # Calculate triangle vertices
    vertices = np.array([
        [[0, height],                                      # Bottom-left corner
         # Bottom-right corner
         [width, height],
         [width//2 - int((width * np.tan(base_angle_rad))/2), 0],  # Top vertex
         [width//2 + int((width * np.tan(base_angle_rad))/2), 0]]  # Top vertex
    ])

    # Create a black mask with the same dimensions as the image
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill the triangle region in the mask with white (255)
    cv2.fillPoly(mask, vertices, 255)

    # Apply the mask to the image to extract the triangle region
    result = cv2.bitwise_and(image, image, mask=mask)

    return result


sk = 10
top, bottom, left, right = [sk, sk, sk, sk]  # толщина рамки

start_frame = 80  # 1000
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

    # frame[:int(frame.shape[0]*0.5), :] = 0
    frame = frame[int(frame.shape[0]*0.5):, :]
    # frame = frame[:, 300:frame.shape[1] - 300]
    # frame = crop_triangle(frame, 0.1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
    # eqhist = cv2.equalizeHist(gray)

    # bord = cv2.copyMakeBorder(blur, 2*top, 2*bottom, 2*left, 2*right, cv2.BORDER_CONSTANT, value=[0,0,0])
    # bord = cv2.copyMakeBorder(bord, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255,255,255])
    bord = cv2.rectangle(
        blur, (1, 1), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 0), 2)

    mask = np.where(bord >= 144, 255, 0).astype(np.uint8)

    edges = cv2.Canny(mask, 50, 150)

    # hough = process_frame(frame, edges)

    cnt_simple = []
    cnt_oi = []

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, cn in enumerate(contours):
        isOI = False
        # area = cv2.contourArea(cn)
        # print(f'{area}, id {current_frame_index}')
        # cv2.drawContours(frame, [cn], -1, (0, 255, 0), 2)
        # # M = cv2.moments(cn)
        # # cX = int(M["m10"] / (M["m00"]))
        # # cY = int(M["m01"] / (M["m00"]))
        # cX = 0
        # cY = 0
        # for p in cn:
        #     cX += p[0][0]
        #     cY += p[0][1]
        # cX = int(cX/len(cn))
        # cY = int(cY/len(cn))
        # if area >= 5000:
        #     cv2.circle(frame, (cX, cY), 3, (0, 255, 255), -1)
        # elif area >= 50:
        #     cv2.circle(frame, (cX, cY), 3, (0, 0, 255), -1)
        # elif area >= 10:
        #     cv2.circle(frame, (cX, cY), 3, (0, 255, 0), -1)
        # elif area >= 1:
        #     cv2.circle(frame, (cX, cY), 3, (0, 255, 255), -1)
        # elif area >=0:
        #     cv2.circle(frame, (cX, cY), 3, (255, 0, 0), -1)

        # (x,y,w,h) = cv2.boundingRect(cn)
        # cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        # cv2.circle(frame, ((x+x+w)//2,(y+y+h)//2), 2, (0,0,255), -1)

        rect = cv2.minAreaRect(cn)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = np.int0(order_points(box))

        cv2.drawContours(frame, [box], -1, (0, 255, 0), 1)
        area = cv2.contourArea(box)

        if 4000 < area:
            cX = 0
            cY = 0
            for i, p in enumerate(box):
                cX += p[0]
                cY += p[1]
                # cv2.putText(frame, f'{i}', p, cv2.FONT_HERSHEY_SIMPLEX, 0.67, (0, 0, 255), 2)
            cX = int(cX/len(box))
            cY = int(cY/len(box))
            cv2.putText(frame, f'{round(area)}', (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.67, (0, 255, 255), 2)
            cv2.circle(frame, (cX, cY), 3, (255, 128, 0), -1)

            m_top = (box[0]+box[1])//2
            m_bott = (box[3]+box[2])//2

            # cv2.circle(frame, m_top, 4, (0,0,255), -1)
            # cv2.circle(frame, m_bott, 4, (0,0,255), -1)

            delta_x = m_top[0] - m_bott[0]
            delta_y = m_top[1] - m_bott[1]
            theta_rad = np.arctan2(delta_y, delta_x)
            theta_deg = -int(np.degrees(theta_rad))
            # cv2.putText(frame, f'{theta_deg}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (0, 0, 255), 2)

            if area >= 80000:
                cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
                isOI = True
                cnt_oi.append((i, cX, cY, area, theta_deg, isOI))
            else:
                cnt_simple.append((i, cX, cY, area, theta_deg, isOI))

    if len(cnt_oi) > 0:
        if len(cnt_oi) > 1:
            print('pizdec')
        else:
            if len(cnt_simple) > 0:
                oi_data = cnt_oi[0]
                mean_y = int(sum(cnt[2] for cnt in cnt_simple)/len(cnt_simple))
                if len(cnt_simple) >= 3:
                    if mean_y < oi_data[2]:
                        print('ПЕРЕКРЁСОК БУДЕТ')
                    elif mean_y > oi_data[2]:
                        print('ПЕРЕКРЁСОК БЫЛ')
                    else:
                        print('ERR че за хуйня')
                else:
                    print("ПОВОРОТ")  ##### СЛАБОЕ МЕСТО СРАЗУ ПОСЛЕ ПРОЕЗДА ПЕРЕХОДА (пизда логике)
            else:
                print('ПОВРОТ 100%')




        #

    # cv2.imshow('eq', eqhist)
    cv2.imshow('g', blur)
    cv2.imshow('Canny', edges)

    cv2.imshow('mask', mask)

    cv2.imshow('bord', bord)
    cv2.imshow('Orig', frame)
   # cv2.imshow('Lines', hough)

    time.sleep(0.06)
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

cap.release()
cv2.destroyAllWindows()

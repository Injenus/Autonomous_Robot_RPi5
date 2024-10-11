import time
import cv2
import numpy as np
import serial

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

def get_signs(frame):
    sign = []  # (id, xy, type)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.5)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1.5, minDist=30,
                               param1=50, param2=0.94,
                               minRadius=18, maxRadius=30)

    mask = np.zeros_like(frame)
    blured_zones = frame.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for count, i in enumerate(circles[0, :]):

            side = int(i[2]/2**0.5*1)
            mean_values = np.mean(
                blured_zones[i[1]-side:i[1]+side, i[0]-side:i[0]+side], axis=(0, 1))

            if mean_values[0] > 90 and mean_values[0] > mean_values[1] and mean_values[0] > mean_values[2] and mean_values[1] < 110 and mean_values[2] < 90:
                # print('c', mean_values)

                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)

                sign_type = 'None'
                sub_side = int(side * 0.8)

                roi = blured_zones[max(
                    1, i[1]-sub_side):i[1]+sub_side, max(1, i[0]-sub_side):i[0]+sub_side]
                roi_g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_g = cv2.resize(roi_g, (roi_g.shape[0]*2, roi_g.shape[0]*2))
                sub_side *= 2
                roi_g = cv2.medianBlur(roi_g, 3)

                roi_bin = np.where(roi_g >= 80, 255, 0).astype(np.uint8)
                roi_bin = cv2.dilate(roi_bin, np.ones(
                    (4, 4), np.uint8), iterations=1)

                white_pixels = np.argwhere(roi_bin == 255)
                # print(white_pixels)

                roi_c = np.stack((roi_bin,) * 3, axis=-1)
                blured_zones[max(1, i[1]-sub_side):i[1]+sub_side,
                             max(1, i[0]-sub_side):i[0]+sub_side] = roi_c

                left_p = (i[0]-3*sub_side//4, i[1]+3*sub_side//4)
                middle_p = (i[0], i[1]+3*sub_side//4)
                right_p = (i[0]+3*sub_side//4, i[1]+3*sub_side//4)

                l, m, r = av_col_3x3(blured_zones, left_p), av_col_3x3(
                    blured_zones, middle_p), av_col_3x3(blured_zones, right_p)
                # print(l,m,r)
                if l > m and l > r:
                    sign_type = 'Right'
                elif m > l and m > r:
                    sign_type = 'Straight'
                elif r > l and r > m:
                    sign_type = 'Left'

                sign.append((count, (i[0], i[1]), sign_type))

                cv2.circle(blured_zones, left_p, 2, (0, 0, 255), -1)
                cv2.circle(blured_zones, middle_p, 2, (0, 0, 255), -1)
                cv2.circle(blured_zones, right_p, 2, (0, 0, 255), -1)

                # x, y, r = i[0], i[1], i[2]
                cv2.putText(blured_zones, f'{sign_type}', (
                    i[0]-i[2], i[1]-i[2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # print(sign_type)

            else:
                print('fuck', mean_values)
                pass

    cv2.imshow('Signs', blured_zones)
    return sign

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


def get_road(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.5)

    bord = cv2.rectangle(
        blur, (1, 1), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 0), 2)

    mask = np.where(bord >= 144, 255, 0).astype(np.uint8)
    edges = cv2.Canny(mask, 50, 150)

    cnt_simple = []
    cnt_oi = []

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, cn in enumerate(contours):
        isOI = False

        rect = cv2.minAreaRect(cn)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = np.int0(order_points(box))

        cv2.drawContours(frame, [box], -1, (0, 255, 0), 1)
        area = cv2.contourArea(box)

        if 1500 < area:
            cX = 0
            cY = 0
            for i, p in enumerate(box):
                cX += p[0]
                cY += p[1]
                # cv2.putText(frame, f'{i}', p, cv2.FONT_HERSHEY_SIMPLEX, 0.67, (0, 0, 255), 2)
            cX = int(cX/len(box))
            cY = int(cY/len(box))
            cv2.putText(frame, f'{round(area), cX, cY}', (cX, cY),
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

            if area >= 15000:
                cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
                isOI = True
                cnt_oi.append((i, cX, cY, area, theta_deg, isOI))
            else:
                cnt_simple.append((i, cX, cY, area, theta_deg, isOI))

    # if len(cnt_oi) > 0:
    #     if len(cnt_oi) > 1:
    #         print('pizdec')
    #     else:
    #         if len(cnt_simple) > 0:
    #             oi_data = cnt_oi[0]
    #             mean_y = int(sum(cnt[2] for cnt in cnt_simple)/len(cnt_simple))
    #             if len(cnt_simple) >= 3:
    #                 if mean_y < oi_data[2]:
    #                     #print('ПЕРЕКРЁСОК БУДЕТ')
    #                 elif mean_y > oi_data[2]:
    #                     #print('ПЕРЕКРЁСОК БЫЛ')
    #                 else:
    #                     #print('ERR че за хуйня')
    #             else:
    #                 #print("ПОВОРОТ")  ##### СЛАБОЕ МЕСТО СРАЗУ ПОСЛЕ ПРОЕЗДА ПЕРЕХОДА (пизда логике)
    #         else:
    #             #print('ПОВРОТ 100%')

    cv2.imshow('Road', frame)

    return cnt_simple, cnt_oi


def get_traffic_light(frame):
    tl_color = None

    return tl_color

#################################3
def find_dominant_direction(side_dict):

    max_value = max(side_dict.values())
    
    dominant_directions = [k for k, v in side_dict.items() if v == max_value]
    
    if len(dominant_directions) > 1 or max_value == 0:
        return None, None
    else:
        return dominant_directions[0], side_dict[dominant_directions[0]]

def average_last_5_x(data):
    if len(data) >= 5:
        last_elements = data[-5:]
    else:
        last_elements = data
    
    x_values = [elem[1][0] for elem in last_elements]
    
    average_x = sum(x_values) / len(x_values)

    return average_x

def get_actual_sign(dict_of_signs):
    actual = None
    left_sign, right_sign = None, None


    left_side = {'Left': 0, 'Straight':0 , 'Right': 0}
    for sign in dict_of_signs['left']:
        left_side[sign[2]] += 1

    right_side = {'Left': 0, 'Straight':0 , 'Right': 0}
    for sign in dict_of_signs['right']:
        right_side[sign[2]] += 1

    left_sign, left_side_count = find_dominant_direction(left_side)
    right_sign, right_side_count = find_dominant_direction(right_side)

    if left_side is None and right_side is None:
        actual = None
    elif left_side is None:
        actual = right_sign
    elif right_side is None:
        actual =  left_sign
    else:
        mean_left_offset = average_last_5_x(dict_of_signs['left']) / 360
        mean_right_offset = (720 - average_last_5_x(dict_of_signs['right'])) / 360
        
        if mean_left_offset == mean_right_offset:
            if left_side_count > right_side_count:
                actual = left_sign
            elif left_side_count < right_side_count:
                actual = right_sign
            else:
                print(' так не бывает')
        elif mean_left_offset < mean_right_offset:
            actual = left_sign
        else:
            actual = right_sign

    return actual


def reset_dict(d):
    for key in d:
        d[key] = []

###################

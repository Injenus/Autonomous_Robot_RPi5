"""
для каждого кадра помним лог:
id, дорожная ситуация, статус робота

когда робот едет по таймингам он не сенсит НИХУЯ

после перекрёстка робот едет если прямо, то до oi, начианется сенс, едем по линии,текущий oi игнорится
если с повротом, то прямо по врмени, потом слепой поврот и до oi, начинается сенс, едем по линии,текущий oi игнорится

в остальном едем по линии, когда встречаем oi ждем, пока его размер не станет уменьшаеться, потом попорачиваем согласно последнему знаку

послдений знак - распознавание последнего знака НЕ буквально
знаки распознаем, когда едем по линии или при текущем oi как бы в конце манёвра (во время)
когда понимаем, что oi и надо манёвр, обарщаемся к списку знаком:
    отдельно зраним знакии левее центра и правее
    среди каждой стороны смотрим какого типа больше, оставляем его
    смотрим на средние посдение координаты знаков для кажолй стороны, выбираем с блажйшей к краю
    это будет знак
    выводим, чистим список

маннёвр - поворот на oi иьи проезд oi, делается по таймингу из эл. команд "прямо"время "поворот"время

если при старте стоим на oi, то просто едем по линии как обычно бы

"""
import time
import cv2
import numpy as np
import serial

### SETUP ####
PERIOD = 0.1

previous_error = 0
integral = 0
Kp = 0.5
Ki = 0.0
Kd = 0.0
dt = PERIOD

cap = cv2.VideoCapture('1080.mp4')
ar_cam = cv2.VideoCapture('1080.mp4')
period_aruco = 0.3

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

sign_c_x = 1080 / 2 - 150
sign_c_y = 720 / 2 - 120
sign_camera_matrix = np.array([[f_x, 0, sign_c_x],
                               [0, f_y, sign_c_y],
                               [0, 0, 1]], dtype=np.float32)
sign_k1 = -0.3  # Параметр радиального искажения
sign_dist_coeffs = np.array([sign_k1, 0, 0, 0, 0], dtype=np.float32)

road_c_x = 540 / 2
road_c_y = 320 / 2
road_camera_matrix = np.array([[f_x, 0, road_c_x],
                               [0, f_y, road_c_y],
                               [0, 0, 1]], dtype=np.float32)
road_k1 = -0.3  # Параметр радиального искажения
road_dist_coeffs = np.array([road_k1, 0, 0, 0, 0], dtype=np.float32)
#
port = '/dev/ttyUSB0'
baudrate = 115200
ser = serial.Serial(port, baudrate, timeout=1)
if ser.is_open:
    print(f"Serial port {port} is open")
#
##### GLOBAL STATE VAR #####
bot_state = None  # 0-стоять 1-еду по лини. 2-поворот налево по таймингу. 3-поврот направо таймингу. 4-еду прямо потаймингу

#
#
#
###### RECOGNITION: ####


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

#
#
#
####### STATE: ######

def find_dominant_direction(side_dict):

    max_value = max(side_dict.values())
    
    dominant_directions = [k for k, v in side_dict.items() if v == max_value]
    
    if len(dominant_directions) > 1 or max_value == 0:
        return None
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


#
#
#
############# MOVE: ##############


def send_command(left_wheel_speed, right_wheel_speed):
    try:
        checksum = left_wheel_speed * 2 + right_wheel_speed * 4
        message = f"s,{left_wheel_speed},{right_wheel_speed},{checksum},f"
        ser.write(message.encode())
        print(f"Sent: {message}")
    except Exception as e:
        print(f'ERROR {e}')

def pid(err, prev_err, integral, Kp, Ki, Kd, dt):
    l_speed, r_speed = 0, 0
    print(f'err {err}')

    P = Kp * err

    integral += err * dt
    I = Ki * integral
    
    derivative = (err - prev_err) / dt
    D = Kd * derivative
    
    output = P + I + D
    
    min_speed = 20
    max_speed = 130
    
    base_speed = 65 
    l_speed = base_speed - output
    r_speed = base_speed + output
    
    if l_speed < min_speed:
        l_speed = min_speed
    elif l_speed > max_speed:
        l_speed = max_speed
    
    if r_speed < min_speed:
        r_speed = min_speed
    elif r_speed > max_speed:
        r_speed = max_speed
    print('pid', l_speed, r_speed)
    return l_speed, r_speed, err, integral

def move_line(target_coord):
    global bot_state
    err = 360 - target_coord[0]
    bot_state = 1
    print(f'едем по линии')
    left_w, right_w, previous_error, integral = pid(err, previous_error, integral, Kp, Ki, Kd, dt)
    send_command(left_w, right_w)

def move_straight_time(time_seconds):
    global bot_state
    print(f'едем прямо {time_seconds} секунд..')
    timer = time.time()
    while time.time() - timer < time_seconds:
        send_command(90, 90)
        time.sleep(0.05)

def move_turn_time(direc):
    global bot_state 
    print(f'поворот на{direc}#############################################################')
    if direc == 'straight':
        move_straight_time(3)
    elif direc == 'left':
        send_command(30, 100)
        time.sleep(3)
    elif direc == 'right':
        send_command(100, 30)
        time.sleep(3)
    

def stop_move():
    global bot_state
    print('стоим__________________________')
    time.sleep(6)
    bot_state = 0
    send_command(0, 0)
#
#
#
####### CAMERA: ###############


def sign_distor_corr(frame):
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        sign_camera_matrix, sign_dist_coeffs, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(
        frame, sign_camera_matrix, sign_dist_coeffs, None, new_camera_matrix)
    return undistorted_frame


def road_distor_corr(frame):
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        sign_camera_matrix, sign_dist_coeffs, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(
        frame, sign_camera_matrix, sign_dist_coeffs, None, new_camera_matrix)
    return undistorted_frame


def detect_aruco():
    aruco = None

    return aruco
#
#
#
##############################################

def main():

    timer = time.time()
    iter_count = 0

    timer_aruco = time.time()

    INIT = None
    while INIT != 'ho':
        INIT = input("Ожидание нажатия начала... ")
    print("ПОГНАЛИ")

    dict_of_signs = {'left': [], 'right': []}
    was_oi = False
    curr_area_oi = 0
    was_crossroad = False

    is_running, is_paused = True, False
    while is_running:
        if time.time() - timer >= PERIOD:
            timer = time.time()

            if not is_paused:                
                ret, frame = cap.read()
                if not ret:
                    break

                if time.time() - timer_aruco > period_aruco:
                    timer_aruco = time.time()
                    
                    ret_ar, frame_ar = ar_cam.read()
                    if not ret_ar:
                        break

                    recog_aruco = detect_aruco(frame_ar)
                    if recog_aruco is not None:
                        pass
                        is_running = False
                        break  


            else:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):  # Pause/resume on 'p' keyl
                    is_paused = not is_paused
                continue
            cv2.imshow('frame', frame)
            cv2.imshow('aruco', frame_ar)
            ###
            sign_frame = frame[120:720-400, 150:1080-150]
            sign_frame = sign_distor_corr(sign_frame)

            road_frame = cv2.resize(frame, (720, 480))   #1080 720
            road_frame = road_distor_corr(road_frame)
            road_frame = road_frame[int(road_frame.shape[0]*0.5):, :]

            signs = get_signs(sign_frame)
            
            cnt_simple, cnt_oi = get_road(road_frame)

            print(f'signs {signs}')
            print(f'road {cnt_simple, cnt_oi}')

            if len(signs) > 0:
                for i, sign in enumerate(signs):
                    x, y = sign[1][0], sign[1][1]
                    if x < 780//2:
                        dict_of_signs['left'].append(sign)
                    else:
                        dict_of_signs['right'].append(sign)

            
            if len(cnt_oi) == 0:
                if len(cnt_simple) > 0:
                    if was_oi:
                        direc = 'хуй' # left or right
                        direc = get_actual_sign(dict_of_signs)
                        reset_dict(dict_of_signs)   

                        move_turn_time(direc)
                        was_oi = False
                    else:                
                        target_coord = [360, 480] # x, y
                        target_id = None
                        for i, simple in enumerate(cnt_simple):
                            if simple[2] <= target_coord[1]:
                                target_coord[1] = cnt_simple[i][2]
                                target_id = i
                        target_coord[0] = cnt_simple[target_id][1]
                        move_line(target_coord)
                else:
                    print(' мы на перекрёстке???')

            elif len(cnt_oi) == 1:
                if not was_oi:
                    curr_area_oi = cnt_oi[0][3]
                    was_oi = True

                if len(cnt_simple) >= 5:
                    mean_y = int(sum(cnt[2] for cnt in cnt_simple)/len(cnt_simple))
                    if mean_y < cnt_oi[0][2]:
                        if not was_crossroad:
                            print('на перекрстке')
                            stop_move()
                            t_light = get_traffic_light(sign_frame)
                            while t_light != 'green' and 0:
                                t_light = get_traffic_light(sign_frame)
                                stop_move()

                            move_straight_time(time_seconds=10)  #едем по перексртёку вслепую
                        else:
                            target_coord = [cnt_oi[0][1], cnt_oi[0][2]]
                            move_line(target_coord)
                else:
                    target_coord = [cnt_oi[0][1], cnt_oi[0][2]]
                    move_line(target_coord)



                ######
                # else:
                #     if cnt_oi[0][3] + 400 < curr_area_oi:
                #         direc = 'хуй' # left or right
                #         get_actual_sign()
                #         move_turn_time(direc)
                #     else:
                #         curr_area_oi = cnt_oi[0][3]
                #         target_coord = [cnt_oi[0][1], cnt_oi[0][2]]
                #         move_line(target_coord)
                # #####
                

            else:
                print('хз че делать')

                

            try:
                cv2.circle(road_frame, target_coord, 6, (0,0,255), 2)
                cv2.imshow('target', road_frame)
            except:
                pass
            ###
            iter_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):  # Pause/resume on 'p' key
                is_paused = not is_paused
            




if __name__ == '__main__':
    main()
    ser.close()

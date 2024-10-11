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
from tools import *
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
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
driver_is_disabled = False
try:
    ser = serial.Serial(port, baudrate, timeout=1)
except:
    print('!!!!driver отключён!!!!! рабоатет без движения')
    driver_is_disabled = True
if not driver_is_disabled:
    if ser.is_open:
        print(f"Serial port {port} is open")
#
##### GLOBAL STATE VAR #####
bot_state = None  # 0-стоять 1-еду по лини. 2-поворот налево по таймингу. 3-поврот направо таймингу. 4-еду прямо потаймингу

#
#
#
###### RECOGNITION: ####



#
#
#
####### STATE: ######

#
#
#
############# MOVE: ##############


def send_command(left_wheel_speed, right_wheel_speed):
    global driver_is_disabled
    if not driver_is_disabled:
        try:
            checksum = left_wheel_speed * 2 + right_wheel_speed * 4
            message = f"s,{left_wheel_speed},{right_wheel_speed},{checksum},f"
            ser.write(message.encode())
            print(f"Sent: {message}")
        except Exception as e:
            print(f'ERROR {e}')
    else:
        print('получили команду на отправку, но драйвер отключён: ', message)

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
    global bot_state, previous_error, integral
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


def detect_aruco(frame, aruco_dict=aruco_dict):
    corners, ids, rej = cv2.aruco.detectMarkers(frame, aruco_dict)
    if ids is None:
        return None
    return ids[0]
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
                        print(f"ARUCO {recog_aruco} detected, stop over!")
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
            print('------new cadr------------------')
            ###
            sign_frame = frame[120:720-400, 150:1080-150]
            sign_frame = sign_distor_corr(sign_frame)

            road_frame = cv2.resize(frame, (720, 480))   #1080 720
            road_frame = road_distor_corr(road_frame)
            road_frame = road_frame[int(road_frame.shape[0]*0.5):, :]

            signs = get_signs(sign_frame) #текущие знаки: кортежи (номер, центр, тип)
            
            cnt_simple, cnt_oi = get_road(road_frame) # списки

            print(f'signs {signs}')
            print(f'road {cnt_simple, cnt_oi}')

            if len(signs) > 0:
                for i, sign in enumerate(signs):
                    x, y = sign[1][0], sign[1][1]
                    if x < 780//2: # добалвяемвм словарь как знак слева или справа
                        dict_of_signs['left'].append(sign)
                    else:
                        dict_of_signs['right'].append(sign)

            
            if len(cnt_oi) == 0: # если нет большой области
                if len(cnt_simple) > 0: # если есть простые области
                    if was_oi: # и до этого была большая область
                        #едем по знаку
                        direc = 'хуй' # left or right
                        direc = get_actual_sign(dict_of_signs)
                        
                        reset_dict(dict_of_signs)   
                        if direc is not None:
                            move_turn_time(direc)
                        else:
                            print('была команда по знаку, но знаков не было')
                        was_oi = False
                    else: # если большой облати не было, то просто едем по линии               
                        target_coord = [360, 480] # x, y
                        target_id = None
                        for i, simple in enumerate(cnt_simple):
                            if simple[2] <= target_coord[1]:
                                target_coord[1] = cnt_simple[i][2]
                                target_id = i
                        target_coord[0] = cnt_simple[target_id][1]
                        move_line(target_coord)
                else: # если и обычноых областей нет, то мы ниче не видим
                    print(' мы на перекрёстке???')

            elif len(cnt_oi) == 1: # есть большая область прям сейчас
                if not was_oi: # если видим её сечас в первый раз
                    curr_area_oi = cnt_oi[0][3]
                    was_oi = True

                if len(cnt_simple) >= 5: #если мелких линий много
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
    stop_move()
    ser.close()

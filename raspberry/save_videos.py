import cv2
import time

def list_connected_cameras(max_cameras=10):
    cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    return cameras

def main():
    # Сканируем доступные камеры
    cameras = list_connected_cameras()
    if len(cameras) < 2:
        print("Недостаточно камер подключено. Найдено только:", len(cameras))
        return

    print("Подключенные камеры:", cameras)

    # Настройки для первой камеры
    camera1_id = cameras[0]
    camera1_output = 'camera1_output.mp4'
    cap1 = cv2.VideoCapture(camera1_id, cv2.CAP_V4L2)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap1.set(cv2.CAP_PROP_FPS, 20)
    fps1_set = cap1.get(cv2.CAP_PROP_FPS)
    print(f'Настроенный FPS для камеры 1: {fps1_set}')

    # Настройки для второй камеры
    camera2_id = cameras[1]
    camera2_output = 'camera2_output.mp4'
    cap2 = cv2.VideoCapture(camera2_id, cv2.CAP_V4L2)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap2.set(cv2.CAP_PROP_FPS, 20)
    fps2_set = cap2.get(cv2.CAP_PROP_FPS)
    print(f'Настроенный FPS для камеры 2: {fps2_set}')

    # Кодек и настройки записи
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out1 = cv2.VideoWriter(camera1_output, fourcc, 20.0, (1280, 720))
    out2 = cv2.VideoWriter(camera2_output, fourcc, 20.0, (1280, 720))

    prev_time1 = time.time()
    prev_time2 = time.time()
    frame_count1 = 0
    frame_count2 = 0
    fps1 = 0
    fps2 = 0

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if ret1 and ret2:
            # Рассчитываем FPS для камеры 1
            frame_count1 += 1
            current_time1 = time.time()
            if current_time1 - prev_time1 >= 1:
                fps1 = frame_count1 / (current_time1 - prev_time1)
                frame_count1 = 0
                prev_time1 = current_time1
            
            # Рассчитываем FPS для камеры 2
            frame_count2 += 1
            current_time2 = time.time()
            if current_time2 - prev_time2 >= 1:
                fps2 = frame_count2 / (current_time2 - prev_time2)
                frame_count2 = 0
                prev_time2 = current_time2

            # Добавляем FPS на кадр
            #cv2.putText(frame1, f'Set FPS: {fps1_set:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame1, f'Real FPS: {fps1:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #cv2.putText(frame2, f'Set FPS: {fps2_set:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame2, f'Real FPS: {fps2:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out1.write(frame1)
            out2.write(frame2)

            # Уменьшаем кадры в 4 раза для отображения
            small_frame1 = cv2.resize(frame1, (320, 180))
            small_frame2 = cv2.resize(frame2, (320, 180))

            cv2.imshow('Camera 1', small_frame1)
            cv2.imshow('Camera 2', small_frame2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap1.release()
    cap2.release()
    out1.release()
    out2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

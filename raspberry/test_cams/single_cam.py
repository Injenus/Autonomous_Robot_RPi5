from picamera2 import Picamera2
import cv2
import time

w, h = 1280, 720
def main():
    # Инициализация камеры
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (w, h)})
    picam2.configure(camera_config)

    # Установка параметров камеры
    picam2.set_controls({
        "AwbEnable": 0,  # Отключение автоматического баланса белого
        "ColourGains": (1.4, 1.5),  # Установка коэффициентов цветовой коррекции
        "AeEnable": 0,  # Отключение автоматической экспозиции
        "AnalogueGain": 3.0,  # Установка аналогового усиления (чувствительности)
        "ExposureTime": 20000  # Установка выдержки в микросекундах (например, 20000 = 20мс)
    })

    picam2.start()

    # Настройка видеозаписи
    frame_size = (w, h)
    fps = 20  # Частота кадров
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для записи видео
    out = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

    # Переменные для расчета FPS
    prev_time = time.time()
    fps_display = 0

    while True:
        # Захват кадра
        frame = picam2.capture_array()

        # Преобразование цвета
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Поворот изображения
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Запись кадра в видеофайл
        out.write(frame)
        
        # Вычисление текущего FPS
        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time
        fps_display = int(1 / elapsed_time)

        # Добавление текста с FPS на кадр
        cv2.putText(frame, f"FPS: {fps_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Отображение кадра
        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Остановка записи и освобождение ресурсов
    picam2.stop()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

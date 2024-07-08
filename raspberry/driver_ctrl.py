import serial
import time

# Укажите имя вашего последовательного порта
port = '/dev/ttyUSB0'  # Замените на /dev/ttyACM0, если у вас другое устройство

# Установите скорость передачи данных
baudrate = 115200

# Откройте последовательный порт
ser = serial.Serial(port, baudrate, timeout=1)

# Убедитесь, что порт открыт
if ser.is_open:
    print(f"Serial port {port} is open")

try:
    # Отправка данных
    number1 = 123  # Ваше первое число
    number2 = -456  # Ваше второе число
    checksum = number1 * 2 + number2 * 4
    message = f"s{number1},{number2},{checksum}f"
    ser.write(message.encode())
    print(f"Sent: {message}")

    # Ждем немного между отправками
    time.sleep(1)

except Exception as e:
    print(f"Error: {e}")

finally:
    pass
    # Закрытие порта
    ser.close()
    print("Serial port closed")

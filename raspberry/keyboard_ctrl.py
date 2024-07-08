import pygame
import serial

port = '/dev/ttyUSB0'
baudrate = 115200
ser = serial.Serial(port, baudrate, timeout=1)
if ser.is_open:
    print(f"Serial port {port} is open")

pygame.init()

window_size = (180, 60)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("crutch")

max_speed = 1000
acceleration = 100

left_wheel_speed = 0
right_wheel_speed = 0

moving_forward = False
moving_backward = False
turning_left = False
turning_right = False

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_w]:
        moving_forward = True
    else:
        moving_forward = False

    if keys[pygame.K_s]:
        moving_backward = True
    else:
        moving_backward = False

    if keys[pygame.K_a]:
        turning_left = True
    else:
        turning_left = False

    if keys[pygame.K_d]:
        turning_right = True
    else:
        turning_right = False

    if moving_forward:
        left_wheel_speed = max_speed
        right_wheel_speed = max_speed
    elif moving_backward:
        left_wheel_speed = max_speed
        right_wheel_speed = max_speed
    else:
        left_wheel_speed = 0
        right_wheel_speed = 0

    if turning_left:
        left_wheel_speed -= max_speed//2
        right_wheel_speed += max_speed//2
    elif turning_right:
        left_wheel_speed += max_speed//2
        right_wheel_speed -= max_speed//2

    if moving_backward:
        left_wheel_speed *= -1
        right_wheel_speed *= -1
          
    if left_wheel_speed > max_speed:
        left_wheel_speed = max_speed
    elif left_wheel_speed < -max_speed:
        left_wheel_speed = -max_speed
    if right_wheel_speed > max_speed:
        right_wheel_speed = max_speed
    elif right_wheel_speed < -max_speed:
        right_wheel_speed = -max_speed

    print(left_wheel_speed, right_wheel_speed)

    try:
        checksum = left_wheel_speed * 2 + right_wheel_speed * 4
        message = f"s{left_wheel_speed},{right_wheel_speed},{checksum}f"
        ser.write(message.encode())
        print(f"Sent: {message}")
    except Exception as e:
        print(f'ERROR {e}')      

    pygame.time.delay(5)
    screen.fill((42, 42, 42))
    pygame.display.flip()

    if keys[pygame.K_l]:
        running = False

ser.close()
print("Serial port closed")
pygame.quit()

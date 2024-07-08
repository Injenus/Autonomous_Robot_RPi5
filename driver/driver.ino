//#define DEBUG
#define PROD

// Пины для светодиодов
const int greenLedPin = 4;
const int redLedPin = 7;

// Пины для ШИМ
const int L_PWM_left = 5;   // PD5 / OC0B
const int R_PWM_left = 6;   // PD6 / OC0A
const int L_PWM_right = 9;  // PB1 / OC1A
const int R_PWM_right = 10; // PB2 / OC1B

void setup() {
    Serial.begin(9600);
    while (!Serial) {
        ; // Дождитесь открытия порта
    }

    pinMode(greenLedPin, OUTPUT);
    pinMode(redLedPin, OUTPUT);

    pinMode(L_PWM_left, OUTPUT);
    pinMode(R_PWM_left, OUTPUT);
    pinMode(L_PWM_right, OUTPUT);
    pinMode(R_PWM_right, OUTPUT);

    // Настройка таймеров для ШИМ
    setupPWM();

#ifdef DEBUG
    Serial.println("Arduino ready to receive data");
#endif
}

void loop() {
    if (Serial.available() > 0) {
        char input[32];
        byte index = 0;

        while (Serial.available() > 0 && index < sizeof(input) - 1) {
            char c = Serial.read();
            if (c == 'f') break;
            input[index++] = c;
        }
        input[index] = '\0';

#ifdef DEBUG
        Serial.print("Received: ");
        Serial.println(input);
#endif

        int num1 = 0;
        int num2 = 0;
        bool valid = false;

        if (input[0] == 's') {
            char* comma1 = strchr(input, ',');
            char* comma2 = strrchr(input, ',');

            if (comma1 != NULL && comma2 != NULL && comma1 != comma2) {
                *comma1 = '\0';
                *comma2 = '\0';
                num1 = atoi(&input[1]);
                num2 = atoi(comma1 + 1);
                int receivedChecksum = atoi(comma2 + 1);
                int calculatedChecksum = num1 * 2 + num2 * 4;

                if (receivedChecksum == calculatedChecksum) {
                    valid = true;
                }
            }
        }

#ifdef DEBUG
        if (valid) {
            Serial.print("Number 1: ");
            Serial.println(num1);
            Serial.print("Number 2: ");
            Serial.println(num2);
            digitalWrite(greenLedPin, HIGH);
            digitalWrite(redLedPin, LOW);
        } else {
            Serial.println("Invalid input or checksum error, defaulting to 0 and 0");
            Serial.print("Number 1: ");
            Serial.println(0);
            Serial.print("Number 2: ");
            Serial.println(0);
            digitalWrite(greenLedPin, LOW);
            digitalWrite(redLedPin, HIGH);
        }
#else
        if (!valid) {
            num1 = 0;
            num2 = 0;
            digitalWrite(greenLedPin, LOW);
            digitalWrite(redLedPin, HIGH);
        } else {
            digitalWrite(greenLedPin, HIGH);
            digitalWrite(redLedPin, LOW);
        }
        Serial.print(num1);
        Serial.print(' ');
        Serial.println(num2);
#endif

        // Управление ШИМ
        updatePWM(&L_PWM_left, &R_PWM_left, &num1);
        updatePWM(&L_PWM_right, &R_PWM_right, &num2);
    }
}

void setupPWM() {
    // Настройка таймера 0 для пинов 5 и 6
    TCCR0A = 0b10100011; // Режим Fast PWM, установка OC0A и OC0B при совпадении
    TCCR0B = 0b00000011; // Делитель 64

    // Настройка таймера 1 для пинов 9 и 10
    TCCR1A = 0b10100000; // Режим Fast PWM, установка OC1A и OC1B при совпадении
    TCCR1B = 0b00001011; // Режим Fast PWM 8-bit, делитель 64
}

void updatePWM(const int* pinL, const int* pinR, const int* value) {
    int pwmValue = *value;
    if (*pinL == L_PWM_left && *pinR == R_PWM_left) {
        if (pwmValue < 0) {
            OCR0B = -pwmValue; // Устанавливаем ШИМ на левый пин
            OCR0A = 0;         // Выключаем правый пин
        } else if (pwmValue > 0) {
            OCR0B = 0;         // Выключаем левый пин
            OCR0A = pwmValue;  // Устанавливаем ШИМ на правый пин
        } else {
            OCR0B = 0;         // Выключаем оба пина
            OCR0A = 0;
        }
    } else if (*pinL == L_PWM_right && *pinR == R_PWM_right) {
        if (pwmValue < 0) {
            OCR1A = -pwmValue; // Устанавливаем ШИМ на левый пин
            OCR1B = 0;         // Выключаем правый пин
        } else if (pwmValue > 0) {
            OCR1A = 0;         // Выключаем левый пин
            OCR1B = pwmValue;  // Устанавливаем ШИМ на правый пин
        } else {
            OCR1A = 0;         // Выключаем оба пина
            OCR1B = 0;
        }
    }
}

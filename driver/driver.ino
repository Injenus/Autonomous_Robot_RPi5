#include <GParser.h>
#include <AsyncStream.h>
AsyncStream<100> serial(&Serial, '\n'); // указали Stream-объект и символ конца
int num1 = 0;
int num2 = 0;
int summ = 0;
int calc_summ = 0;

// Пины для светодиодов
const int greenLedPin = 4;
const int redLedPin = 7;

// Пины для ШИМ
const int L_PWM_left = 9;   // PD5 / OC0B
const int R_PWM_left = 10;   // PD6 / OC0A
const int L_PWM_right = 6;  // PB1 / OC1A
const int R_PWM_right = 5; // PB2 / OC1B

void setup() {
  Serial.begin(115200);
  serial.setTimeout(9);    // установить другой таймаут
  //serial.setEOL(';');        // установить другой терминатор (EOL)

  pinMode(greenLedPin, OUTPUT);
  pinMode(redLedPin, OUTPUT);

  pinMode(L_PWM_left, OUTPUT);
  pinMode(R_PWM_left, OUTPUT);
  pinMode(L_PWM_right, OUTPUT);
  pinMode(R_PWM_right, OUTPUT);

  // Настройка таймеров для ШИМ  
  //setupPWM();
}


void loop() {
  if (serial.available()) {     // если данные получены
    Serial.println(serial.buf); // выводим их (как char*)
    GParser data(serial.buf, ',');
    int am = data.split();    // разделяем, получаем количество данных
    //Serial.println(am); // выводим количество
    // можем обратиться к полученным строкам как data[i] или data.str[i]
    //for (byte i = 0; i < am; i++) Serial.println(data[i]);
    if (am == 5 && data.equals(0, "s") && data.equals(4, "f")) {
      num1 = data.getInt(1);
      num2 = data.getInt(2);
      summ = data.getInt(3);
      calc_summ = num1 * 2 + num2 * 4;
      if (calc_summ != summ) {
        num1 = 0;
        num2 = 0;
      }
    }
    else {
      num1 = 0;
      num2 = 0;
    }

    // Управление ШИМ
    //updatePWM(&L_PWM_left, &R_PWM_left, &num1);
    //updatePWM(&L_PWM_right, &R_PWM_right, &num2);
    Serial.print(num1);
    Serial.print(',');
    Serial.println(num2);
    if (num1>=0){
      analogWrite(L_PWM_left, 0);
      analogWrite(R_PWM_left, num1);
    }
    else{
      analogWrite(L_PWM_left, -num1);
      analogWrite(R_PWM_left, 0);
    }
    if (num2>=0){
      analogWrite(L_PWM_right, 0);
      analogWrite(R_PWM_right, num2);
    }
    else{
      analogWrite(L_PWM_right, -num2);
      analogWrite(R_PWM_right, 0);
    }

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

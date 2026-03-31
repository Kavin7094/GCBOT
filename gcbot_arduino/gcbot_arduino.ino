#include <Servo.h>
#include "HX711.h"

/* ---------------- SERVOS ---------------- */
Servo liftA, liftB;
Servo grabA, grabB;

/* ---------------- MOTOR DRIVER ---------------- */
#define IN1 2
#define IN2 3
#define IN3 4
#define IN4 7
#define ENA 8
#define ENB 12

/* ---------------- LOAD CELL ---------------- */
#define DT A1
#define SCK A0
HX711 scale;

/* ---------------- WEIGHT ---------------- */
// Weight is ONLY read when the WEIGHTNOW command is received.
// Score is computed on the laptop (AI detection + weight combined).
float lastWeight = 0;

/* ---------------- SERIAL BUFFER ---------------- */
String command = "";

/* ---------------- SETUP ---------------- */
void setup() {
  Serial.begin(9600);

  // ---- Servos ----
  liftA.attach(9);
  liftB.attach(10);
  grabA.attach(5);
  grabB.attach(6);

  moveLift(90);
  moveGrab(90);

  // ---- Motors ----
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);

  analogWrite(ENA, 200);
  analogWrite(ENB, 200);
  stopMotors();

  // ---- Load Cell ----
  scale.begin(DT, SCK);
  // TODO: replace 2280.0 with YOUR calibration factor
  // Run a calibration sketch, place known weight, find the factor.
  scale.set_scale(2280.0);  // <-- adjust this value!
  delay(500);
  if (scale.is_ready()) {
    scale.tare();            // zero the scale on boot
    lastWeight = scale.get_units(3);
    Serial.print("W:baseline "); Serial.println(lastWeight, 2);
  } else {
    Serial.println("W:scale not ready on boot");
  }

  Serial.println("SYSTEM READY");
}

/* ---------------- SERVO FUNCTIONS ---------------- */
void moveLift(int angle) {
  angle = constrain(angle, 0, 180);
  liftA.write(angle);
  liftB.write(180 - angle);
}

void moveGrab(int angle) {
  angle = constrain(angle, 0, 180);
  grabA.write(angle);
  grabB.write(180 - angle);
}

/* ---------------- MOTOR FUNCTIONS ---------------- */
void forward()  { digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW);  digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW);  }
void backward() { digitalWrite(IN1,LOW);  digitalWrite(IN2,HIGH); digitalWrite(IN3,LOW);  digitalWrite(IN4,HIGH); }
void left()     { digitalWrite(IN1,LOW);  digitalWrite(IN2,HIGH); digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW);  }
void right()    { digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW);  digitalWrite(IN3,LOW);  digitalWrite(IN4,HIGH); }
void stopMotors(){ digitalWrite(IN1,LOW); digitalWrite(IN2,LOW);  digitalWrite(IN3,LOW);  digitalWrite(IN4,LOW);  }

/* ---------------- WEIGHT CHECK (command-triggered only) ---------------- */
// Called ONLY when WEIGHTNOW is received from the Pi.
// Sends W:<diff_grams> — laptop combines with AI detection for scoring.
void doWeightCheck() {
  if (!scale.is_ready()) {
    Serial.println("W:ERR");
    return;
  }

  float current = scale.get_units(3);   // average 3 readings
  float diff    = current - lastWeight;  // positive = object added

  // Send the diff — laptop combines this with last detected class
  Serial.print("W:"); Serial.println(diff, 2);

  // Update baseline for next reading
  if (abs(diff) > 5.0) {
    lastWeight = current;
  }
}

/* ---------------- COMMAND EXECUTION ---------------- */
void executeCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  char type = cmd.charAt(0);
  int  value = cmd.substring(1).toInt();

  if      (type == 'F') forward();
  else if (type == 'B') backward();
  else if (type == 'L') left();
  else if (type == 'R') right();
  else if (type == 'S') stopMotors();
  else if (type == 'U') moveLift(value);
  else if (type == 'G') moveGrab(value);
  else if (cmd == "WEIGHTNOW") doWeightCheck();  // only triggered by DROP OBJECT button
}

/* ---------------- LOOP ---------------- */
void loop() {
  // 1. Drain serial — always first, never skip
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      command.trim();
      if (command.length() > 0) {
        executeCommand(command);
      }
      command = "";
    } else {
      command += c;
    }
  }

  // Weight is only checked on WEIGHTNOW command — no automatic polling.
}

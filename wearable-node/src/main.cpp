#include <Arduino.h>
#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h> // Correct library for Rev 2

BLEService sensorService("180C"); 
BLEStringCharacteristic dataCharacteristic("2A56", BLERead | BLENotify, 30);

// Declare variables globally to avoid reallocation
float x, y, z;

void setup() {
  Serial.begin(9600);
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");
    while (1);
  }

  BLE.setLocalName("NanoBLE");
  BLE.setAdvertisedService(sensorService);
  sensorService.addCharacteristic(dataCharacteristic);
  BLE.addService(sensorService);
  BLE.advertise();
}

void loop() {
  BLEDevice central = BLE.central();
  if (central) {
    while (central.connected()) {
      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(x, y, z); // Updates the global variables
        String payload = String(millis()) + "," + String(x, 2) + "," + String(y, 2) + "," + String(z, 2) + ",99";
        dataCharacteristic.writeValue(payload);
        delay(20); 
      }
    }
  }
}

// #include <Arduino.h>

// // put function declarations here:
// int myFunction(int, int);

// void setup() {
//   // put your setup code here, to run once:
//   int result = myFunction(2, 3);
// }

// void loop() {
//   // put your main code here, to run repeatedly:
// }

// // put function definitions here:
// int myFunction(int x, int y) {
//   return x + y;
// }
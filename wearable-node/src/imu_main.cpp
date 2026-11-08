// #include <Arduino.h>
// #include <ArduinoBLE.h>
// #include <Arduino_BMI270_BMM150.h> // Correct library for Rev 2

// BLEService sensorService("180C"); 
// // INCREASED buffer size from 30 to 100 bytes so the larger payload doesn't get cut off
// BLEStringCharacteristic dataCharacteristic("2A56", BLERead | BLENotify, 100);

// // Declare variables globally to avoid reallocation
// float x, y, z;
// float gx, gy, gz; // Added for Gyroscope

// void setup() {
//   Serial.begin(9600);
//   if (!IMU.begin()) {
//     Serial.println("Failed to initialize IMU!");
//     while (1);
//   }
//   if (!BLE.begin()) {
//     Serial.println("starting BLE failed!");
//     while (1);
//   }

//   BLE.setLocalName("NanoBLE");
//   BLE.setAdvertisedService(sensorService);
//   sensorService.addCharacteristic(dataCharacteristic);
//   BLE.addService(sensorService);
//   BLE.advertise();
// }

// void loop() {
//   BLEDevice central = BLE.central();
//   if (central) {
//     while (central.connected()) {
//       // Check if BOTH acceleration and gyroscope data are ready
//       if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
        
//         IMU.readAcceleration(x, y, z);
//         IMU.readGyroscope(gx, gy, gz); 
        
//         // Construct the new 6-channel payload
//         // Format: Timestamp, Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z
//         String payload = String(millis()) + "," + 
//                          String(x, 2) + "," + String(y, 2) + "," + String(z, 2) + "," +
//                          String(gx, 2) + "," + String(gy, 2) + "," + String(gz, 2);
                         
//         dataCharacteristic.writeValue(payload);
        
//         // I reduced this from 20 to 15. The Arduino takes a few milliseconds to build 
//         // the string and send the Bluetooth packet. A delay of 15ms + overhead 
//         // will get you much closer to the target 50Hz (20ms total) loop speed!
//         delay(15); 
//       }
//     }
//   }
// }

// // #include <Arduino.h>

// // // put function declarations here:
// // int myFunction(int, int);

// // void setup() {
// //   // put your setup code here, to run once:
// //   int result = myFunction(2, 3);
// // }

// // void loop() {
// //   // put your main code here, to run repeatedly:
// // }

// // // put function definitions here:
// // int myFunction(int x, int y) {
// //   return x + y;
// // }
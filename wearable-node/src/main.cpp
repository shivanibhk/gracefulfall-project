#include <Arduino.h>
#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>
#include <PDM.h>

// ============================================================
// BLE IMU configuration
// ============================================================

BLEService sensorService("180C");

BLEStringCharacteristic dataCharacteristic(
    "2A56",
    BLERead | BLENotify,
    100
);

float x;
float y;
float z;

float gx;
float gy;
float gz;

unsigned long last_imu_send_time = 0;
const unsigned long IMU_INTERVAL_MS = 20;


// ============================================================
// Audio configuration
// ============================================================

const int SAMPLE_RATE = 16000;
const int BUFFER_SIZE = 512;

volatile int samples_ready = 0;
volatile bool buffer_busy = false;

int16_t audio_buffer[BUFFER_SIZE];
int16_t discard_buffer[BUFFER_SIZE];


// ============================================================
// Microphone interrupt callback
// ============================================================

void onPDMdata() {
    int bytes_available = PDM.available();

    if (bytes_available <= 0) {
        return;
    }

    int samples_available =
        bytes_available / sizeof(int16_t);

    if (samples_available > BUFFER_SIZE) {
        samples_available = BUFFER_SIZE;
    }

    if (
        samples_ready == 0
        && !buffer_busy
    ) {
        PDM.read(
            audio_buffer,
            samples_available * sizeof(int16_t)
        );

        samples_ready = samples_available;
    } else {
        // The host is not keeping up.
        // Read and discard the block.
        PDM.read(
            discard_buffer,
            samples_available * sizeof(int16_t)
        );
    }
}


// ============================================================
// Send one audio frame over USB Serial
// ============================================================

void send_audio_frame() {
    if (samples_ready <= 0) {
        return;
    }

    int16_t local_buffer[BUFFER_SIZE];
    int local_count;

    noInterrupts();

    buffer_busy = true;

    local_count = samples_ready;

    memcpy(
        local_buffer,
        audio_buffer,
        local_count * sizeof(int16_t)
    );

    samples_ready = 0;

    interrupts();

    /*
       Binary frame format:

       2 bytes: magic number 0xA55A
       4 bytes: timestamp in milliseconds
       2 bytes: number of samples
       N bytes: signed 16-bit PCM samples
    */

    uint16_t magic = 0xA55A;
    uint32_t timestamp = millis();
    uint16_t count = local_count;

    Serial.write(
        reinterpret_cast<uint8_t*>(&magic),
        sizeof(magic)
    );

    Serial.write(
        reinterpret_cast<uint8_t*>(&timestamp),
        sizeof(timestamp)
    );

    Serial.write(
        reinterpret_cast<uint8_t*>(&count),
        sizeof(count)
    );

    Serial.write(
        reinterpret_cast<uint8_t*>(local_buffer),
        count * sizeof(int16_t)
    );

    Serial.flush();

    noInterrupts();

    buffer_busy = false;

    interrupts();
}


// ============================================================
// Send one IMU packet over BLE
// ============================================================

void send_imu_sample() {
    if (
        !IMU.accelerationAvailable()
        || !IMU.gyroscopeAvailable()
    ) {
        return;
    }

    unsigned long now = millis();

    if (
        now - last_imu_send_time
        < IMU_INTERVAL_MS
    ) {
        return;
    }

    last_imu_send_time = now;

    IMU.readAcceleration(
        x,
        y,
        z
    );

    IMU.readGyroscope(
        gx,
        gy,
        gz
    );

    /*
       BLE format:

       timestamp,ax,ay,az,gx,gy,gz
    */

    String payload =
        String(now) + ","
        + String(x, 2) + ","
        + String(y, 2) + ","
        + String(z, 2) + ","
        + String(gx, 2) + ","
        + String(gy, 2) + ","
        + String(gz, 2);

    dataCharacteristic.writeValue(
        payload
    );
}


// ============================================================
// Setup
// ============================================================

void setup() {
    /*
       The audio Python script expects 921600 baud rate.
    */
    Serial.begin(921600);

    while (
        !Serial
        && millis() < 3000
    ) {
        delay(10);
    }


    // Initialize IMU
    if (!IMU.begin()) {
        Serial.println(
            "IMU initialization failed"
        );

        while (true) {
            delay(1000);
        }
    }


    // Initialize BLE
    if (!BLE.begin()) {
        Serial.println(
            "BLE initialization failed"
        );

        while (true) {
            delay(1000);
        }
    }

    BLE.setLocalName("NanoBLE");

    BLE.setAdvertisedService(
        sensorService
    );

    sensorService.addCharacteristic(
        dataCharacteristic
    );

    BLE.addService(
        sensorService
    );

    BLE.advertise();


    // Initialize microphone
    PDM.onReceive(
        onPDMdata
    );

    if (!PDM.begin(1, SAMPLE_RATE)) {
        Serial.println(
            "PDM microphone initialization failed"
        );

        while (true) {
            delay(1000);
        }
    }

    /*
       Use 10 instead of 30 initially.
    */
    PDM.setGain(10);

    Serial.println(
        "IMU_READY"
    );

    Serial.println(
        "AUDIO_READY"
    );
}


// ============================================================
// Main loop
// ============================================================

void loop() {
    /*
       Keep BLE processing active.
    */
    BLE.poll();

    /*
       If a BLE central is connected, send IMU data.
       Audio is sent regardless of BLE connection.
    */
    send_imu_sample();

    send_audio_frame();

    delay(1);
}
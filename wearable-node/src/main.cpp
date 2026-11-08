#include <Arduino.h>
#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>
#include <PDM.h>
#include <cstring>

// BLE IMU configuration
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


// Audio configuration
const int SAMPLE_RATE = 16000;
const int BUFFER_SIZE = 512;

const int AUDIO_BLOCK_COUNT = 4;

int16_t audio_blocks[
    AUDIO_BLOCK_COUNT
][BUFFER_SIZE];

uint16_t audio_block_counts[
    AUDIO_BLOCK_COUNT
];

int16_t discard_buffer[BUFFER_SIZE];

volatile uint8_t audio_write_index = 0;
volatile uint8_t audio_read_index = 0;
volatile uint8_t audio_ready_blocks = 0;

volatile uint32_t dropped_audio_blocks = 0;


// Streaming control
volatile bool streaming_enabled = false;


// Microphone interrupt callback
void onPDMdata() {
    int bytes_available = PDM.available();

    if (bytes_available <= 0) {
        return;
    }

    int samples_available =
        bytes_available / sizeof(int16_t);

    if (
        samples_available
        > BUFFER_SIZE
    ) {
        samples_available = BUFFER_SIZE;
    }

    /*
       Before START, consume and discard microphone data
       so the PDM buffer does not overflow.
    */
    if (!streaming_enabled) {
        PDM.read(
            discard_buffer,
            samples_available * sizeof(int16_t)
        );

        return;
    }

    /*
       Store the block if the ring buffer has space.
    */
    if (
        audio_ready_blocks
        < AUDIO_BLOCK_COUNT
    ) {
        PDM.read(
            audio_blocks[audio_write_index],
            samples_available * sizeof(int16_t)
        );

        audio_block_counts[
            audio_write_index
        ] = samples_available;

        audio_write_index =
            (
                audio_write_index + 1
            )
            % AUDIO_BLOCK_COUNT;

        audio_ready_blocks++;
    }

    /*
       If all blocks are full, discard this block.
    */
    else {
        PDM.read(
            discard_buffer,
            samples_available * sizeof(int16_t)
        );

        dropped_audio_blocks++;
    }
}


// Process commands sent from Python over USB
void process_host_commands() {
    static char command_buffer[32];
    static int command_length = 0;

    while (Serial.available() > 0) {
        char character = Serial.read();

        if (
            character == '\n'
            || character == '\r'
        ) {
            if (command_length == 0) {
                continue;
            }

            command_buffer[
                command_length
            ] = '\0';

            if (
                strcmp(
                    command_buffer,
                    "START"
                ) == 0
            ) {
                noInterrupts();

                audio_write_index = 0;
                audio_read_index = 0;
                audio_ready_blocks = 0;
                dropped_audio_blocks = 0;

                streaming_enabled = true;

                interrupts();

                Serial.println(
                    "STREAM_STARTED"
                );
            }

            else if (
                strcmp(
                    command_buffer,
                    "STOP"
                ) == 0
            ) {
                noInterrupts();

                streaming_enabled = false;
                audio_ready_blocks = 0;

                interrupts();

                Serial.println(
                    "STREAM_STOPPED"
                );
            }

            command_length = 0;
        }

        else if (
            command_length
            < sizeof(command_buffer) - 1
        ) {
            command_buffer[
                command_length
            ] = character;

            command_length++;
        }
    }
}


// Send one binary audio frame over USB Serial
void send_audio_frame() {
    if (!streaming_enabled) {
        return;
    }

    int16_t local_buffer[BUFFER_SIZE];
    int local_count;

    noInterrupts();

    if (audio_ready_blocks == 0) {
        interrupts();
        return;
    }

    local_count =
        audio_block_counts[audio_read_index];

    memcpy(
        local_buffer,
        audio_blocks[audio_read_index],
        local_count * sizeof(int16_t)
    );

    audio_read_index =
        (
            audio_read_index + 1
        )
        % AUDIO_BLOCK_COUNT;

    audio_ready_blocks--;

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
}


// Send one IMU sample over BLE
void send_imu_sample() {
    if (!streaming_enabled) {
        return;
    }

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

    // BLE packet format: timestamp,ax,ay,az,gx,gy,gz
    char payload[100];

    snprintf(
        payload,
        sizeof(payload),
        "%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",
        now,
        x,
        y,
        z,
        gx,
        gy,
        gz
    );

    dataCharacteristic.writeValue(
        payload
    );
}


// Setup
void setup() {
    // The Python audio receiver expects 921600 baud.
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

    PDM.setGain(10);

    streaming_enabled = false;

    Serial.println("IMU_READY");
    Serial.println("AUDIO_READY");
    Serial.println("WAITING_FOR_START");
}



// Main loop
void loop() {
    BLE.poll();

    process_host_commands();

    send_imu_sample();

    send_audio_frame();

    delay(1);
}
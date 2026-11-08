#include <Arduino.h>
#include <PDM.h>

const int SAMPLE_RATE = 16000;
const int BUFFER_SIZE = 512;

volatile int samples_ready = 0;
volatile bool buffer_busy = false;

int16_t audio_buffer[BUFFER_SIZE];

void onPDMdata() {
    int bytes_available = PDM.available();

    if (bytes_available <= 0) {
        return;
    }

    int samples_available = bytes_available / sizeof(int16_t);

    if (samples_available > BUFFER_SIZE) {
        samples_available = BUFFER_SIZE;
    }

    if (samples_ready == 0 && !buffer_busy) {
        PDM.read(audio_buffer,
                 samples_available * sizeof(int16_t));

        samples_ready = samples_available;
    } else {
        // Discard this block if the host is not keeping up.
        int16_t discard_buffer[BUFFER_SIZE];

        PDM.read(
            discard_buffer,
            samples_available * sizeof(int16_t)
        );
    }
}

void setup() {
    Serial.begin(921600);

    while (!Serial && millis() < 3000) {
        delay(10);
    }

    PDM.onReceive(onPDMdata);

    if (!PDM.begin(1, SAMPLE_RATE)) {
        Serial.println("PDM microphone initialization failed");
        while (true) {
            delay(1000);
        }
    }

    PDM.setGain(30);

    Serial.println("AUDIO_READY");
}

void loop() {
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
from sensors.imu_stream import IMUBuffer


buffer = IMUBuffer(
    window_size=100
)

for index in range(100):
    packet = (
        f"{index * 20},"
        "0.0,0.0,1.0,"
        "0.0,0.0,0.0"
    ).encode("utf-8")

    buffer.parse_packet(
        sender=None,
        data=packet,
    )

assert buffer.ready()

window = buffer.snapshot()

assert window.shape == (100, 6)
assert buffer.stats.sample_count == 100

print("IMU buffer test passed")
print(f"Window shape: {window.shape}")
print(f"Samples: {buffer.stats.sample_count}")
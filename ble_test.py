import asyncio
import time
from bleak import BleakClient, BleakScanner

# Standard UUID used in the Arduino code
CHAR_UUID = "00002A56-0000-1000-8000-00805f9b34fb"

packet_count = 0
start_time = 0

def handle_data(sender, data):
    global packet_count, start_time
    if packet_count == 0:
        start_time = time.time()
    packet_count += 1
    try:
        decoded = data.decode("utf-8").strip()
    except UnicodeDecodeError:
        decoded = repr(data)

    # Calculate frequency
    elapsed = time.time() - start_time
    frequency = ( packet_count / elapsed if elapsed > 0 else 0)

    print(f"[{elapsed:.2f}s] Received: {decoded} | Rate: {frequency:.1f} Hz")


async def main():
    print("Scanning for NanoBLE...")

    discovered = await BleakScanner.discover(
        timeout=15,
        return_adv=True,
    )

    nano = None

    for address, result in discovered.items():
        device, advertisement = result

        device_name = device.name
        local_name = advertisement.local_name

        if (
            device_name == "NanoBLE"
            or local_name == "NanoBLE"
        ):
            nano = device
            break

    if nano is None:
        print(
            "\nNanoBLE was not found."
            "\nCheck power, reset state, Bluetooth permissions, "
            "and whether another device is connected."
        )
        return

    print(
        f"\nFound NanoBLE at {nano.address}"
    )
    print("Connecting...")

    async with BleakClient(nano.address) as client:
        print(
            "Connected. Listening for data "
            "for 10 seconds.\n"
        )

        await client.start_notify(
            CHAR_UUID,
            handle_data,
        )

        await asyncio.sleep(10)

        await client.stop_notify(CHAR_UUID)

        print(
            f"\nTest complete. "
            f"Packets received: {packet_count}"
        )


if __name__ == "__main__":
    asyncio.run(main())
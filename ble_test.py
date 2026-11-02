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
    decoded = data.decode('utf-8')
    
    # Calculate frequency
    elapsed = time.time() - start_time
    freq = packet_count / elapsed if elapsed > 0 else 0
    
    print(f"[{elapsed:.2f}s] Received: {decoded} | Rate: {freq:.1f} Hz")

async def main():
    print("Scanning for NanoBLE...")
    devices = await BleakScanner.discover()
    nano = next((d for d in devices if d.name == "NanoBLE"), None)
    
    if not nano:
        print("Could not find the Arduino. Make sure it is powered on and flashing.")
        return
        
    print(f"Found {nano.name}. Connecting...")
    async with BleakClient(nano.address) as client:
        print("Connected! Listening for data for 10 seconds...\n")
        await client.start_notify(CHAR_UUID, handle_data)
        await asyncio.sleep(10)
        await client.stop_notify(CHAR_UUID)
        print(f"\nTest complete. Total packets received: {packet_count}")

asyncio.run(main())
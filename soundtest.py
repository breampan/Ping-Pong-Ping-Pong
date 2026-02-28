import asyncio
import struct
import time
import random
import numpy as np
import sounddevice as sd
from bleak import BleakClient, BleakScanner

# --- 音訊全域設定 ---
SAMPLERATE = 44100
CHANNELS = 2
poly_voices = {} 
last_accel = {}    
last_trigger = {}  

# --- 隨機 Echo 緩衝區設定 ---
MAX_DELAY_SEC = 0.5 
MAX_DELAY_SAMPLES = int(SAMPLERATE * MAX_DELAY_SEC)
delay_buffer = np.zeros((MAX_DELAY_SAMPLES, CHANNELS))
delay_ptr = 0
current_delay_samples = int(SAMPLERATE * 0.3) # 初始間隔

class FMVoice:
    def __init__(self, voice_id):
        self.id = voice_id
        self.freq = 880.0
        self.mod_index = 0.8
        self.ratio = 11.72  
        self.phase_c = 0
        self.phase_m = 0
        self.amp = 0.0
        self.decay_rate = 0.9998
        self.cutoff = 2000.0      
        self.last_out = 0.0
        self.pan = random.uniform(0.2, 0.8) 

    def trigger(self, power):
        global current_delay_samples
        self.amp = min(1.0, power / 3.0 + 0.2)
        
        # 1. 限制聲音長度在 2.5 秒內隨機 (調大衰減係數)
        # 0.99975 (~1.2s) 到 0.99988 (~2.5s)
        self.decay_rate = random.uniform(0.99975, 0.99988)
        
        # 2. 每次觸發都改變下一次 Echo 的時間間隔 (可愛的隨機感)
        random_sec = random.uniform(0.15, 0.45)
        current_delay_samples = int(SAMPLERATE * random_sec)
        
        self.ratio = 11.72 + random.uniform(-0.1, 0.1)

    def next_block(self, frames):
        t = (np.arange(frames) / SAMPLERATE)
        mod_freq = self.freq * self.ratio
        m_vals = np.sin(self.phase_m + 2 * np.pi * mod_freq * t) * self.mod_index
        raw_out = np.sin(self.phase_c + 2 * np.pi * self.freq * t + m_vals) * self.amp
        
        alpha = self.cutoff / (self.cutoff + SAMPLERATE / (2 * np.pi))
        filtered_out = np.zeros(frames)
        current_last = self.last_out
        for i in range(frames):
            current_last = current_last + alpha * (raw_out[i] - current_last)
            filtered_out[i] = current_last
        self.last_out = current_last
        
        self.phase_c = (self.phase_c + 2 * np.pi * self.freq * frames / SAMPLERATE) % (2 * np.pi)
        self.phase_m = (self.phase_m + 2 * np.pi * mod_freq * frames / SAMPLERATE) % (2 * np.pi)
        
        self.amp *= (self.decay_rate ** frames)
        if self.amp < 0.0005: self.amp = 0
        
        stereo_out = np.zeros((frames, 2))
        stereo_out[:, 0] = filtered_out * (1.0 - self.pan)
        stereo_out[:, 1] = filtered_out * self.pan
        return stereo_out

def audio_callback(outdata, frames, time, status):
    global delay_ptr
    mixed_out = np.zeros((frames, 2))
    for v in poly_voices.values():
        mixed_out += v.next_block(frames)
    
    for i in range(frames):
        # 使用動態的間隔長度來讀取延遲訊號
        read_ptr = (delay_ptr - current_delay_samples) % MAX_DELAY_SAMPLES
        delayed_signal = delay_buffer[read_ptr]
        
        outdata[i] = mixed_out[i] * 0.6 + delayed_signal * 0.35
        
        # Feedback 加入隨機微調
        dynamic_feedback = 0.42 + random.uniform(-0.03, 0.03)
        delay_buffer[delay_ptr] = (mixed_out[i] + delayed_signal * dynamic_feedback)
        delay_ptr = (delay_ptr + 1) % MAX_DELAY_SAMPLES

def handle_imu_data(imu_id, data):
    if len(data) < 20 or data[0] != 0x55 or data[1] != 0x61: return
    vals = struct.unpack('<hhhhhhhhh', data[2:20])
    ax, ay, az = [v / 32768.0 * 16 for v in vals[0:3]]
    current_g = (ax**2 + ay**2 + az**2)**0.5
    roll = vals[6] / 32768.0 * 180 
    pitch = vals[7] / 32768.0 * 180

    if imu_id in poly_voices:
        v = poly_voices[imu_id]
        v.freq = 600 + (pitch + 90) * 12 
        v.cutoff = 2000 + abs(roll) * 45
        
        now = time.time()
        prev_g = last_accel.get(imu_id, 1.0)
        delta_g = abs(current_g - prev_g)
        last_accel[imu_id] = current_g
        
        if (current_g > 1.8 or delta_g > 0.8) and (now - last_trigger.get(imu_id, 0) > 0.15):
            v.trigger(current_g) 
            last_trigger[imu_id] = now
            print(f"💧 IMU {imu_id} Jittery Trigger! G:{current_g:.2f}")

async def connect_imu(device, imu_id):
    WRITE_CHAR = "0000ffe9-0000-1000-8000-00805f9a34fb"
    NOTIFY_CHAR = "0000ffe4-0000-1000-8000-00805f9a34fb"
    try:
        async with BleakClient(device) as client:
            poly_voices[imu_id] = FMVoice(imu_id)
            print(f"✅ IMU {imu_id} OK")
            await client.start_notify(NOTIFY_CHAR, lambda s, d: handle_imu_data(imu_id, d))
            await client.write_gatt_char(WRITE_CHAR, bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5]))
            while client.is_connected: await asyncio.sleep(1)
    except Exception: pass
    finally: poly_voices.pop(imu_id, None)

async def manager():
    with sd.OutputStream(channels=2, callback=audio_callback, samplerate=SAMPLERATE):
        print("=== Sonic Squid: Jittery Rain Edition 啟動 ===")
        connected_addresses = set()
        while True:
            if len(poly_voices) < 4:
                devices = await BleakScanner.discover(timeout=1.0)
                for d in devices:
                    if d.name and d.name.startswith("WT") and d.address not in connected_addresses:
                        imu_id = len(poly_voices) + 1
                        connected_addresses.add(d.address)
                        asyncio.create_task(connect_imu(d, imu_id))
            await asyncio.sleep(3)

if __name__ == "__main__":
    try: asyncio.run(manager())
    except KeyboardInterrupt: print("\n已停止。")
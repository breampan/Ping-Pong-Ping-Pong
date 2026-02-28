import asyncio
import struct
import time
import random
import numpy as np
import sounddevice as sd
from bleak import BleakClient, BleakScanner

SAMPLERATE = 44100
CHANNELS = 2
poly_voices = {} 
last_accel = {}    
last_trigger = {}  
connected_addresses = set()

MAX_DELAY_SEC = 0.5 
MAX_DELAY_SAMPLES = int(SAMPLERATE * MAX_DELAY_SEC)
delay_buffer = np.zeros((MAX_DELAY_SAMPLES, CHANNELS))
delay_ptr = 0
current_delay_samples = int(SAMPLERATE * 0.3)

class FMVoice:
    def __init__(self, voice_id):
        self.id = voice_id
        
        # 高頻 FM 設定
        self.freq = 880.0
        self.mod_index = 0.8
        self.ratio = 11.72  
        self.phase_c = 0
        self.phase_m = 0
        
        # 80Hz 低頻設定
        self.sub_freq = 80.0 
        self.phase_sub = 0
        self.sub_gain = 0.25  # 獨立調整 80Hz 的喇叭推力 (若水波不夠明顯可調高至 0.5)
        
        # 完整的 Sample-accurate ADSR 狀態機
        self.state = 'IDLE'
        self.current_amp = 0.0
        self.target_amp = 0.0
        self.attack_step = 0.0
        self.decay_rate = 0.9998
        
        self.cutoff = 2000.0      
        self.last_out = 0.0
        self.pan = random.uniform(0.2, 0.8) 

    def trigger(self, power):
        global current_delay_samples
        self.target_amp = min(1.0, power / 3.0 + 0.2)
        
        # 隨機化 ADSR attack time (80ms ~ 180ms)，讓起振像呼吸般自然
        attack_sec = random.uniform(0.08, 0.18)
        self.attack_step = self.target_amp / (SAMPLERATE * attack_sec)
        
        self.decay_rate = random.uniform(0.99975, 0.99988)
        self.state = 'ATTACK'  # 切換至 Attack 狀態
        
        random_sec = random.uniform(0.15, 0.45)
        current_delay_samples = int(SAMPLERATE * random_sec)
        self.ratio = 11.72 + random.uniform(-0.1, 0.1)

    def next_block(self, frames):
        # 1. 產生高解析度平滑包絡線 (消除所有物理爆音)
        env = np.zeros(frames)
        for i in range(frames):
            if self.state == 'ATTACK':
                self.current_amp += self.attack_step
                if self.current_amp >= self.target_amp:
                    self.current_amp = self.target_amp
                    self.state = 'DECAY'
            elif self.state == 'DECAY':
                self.current_amp *= self.decay_rate
                if self.current_amp < 0.0001:
                    self.current_amp = 0.0
                    self.state = 'IDLE'
            env[i] = self.current_amp

        t = (np.arange(frames) / SAMPLERATE)
        
        # 2. 生成波形 (套用平滑包絡線)
        mod_freq = self.freq * self.ratio
        m_vals = np.sin(self.phase_m + 2 * np.pi * mod_freq * t) * self.mod_index
        raw_fm = np.sin(self.phase_c + 2 * np.pi * self.freq * t + m_vals) * env
        
        # 80Hz 獨立乘上 sub_gain，不再過載喇叭
        raw_sub = np.sin(self.phase_sub + 2 * np.pi * self.sub_freq * t) * env * self.sub_gain
        
        # Resonance 濾波
        alpha = self.cutoff / (self.cutoff + SAMPLERATE / (2 * np.pi))
        filtered_fm = np.zeros(frames)
        current_last = self.last_out
        for i in range(frames):
            current_last = current_last + alpha * (raw_fm[i] - current_last)
            filtered_fm[i] = current_last
        self.last_out = current_last
        
        # 更新相位
        self.phase_c = (self.phase_c + 2 * np.pi * self.freq * frames / SAMPLERATE) % (2 * np.pi)
        self.phase_m = (self.phase_m + 2 * np.pi * mod_freq * frames / SAMPLERATE) % (2 * np.pi)
        self.phase_sub = (self.phase_sub + 2 * np.pi * self.sub_freq * frames / SAMPLERATE) % (2 * np.pi)
        
        # 空間分配
        fm_stereo = np.zeros((frames, 2))
        fm_stereo[:, 0] = filtered_fm * (1.0 - self.pan)
        fm_stereo[:, 1] = filtered_fm * self.pan
        
        sub_stereo = np.zeros((frames, 2))
        sub_stereo[:, 0] = raw_sub
        sub_stereo[:, 1] = raw_sub
        
        return fm_stereo, sub_stereo

def audio_callback(outdata, frames, time, status):
    global delay_ptr
    mixed_fm = np.zeros((frames, 2))
    mixed_sub = np.zeros((frames, 2))
    
    for v in poly_voices.values():
        fm_out, sub_out = v.next_block(frames)
        mixed_fm += fm_out
        mixed_sub += sub_out
    
    for i in range(frames):
        read_ptr = (delay_ptr - current_delay_samples) % MAX_DELAY_SAMPLES
        delayed_signal = delay_buffer[read_ptr]
        
        # 最終混音，低頻(mixed_sub)穩定輸出不進 Echo
        outdata[i] = mixed_fm[i] * 0.6 + delayed_signal * 0.35 + mixed_sub[i] * 1.0
        
        dynamic_feedback = 0.42 + random.uniform(-0.03, 0.03)
        delay_buffer[delay_ptr] = (mixed_fm[i] + delayed_signal * dynamic_feedback)
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
            print(f"💧 IMU {imu_id} Soft Trigger! G:{current_g:.2f}")

async def connect_imu(device, imu_id):
    WRITE_CHAR = "0000ffe9-0000-1000-8000-00805f9a34fb"
    NOTIFY_CHAR = "0000ffe4-0000-1000-8000-00805f9a34fb"
    try:
        async with BleakClient(device) as client:
            poly_voices[imu_id] = FMVoice(imu_id)
            print(f"✅ IMU {imu_id} 就緒 ({device.name})")
            await client.start_notify(NOTIFY_CHAR, lambda s, d: handle_imu_data(imu_id, d))
            await client.write_gatt_char(WRITE_CHAR, bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5]))
            while client.is_connected: 
                await asyncio.sleep(1)
    except Exception as e: 
        print(f"❌ IMU {imu_id} 連線中斷: {e}")
    finally: 
        poly_voices.pop(imu_id, None)
        connected_addresses.discard(device.address)
        print(f"ℹ️ IMU {imu_id} 等待重新連線")

async def manager():
    with sd.OutputStream(channels=2, callback=audio_callback, samplerate=SAMPLERATE):
        print("=== 80Hz Soft-Attack 水波驅動引擎 啟動 ===")
        while True:
            if len(poly_voices) < 4:
                devices = await BleakScanner.discover(timeout=1.0)
                for d in devices:
                    if d.name and d.name.startswith("WT") and d.address not in connected_addresses:
                        used_ids = set(poly_voices.keys())
                        if len(used_ids) >= 4: break
                        new_id = next(i for i in range(1, 5) if i not in used_ids)
                        connected_addresses.add(d.address)
                        asyncio.create_task(connect_imu(d, new_id))
                await asyncio.sleep(0.2)
            else:
                await asyncio.sleep(2.0)

if __name__ == "__main__":
    try: asyncio.run(manager())
    except KeyboardInterrupt: print("\n已停止。")
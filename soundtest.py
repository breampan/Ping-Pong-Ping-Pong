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
reserved_ids = set() 

MAX_DELAY_SEC = 0.5 
MAX_DELAY_SAMPLES = int(SAMPLERATE * MAX_DELAY_SEC)
delay_buffer = np.zeros((MAX_DELAY_SAMPLES, CHANNELS))
delay_ptr = 0
current_delay_samples = int(SAMPLERATE * 0.3)

SUB_GAIN = 0.2  

class FMVoice:
    def __init__(self, voice_id):
        self.id = voice_id
        
        # --- 核心修改 1：根據 IMU 1~4 給予不同音色 ---
        if voice_id == 1:
            self.base_ratio = 11.72 # 晶瑩玻璃
            self.mod_index = 0.8
            self.base_freq = 600.0
        elif voice_id == 2:
            self.base_ratio = 3.41  # 木質馬林巴
            self.mod_index = 1.2
            self.base_freq = 400.0
        elif voice_id == 3:
            self.base_ratio = 7.13  # 金屬鐘聲
            self.mod_index = 0.6
            self.base_freq = 550.0
        else: # voice_id == 4
            self.base_ratio = 2.0   # 溫潤空靈
            self.mod_index = 0.4
            self.base_freq = 350.0

        self.freq = self.base_freq
        self.sub_freq = 80.0  
        self.ratio = self.base_ratio
        
        self.phase_c = 0
        self.phase_m = 0
        self.phase_sub = 0
        
        self.state = 'IDLE'
        self.current_amp = 0.0
        self.target_amp = 0.0
        self.attack_step = 0.0
        self.decay_rate = 0.9998
        
        self.cutoff = 2000.0      
        self.last_out = 0.0
        
        # 為了讓 Ping-Pong 更明顯，觸發時隨機偏向極左或極右
        self.pan = random.choice([random.uniform(0.1, 0.3), random.uniform(0.7, 0.9)])

    def trigger(self, power):
        global current_delay_samples
        
        self.target_amp = min(0.8, power / 4.0 + 0.15)
        
        # 軟打擊的 ADSR attack time
        attack_sec = random.uniform(0.03, 0.07)
        self.attack_step = self.target_amp / (SAMPLERATE * attack_sec)
        
        self.decay_rate = random.uniform(0.99975, 0.99988)
        self.state = 'ATTACK'  
        
        random_sec = random.uniform(0.15, 0.40)
        current_delay_samples = int(SAMPLERATE * random_sec)
        # 音色微調
        self.ratio = self.base_ratio + random.uniform(-0.05, 0.05)
        # 每次觸發重新決定起始位置，增加跳躍感
        self.pan = random.choice([random.uniform(0.1, 0.3), random.uniform(0.7, 0.9)])

    def next_block(self, frames):
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
        
        mod_freq = self.freq * self.ratio
        m_vals = np.sin(self.phase_m + 2 * np.pi * mod_freq * t) * self.mod_index
        raw_fm = np.sin(self.phase_c + 2 * np.pi * self.freq * t + m_vals) * env
        
        # 平行低頻，不受音色差異影響
        raw_sub = np.sin(self.phase_sub + 2 * np.pi * self.sub_freq * t) * env * SUB_GAIN
        
        alpha = self.cutoff / (self.cutoff + SAMPLERATE / (2 * np.pi))
        filtered_fm = np.zeros(frames)
        current_last = self.last_out
        for i in range(frames):
            current_last = current_last + alpha * (raw_fm[i] - current_last)
            filtered_fm[i] = current_last
        self.last_out = current_last
        
        self.phase_c = (self.phase_c + 2 * np.pi * self.freq * frames / SAMPLERATE) % (2 * np.pi)
        self.phase_m = (self.phase_m + 2 * np.pi * mod_freq * frames / SAMPLERATE) % (2 * np.pi)
        self.phase_sub = (self.phase_sub + 2 * np.pi * self.sub_freq * frames / SAMPLERATE) % (2 * np.pi)
        
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
        
        outdata[i, 0] = mixed_fm[i, 0] * 0.6 + delayed_signal[0] * 0.4 + mixed_sub[i, 0]
        outdata[i, 1] = mixed_fm[i, 1] * 0.6 + delayed_signal[1] * 0.4 + mixed_sub[i, 1]
        
        # --- 核心修改 2：Ping-Pong 交叉反饋 ---
        dynamic_feedback = 0.45 + random.uniform(-0.02, 0.02)
        # 左聲道緩衝區吃右聲道的延遲，右聲道緩衝區吃左聲道的延遲
        delay_buffer[delay_ptr, 0] = mixed_fm[i, 0] + delayed_signal[1] * dynamic_feedback
        delay_buffer[delay_ptr, 1] = mixed_fm[i, 1] + delayed_signal[0] * dynamic_feedback
        
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
        
        v.freq = v.base_freq + (pitch + 90) * 8 
        v.sub_freq = 65.0 + (pitch + 90) * 0.15 
        v.cutoff = 2000 + abs(roll) * 45
        
        now = time.time()
        prev_g = last_accel.get(imu_id, 1.0)
        delta_g = abs(current_g - prev_g)
        last_accel[imu_id] = current_g
        
        if (current_g > 1.8 or delta_g > 0.8) and (now - last_trigger.get(imu_id, 0) > 0.15):
            v.trigger(current_g) 
            last_trigger[imu_id] = now
            print(f"🎵 IMU {imu_id} Pluck! High:{v.freq:.0f}Hz / Sub:{v.sub_freq:.1f}Hz")

async def connect_imu(device, imu_id):
    WRITE_CHAR = "0000ffe9-0000-1000-8000-00805f9a34fb"
    NOTIFY_CHAR = "0000ffe4-0000-1000-8000-00805f9a34fb"
    try:
        async with BleakClient(device) as client:
            poly_voices[imu_id] = FMVoice(imu_id)
            reserved_ids.discard(imu_id)
            print(f"✅ IMU {imu_id} 專屬音色就緒 ({device.name})")
            
            await client.start_notify(NOTIFY_CHAR, lambda s, d: handle_imu_data(imu_id, d))
            await client.write_gatt_char(WRITE_CHAR, bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5]))
            
            while client.is_connected: 
                await asyncio.sleep(1)
    except Exception as e: 
        print(f"❌ IMU {imu_id} 連線中斷: {e}")
    finally: 
        poly_voices.pop(imu_id, None)
        reserved_ids.discard(imu_id)
        connected_addresses.discard(device.address)
        print(f"ℹ️ IMU {imu_id} 等待重新連線")

async def manager():
    with sd.OutputStream(channels=2, callback=audio_callback, samplerate=SAMPLERATE):
        print("=== 四重奏 Ping-Pong 水波引擎 啟動 ===")
        while True:
            if len(poly_voices) + len(reserved_ids) < 4:
                devices = await BleakScanner.discover(timeout=1.0)
                for d in devices:
                    if d.name and d.name.startswith("WT") and d.address not in connected_addresses:
                        used_ids = set(poly_voices.keys()).union(reserved_ids)
                        if len(used_ids) >= 4: break
                        new_id = next(i for i in range(1, 5) if i not in used_ids)
                        connected_addresses.add(d.address)
                        reserved_ids.add(new_id)
                        asyncio.create_task(connect_imu(d, new_id))
                await asyncio.sleep(0.2)
            else:
                await asyncio.sleep(2.0)

if __name__ == "__main__":
    try: asyncio.run(manager())
    except KeyboardInterrupt: print("\n已停止。")
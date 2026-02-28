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

# --- 防搶號機制 ---
connected_addresses = set()
reserved_ids = set() 

MAX_DELAY_SEC = 0.5 
MAX_DELAY_SAMPLES = int(SAMPLERATE * MAX_DELAY_SEC)
delay_buffer = np.zeros((MAX_DELAY_SAMPLES, CHANNELS))
delay_ptr = 0
current_delay_samples = int(SAMPLERATE * 0.3)

# --- 全局水波低頻設定 ---
global_sub_phase = 0.0
env_follower = 0.0
SUB_GAIN = 0.15  # 低頻強度 (若覺得水花太大可再調低)

class FMVoice:
    def __init__(self, voice_id):
        self.id = voice_id
        
        # 頻率設定
        self.freq = 600.0
        self.sub_freq = 50.0  # 初始化低頻
        
        self.mod_index = 0.8
        self.ratio = 11.72  
        self.phase_c = 0
        self.phase_m = 0
        
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
        
        attack_sec = random.uniform(0.08, 0.18)
        self.attack_step = self.target_amp / (SAMPLERATE * attack_sec)
        
        self.decay_rate = random.uniform(0.99975, 0.99988)
        self.state = 'ATTACK'  
        
        random_sec = random.uniform(0.15, 0.45)
        current_delay_samples = int(SAMPLERATE * random_sec)
        self.ratio = 11.72 + random.uniform(-0.1, 0.1)

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
        
        alpha = self.cutoff / (self.cutoff + SAMPLERATE / (2 * np.pi))
        filtered_fm = np.zeros(frames)
        current_last = self.last_out
        for i in range(frames):
            current_last = current_last + alpha * (raw_fm[i] - current_last)
            filtered_fm[i] = current_last
        self.last_out = current_last
        
        self.phase_c = (self.phase_c + 2 * np.pi * self.freq * frames / SAMPLERATE) % (2 * np.pi)
        self.phase_m = (self.phase_m + 2 * np.pi * mod_freq * frames / SAMPLERATE) % (2 * np.pi)
        
        fm_stereo = np.zeros((frames, 2))
        fm_stereo[:, 0] = filtered_fm * (1.0 - self.pan)
        fm_stereo[:, 1] = filtered_fm * self.pan
        
        return fm_stereo

def audio_callback(outdata, frames, time, status):
    global delay_ptr, global_sub_phase, env_follower
    mixed_fm = np.zeros((frames, 2))
    
    # 這裡計算當前最高的 Sub 頻率，或者取平均值 (這裡我們取第一顆活躍傳感器的頻率作為主水波頻率，避免多重低頻打架)
    active_sub_freq = 50.0 
    for v in poly_voices.values():
        mixed_fm += v.next_block(frames)
        if v.current_amp > 0.01: 
            active_sub_freq = v.sub_freq # 追蹤活躍的低頻
    
    for i in range(frames):
        read_ptr = (delay_ptr - current_delay_samples) % MAX_DELAY_SAMPLES
        delayed_signal = delay_buffer[read_ptr]
        
        final_fm_L = mixed_fm[i, 0] * 0.6 + delayed_signal[0] * 0.35
        final_fm_R = mixed_fm[i, 1] * 0.6 + delayed_signal[1] * 0.35
        
        # 音量平行：計算包絡線
        current_fm_amp = (abs(final_fm_L) + abs(final_fm_R)) * 0.5
        env_follower = env_follower * 0.999 + current_fm_amp * 0.001
        
        # 頻率平行：使用動態跟隨的高頻比例來生成低頻
        global_sub_phase = (global_sub_phase + 2 * np.pi * active_sub_freq / SAMPLERATE) % (2 * np.pi)
        sub_out = np.sin(global_sub_phase) * env_follower * SUB_GAIN
        
        outdata[i, 0] = final_fm_L + sub_out
        outdata[i, 1] = final_fm_R + sub_out
        
        dynamic_feedback = 0.42 + random.uniform(-0.03, 0.03)
        delay_buffer[delay_ptr, 0] = mixed_fm[i, 0] + delayed_signal[0] * dynamic_feedback
        delay_buffer[delay_ptr, 1] = mixed_fm[i, 1] + delayed_signal[1] * dynamic_feedback
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
        
        # 核心修改：高頻與低頻的頻率平行聯動
        v.freq = 600 + (pitch + 90) * 12 
        # 低頻永遠是高頻的 1/12 (例如高頻 600Hz 時，低頻 50Hz；高頻 1200Hz 時，低頻 100Hz)
        v.sub_freq = v.freq / 12.0 
        
        v.cutoff = 2000 + abs(roll) * 45
        
        now = time.time()
        prev_g = last_accel.get(imu_id, 1.0)
        delta_g = abs(current_g - prev_g)
        last_accel[imu_id] = current_g
        
        if (current_g > 1.8 or delta_g > 0.8) and (now - last_trigger.get(imu_id, 0) > 0.15):
            v.trigger(current_g) 
            last_trigger[imu_id] = now
            print(f"💧 IMU {imu_id} Pitch Trigger! High:{v.freq:.0f}Hz / Sub:{v.sub_freq:.0f}Hz")

async def connect_imu(device, imu_id):
    WRITE_CHAR = "0000ffe9-0000-1000-8000-00805f9a34fb"
    NOTIFY_CHAR = "0000ffe4-0000-1000-8000-00805f9a34fb"
    try:
        async with BleakClient(device) as client:
            poly_voices[imu_id] = FMVoice(imu_id)
            reserved_ids.discard(imu_id)
            print(f"✅ IMU {imu_id} 就緒 ({device.name})")
            
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
        print("=== True Parallel 水波引擎 啟動 ===")
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
import asyncio
import struct
import time
import random
import threading
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
from bleak import BleakClient, BleakScanner

SAMPLERATE = 44100
CHANNELS = 2

last_accel = {}    
last_trigger = {}  
connected_addresses = set()
reserved_ids = set() 
active_ids = set() 

gui_params = {
    1: {'vol': 0.8, 'tail': 0.5},
    2: {'vol': 0.3, 'tail': 0.5}, 
    3: {'vol': 0.8, 'tail': 0.5},
    4: {'vol': 0.8, 'tail': 0.5},
}

SUB_GAIN_BASE = 0.2  

# --- 核心新增：C 大九和弦 (C Major 9) 頻率表 ---
# 包含 C, E, G, B, D，橫跨三個八度，帶來豎琴般的琶音感
MAJOR_9_ARPEGGIO = [
    261.63, # C4 (中央 C)
    329.63, # E4
    392.00, # G4
    493.88, # B4
    587.33, # D5
    659.25, # E5
    783.99, # G5
    987.77, # B5
    1174.66 # D6
]

class FMVoice:
    def __init__(self, voice_id):
        self.id = voice_id
        
        # --- 音色和諧化 ---
        # 為了配合大九和弦的喜氣，將 FM Ratio 改為和諧的整數或簡單分數
        if voice_id == 1:
            self.base_ratio = 2.0  # 亮麗的鐵琴
            self.mod_index = 0.8
        elif voice_id == 2:
            self.base_ratio = 1.0  # 溫潤的木琴
            self.mod_index = 1.2
        elif voice_id == 3:
            self.base_ratio = 4.0  # 喜氣的音樂盒鐘聲
            self.mod_index = 0.6
        else: 
            self.base_ratio = 0.5  # 厚實的電鋼琴
            self.mod_index = 0.4

        self.freq = MAJOR_9_ARPEGGIO[0]
        self.sub_freq = 65.41 # 預設 C2 低頻
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
        self.pan = random.choice([random.uniform(0.1, 0.3), random.uniform(0.7, 0.9)])

        self.max_delay_samples = int(SAMPLERATE * 0.5)
        self.delay_buffer = np.zeros((self.max_delay_samples, 2))
        self.delay_ptr = 0
        self.current_delay_samples = int(SAMPLERATE * 0.3)

    def trigger(self, power):
        tail_val = gui_params[self.id]['tail']
        decay_time_sec = 0.2 + (tail_val * 3.8)
        self.decay_rate = 0.001 ** (1.0 / (decay_time_sec * SAMPLERATE))
        
        self.target_amp = min(0.8, power / 4.0 + 0.15)
        attack_sec = random.uniform(0.03, 0.07)
        self.attack_step = self.target_amp / (SAMPLERATE * attack_sec)
        self.state = 'ATTACK'  
        
        random_sec = random.uniform(0.15, 0.40)
        self.current_delay_samples = int(SAMPLERATE * random_sec)
        # 取消音色的隨機微調，確保和弦純正不走音
        self.ratio = self.base_ratio 
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
        
        raw_sub = np.sin(self.phase_sub + 2 * np.pi * self.sub_freq * t) * env * SUB_GAIN_BASE
        
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
        
        tail_val = gui_params[self.id]['tail']
        feedback_base = 0.1 + (tail_val * 0.6) 

        out_stereo = np.zeros((frames, 2))
        for i in range(frames):
            fm_L = filtered_fm[i] * (1.0 - self.pan)
            fm_R = filtered_fm[i] * self.pan
            
            read_ptr = (self.delay_ptr - self.current_delay_samples) % self.max_delay_samples
            delayed_signal = self.delay_buffer[read_ptr]
            
            mix_L = fm_L * 0.7 + delayed_signal[0] * 0.4
            mix_R = fm_R * 0.7 + delayed_signal[1] * 0.4
            
            out_stereo[i, 0] = mix_L + raw_sub[i]
            out_stereo[i, 1] = mix_R + raw_sub[i]
            
            feedback = feedback_base + random.uniform(-0.02, 0.02)
            self.delay_buffer[self.delay_ptr, 0] = fm_L + delayed_signal[1] * feedback
            self.delay_buffer[self.delay_ptr, 1] = fm_R + delayed_signal[0] * feedback
            self.delay_ptr = (self.delay_ptr + 1) % self.max_delay_samples

        current_vol = gui_params[self.id]['vol']
        return out_stereo * current_vol

poly_voices = {1: FMVoice(1), 2: FMVoice(2), 3: FMVoice(3), 4: FMVoice(4)}

def audio_callback(outdata, frames, time, status):
    mixed_all = np.zeros((frames, 2))
    for v in poly_voices.values():
        mixed_all += v.next_block(frames)
    outdata[:] = mixed_all

def handle_imu_data(imu_id, data):
    if len(data) < 20 or data[0] != 0x55 or data[1] != 0x61: return
    vals = struct.unpack('<hhhhhhhhh', data[2:20])
    ax, ay, az = [v / 32768.0 * 16 for v in vals[0:3]]
    current_g = (ax**2 + ay**2 + az**2)**0.5
    roll = vals[6] / 32768.0 * 180 
    pitch = vals[7] / 32768.0 * 180

    v = poly_voices[imu_id]
    
    # --- 核心修改：和弦鎖定 (Quantization) ---
    # 將 -90 到 90 度的 pitch 角度，轉換為 0 到 8 的陣列索引
    normalized_pitch = (pitch + 90) / 180.0
    note_idx = int(normalized_pitch * len(MAJOR_9_ARPEGGIO))
    note_idx = max(0, min(len(MAJOR_9_ARPEGGIO) - 1, note_idx)) # 確保不超出陣列範圍
    
    # 從大九和弦表中抓取正確的頻率
    v.freq = MAJOR_9_ARPEGGIO[note_idx]
    
    # 讓推動水波的低頻，跟隨和弦的「根音(C)與屬音(G)」，保持在 65Hz ~ 98Hz 之間
    # 這能確保水波的幾何圖形穩定且漂亮
    sub_notes = [65.41, 82.41, 98.00] # C2, E2, G2
    v.sub_freq = sub_notes[note_idx % 3] 
    
    v.cutoff = 2000 + abs(roll) * 45
    
    now = time.time()
    prev_g = last_accel.get(imu_id, 1.0)
    delta_g = abs(current_g - prev_g)
    last_accel[imu_id] = current_g
    
    if (current_g > 1.8 or delta_g > 0.8) and (now - last_trigger.get(imu_id, 0) > 0.15):
        v.trigger(current_g) 
        last_trigger[imu_id] = now
        print(f"🌸 IMU {imu_id} Arp Note! Freq: {v.freq:.1f} Hz")

async def connect_imu(device, imu_id):
    WRITE_CHAR = "0000ffe9-0000-1000-8000-00805f9a34fb"
    NOTIFY_CHAR = "0000ffe4-0000-1000-8000-00805f9a34fb"
    try:
        async with BleakClient(device) as client:
            active_ids.add(imu_id)
            reserved_ids.discard(imu_id)
            
            poly_voices[imu_id].current_amp = 0.0
            poly_voices[imu_id].state = 'IDLE'
            
            print(f"✅ IMU {imu_id} 喜氣琶音就緒")
            await client.start_notify(NOTIFY_CHAR, lambda s, d: handle_imu_data(imu_id, d))
            await client.write_gatt_char(WRITE_CHAR, bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5]))
            while client.is_connected: 
                await asyncio.sleep(1)
    except Exception as e: 
        print(f"❌ IMU {imu_id} 連線中斷: {e}")
    finally: 
        active_ids.discard(imu_id)
        reserved_ids.discard(imu_id)
        connected_addresses.discard(device.address)

async def manager():
    with sd.OutputStream(channels=2, callback=audio_callback, samplerate=SAMPLERATE):
        print("=== 系統運作中，請於彈出視窗調整參數 ===")
        while True:
            used_ids = active_ids.union(reserved_ids)
            if len(used_ids) < 4:
                devices = await BleakScanner.discover(timeout=1.0)
                for d in devices:
                    if d.name and d.name.startswith("WT") and d.address not in connected_addresses:
                        used_ids = active_ids.union(reserved_ids)
                        if len(used_ids) >= 4: break
                        new_id = next(i for i in range(1, 5) if i not in used_ids)
                        connected_addresses.add(d.address)
                        reserved_ids.add(new_id)
                        asyncio.create_task(connect_imu(d, new_id))
                await asyncio.sleep(0.2)
            else:
                await asyncio.sleep(2.0)

def run_audio_engine():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(manager())

vol_sliders = {}
tail_sliders = {}

def mute_all():
    for i in range(1, 5):
        gui_params[i]['vol'] = 0.0
        vol_sliders[i].set(0.0)

def create_gui():
    root = tk.Tk()
    root.title("Ping Pong Ping Pong - Mixer")
    root.geometry("500x420")
    root.configure(padx=20, pady=20)
    
    title = tk.Label(root, text="Sonic Squid - 展演控制台", font=("Helvetica", 16, "bold"))
    title.grid(row=0, column=0, columnspan=4, pady=(0, 20))

    for i in range(1, 5):
        frame = ttk.LabelFrame(root, text=f"IMU {i}")
        frame.grid(row=1, column=i-1, padx=10, sticky="n")
        
        tk.Label(frame, text="音量").pack(pady=(5, 0))
        vol_sliders[i] = tk.Scale(frame, from_=1.0, to=0.0, orient="vertical", length=130, resolution=0.01, showvalue=False)
        vol_sliders[i].set(gui_params[i]['vol'])
        vol_sliders[i].pack(pady=5)
        vol_sliders[i].config(command=lambda val, idx=i: gui_params[idx].update({'vol': float(val)}))
        
        tk.Label(frame, text="尾音長度").pack(pady=(10, 0))
        tail_sliders[i] = tk.Scale(frame, from_=1.0, to=0.0, orient="vertical", length=130, resolution=0.01, showvalue=False)
        tail_sliders[i].set(gui_params[i]['tail'])
        tail_sliders[i].pack(pady=5)
        tail_sliders[i].config(command=lambda val, idx=i: gui_params[idx].update({'tail': float(val)}))

    mute_btn = ttk.Button(root, text="一鍵靜音 (Mute All)", command=mute_all)
    mute_btn.grid(row=2, column=0, columnspan=4, pady=25)

    root.mainloop()

if __name__ == "__main__":
    audio_thread = threading.Thread(target=run_audio_engine, daemon=True)
    audio_thread.start()
    
    try:
        create_gui()
    except KeyboardInterrupt:
        print("\n已停止。")
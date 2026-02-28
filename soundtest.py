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

SAMPLERATE = 48000
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

# 極低的基礎增益，讓 40Hz 變成隱形的物理推手
SUB_GAIN_BASE = 0.12  

class FMVoice:
    def __init__(self, voice_id):
        self.id = voice_id
        
        # 回歸科技感的 FM 比例設定
        if voice_id == 1:
            self.base_ratio = 11.72
            self.mod_index = 0.8
            self.base_freq = 600.0
        elif voice_id == 2:
            self.base_ratio = 3.41
            self.mod_index = 1.2
            self.base_freq = 400.0
        elif voice_id == 3:
            self.base_ratio = 7.13
            self.mod_index = 0.6
            self.base_freq = 550.0
        else: 
            self.base_ratio = 2.0
            self.mod_index = 0.4
            self.base_freq = 350.0

        self.freq = self.base_freq
        self.target_freq = self.base_freq
        
        # 初始低頻設定在 40Hz
        self.sub_freq = 40.0  
        self.target_sub_freq = 40.0
        
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

        self.max_delay_samples = int(SAMPLERATE * 0.6)
        self.delay_buffer = np.zeros((self.max_delay_samples, 2))
        self.delay_ptr = 0
        
        delay_times = {1: 0.3, 2: 0.35, 3: 0.4, 4: 0.25}
        self.current_delay_samples = int(SAMPLERATE * delay_times.get(voice_id, 0.3))

    def trigger(self, power):
        tail_val = gui_params[self.id]['tail']
        decay_time_sec = 0.2 + (tail_val * 3.8)
        self.decay_rate = 0.001 ** (1.0 / (decay_time_sec * SAMPLERATE))
        
        self.target_amp = min(0.8, power / 4.0 + 0.15)
        
        # 為了搭配 40Hz 且避免爆音，Attack 設為 20ms~40ms (約一個 40Hz 的週期)
        attack_sec = random.uniform(0.02, 0.04)
        self.attack_step = self.target_amp / (SAMPLERATE * attack_sec)
        self.state = 'ATTACK'  
        
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

        # 恢復科技感的平滑滑音 (Portamento)
        glide_speed = 0.015 
        
        out_stereo = np.zeros((frames, 2))
        
        for i in range(frames):
            self.freq += (self.target_freq - self.freq) * glide_speed
            self.sub_freq += (self.target_sub_freq - self.sub_freq) * glide_speed
            
            # 高頻 FM 運算
            mod_freq = self.freq * self.ratio
            m_val = np.sin(self.phase_m) * self.mod_index
            raw_fm = np.sin(self.phase_c + m_val) * env[i]
            
            # --- 核心修改：Sub-Pizzicato ---
            # 將 env[i] 平方，創造出極短促、緊實的弱撥弦推力，不影響高頻
            pizzicato_env = env[i] ** 2
            raw_sub = np.sin(self.phase_sub) * pizzicato_env * SUB_GAIN_BASE
            
            alpha = self.cutoff / (self.cutoff + SAMPLERATE / (2 * np.pi))
            self.last_out = self.last_out + alpha * (raw_fm - self.last_out)
            
            self.phase_c = (self.phase_c + 2 * np.pi * self.freq / SAMPLERATE) % (2 * np.pi)
            self.phase_m = (self.phase_m + 2 * np.pi * mod_freq / SAMPLERATE) % (2 * np.pi)
            self.phase_sub = (self.phase_sub + 2 * np.pi * self.sub_freq / SAMPLERATE) % (2 * np.pi)
            
            fm_L = self.last_out * (1.0 - self.pan)
            fm_R = self.last_out * self.pan
            
            read_ptr = (self.delay_ptr - self.current_delay_samples) % self.max_delay_samples
            delayed_signal = self.delay_buffer[read_ptr]
            
            mix_L = fm_L * 0.7 + delayed_signal[0] * 0.4
            mix_R = fm_R * 0.7 + delayed_signal[1] * 0.4
            
            # 結合：高頻 + 隱形極低頻水波驅動
            out_stereo[i, 0] = mix_L + raw_sub
            out_stereo[i, 1] = mix_R + raw_sub
            
            tail_val = gui_params[self.id]['tail']
            feedback = (0.1 + (tail_val * 0.6)) + random.uniform(-0.02, 0.02)
            
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
    
    outdata[:] = np.tanh(mixed_all)

def handle_imu_data(imu_id, data):
    if len(data) < 20 or data[0] != 0x55 or data[1] != 0x61: return
    vals = struct.unpack('<hhhhhhhhh', data[2:20])
    ax, ay, az = [v / 32768.0 * 16 for v in vals[0:3]]
    current_g = (ax**2 + ay**2 + az**2)**0.5
    roll = vals[6] / 32768.0 * 180 
    pitch = vals[7] / 32768.0 * 180

    v = poly_voices[imu_id]
    
    # --- 回歸科技滑音 ---
    # 高頻：無段連續變化，帶有空靈的科技感
    v.target_freq = v.base_freq + (pitch + 90) * 8.0 
    
    # 低頻：將角度映射到 35Hz ~ 45Hz 的窄頻帶 (中央為 40Hz)
    normalized_pitch = max(0.0, min(1.0, (pitch + 90) / 180.0))
    v.target_sub_freq = 35.0 + (normalized_pitch * 10.0)
    
    v.cutoff = 2000 + abs(roll) * 45
    
    now = time.time()
    prev_g = last_accel.get(imu_id, 1.0)
    delta_g = abs(current_g - prev_g)
    last_accel[imu_id] = current_g
    
    if (current_g > 1.8 or delta_g > 0.8) and (now - last_trigger.get(imu_id, 0) > 0.15):
        v.trigger(current_g) 
        last_trigger[imu_id] = now
        print(f"🛸 IMU {imu_id} Sci-Fi Pluck! High:{v.target_freq:.0f}Hz | Sub:{v.target_sub_freq:.1f}Hz")

async def connect_imu(device, imu_id):
    WRITE_CHAR = "0000ffe9-0000-1000-8000-00805f9a34fb"
    NOTIFY_CHAR = "0000ffe4-0000-1000-8000-00805f9a34fb"
    try:
        async with BleakClient(device) as client:
            active_ids.add(imu_id)
            reserved_ids.discard(imu_id)
            
            poly_voices[imu_id].current_amp = 0.0
            poly_voices[imu_id].state = 'IDLE'
            
            print(f"✅ IMU {imu_id} 科技滑音引擎就緒")
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
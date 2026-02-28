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
    2: {'vol': 0.4, 'tail': 0.5}, 
    3: {'vol': 0.8, 'tail': 0.5},
    4: {'vol': 0.8, 'tail': 0.5},
}

# 經過優化後的低頻推力
SUB_GAIN_BASE = 0.4  

class SciFiVoice:
    def __init__(self, voice_id):
        self.id = voice_id
        
        # 給予每顆傳感器不同的基礎音高，確保空間層次感
        base_freqs = {1: 600.0, 2: 400.0, 3: 800.0, 4: 500.0}
        self.base_freq = base_freqs[voice_id]
        
        self.freq = self.base_freq
        self.target_freq = self.base_freq
        
        self.sub_freq = 40.0  
        self.target_sub_freq = 40.0
        
        # 雙振盪器相位 + 低頻相位
        self.phase_1 = 0
        self.phase_2 = 0
        self.phase_sub = 0
        
        # --- 獨立高頻包絡線 ---
        self.state = 'IDLE'
        self.current_amp = 0.0
        self.target_amp = 0.0
        self.attack_step = 0.0
        self.decay_rate = 0.9998
        
        # --- 獨立低頻包絡線 (水波專用) ---
        self.sub_state = 'IDLE'
        self.current_sub_amp = 0.0
        self.sub_target = 0.0
        self.sub_attack_step = 0.0
        self.sub_decay_rate = 0.9998
        
        self.pan = random.choice([random.uniform(0.1, 0.3), random.uniform(0.7, 0.9)])

        self.max_delay_samples = int(SAMPLERATE * 0.6)
        self.delay_buffer = np.zeros((self.max_delay_samples, 2))
        self.delay_ptr = 0
        
        delay_times = {1: 0.3, 2: 0.35, 3: 0.4, 4: 0.25}
        self.current_delay_samples = int(SAMPLERATE * delay_times[voice_id])

    def trigger(self, power):
        # 消除爆音核心：如果在完全靜音狀態下觸發，強制將相位歸零 (Zero-Crossing)
        if self.current_amp < 0.005 and self.current_sub_amp < 0.005:
            self.phase_1 = 0
            self.phase_2 = 0
            self.phase_sub = 0

        # --- 1. 高頻音量設定 (受 GUI 控制) ---
        tail_val = gui_params[self.id]['tail']
        decay_time_sec = 0.2 + (tail_val * 3.8)
        self.decay_rate = 0.001 ** (1.0 / (decay_time_sec * SAMPLERATE))
        
        self.target_amp = min(0.6, power / 4.0 + 0.1)
        attack_sec = random.uniform(0.04, 0.08)
        self.attack_step = self.target_amp / (SAMPLERATE * attack_sec)
        self.state = 'ATTACK'  
        
        # --- 2. 低頻音量設定 (不受 GUI 影響的完美撥弦 Pizzicato) ---
        # 固定的 60ms 溫和推力，300ms 極速收斂，確保水波乾淨漂亮
        self.sub_target = min(1.0, power / 3.0 + 0.2)
        self.sub_attack_step = self.sub_target / (SAMPLERATE * 0.06)
        self.sub_decay_rate = 0.001 ** (1.0 / (0.3 * SAMPLERATE))
        self.sub_state = 'ATTACK'
        
        self.pan = random.choice([random.uniform(0.1, 0.3), random.uniform(0.7, 0.9)])

    def next_block(self, frames):
        env = np.zeros(frames)
        sub_env = np.zeros(frames)
        
        # 雙重包絡線即時運算
        for i in range(frames):
            # 高頻
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
            
            # 低頻
            if self.sub_state == 'ATTACK':
                self.current_sub_amp += self.sub_attack_step
                if self.current_sub_amp >= self.sub_target:
                    self.current_sub_amp = self.sub_target
                    self.sub_state = 'DECAY'
            elif self.sub_state == 'DECAY':
                self.current_sub_amp *= self.sub_decay_rate
                if self.current_sub_amp < 0.0001:
                    self.current_sub_amp = 0.0
                    self.sub_state = 'IDLE'

            env[i] = self.current_amp
            sub_env[i] = self.current_sub_amp

        glide_speed = 0.015 
        out_stereo = np.zeros((frames, 2))
        
        for i in range(frames):
            # 滑音平滑器
            self.freq += (self.target_freq - self.freq) * glide_speed
            self.sub_freq += (self.target_sub_freq - self.sub_freq) * glide_speed
            
            # --- 科技感核心：Detuned Dual Oscillators (失諧雙振盪器) ---
            # 創造極致純淨、寬廣的科幻音色，沒有任何詭異的金屬聲
            freq2 = self.freq * 1.008 # 輕微的音高偏移，產生 Lush 的合唱感
            
            raw_osc1 = np.sin(self.phase_1) * env[i]
            raw_osc2 = np.sin(self.phase_2) * env[i]
            
            # 專注於推動物理水波的獨立低頻
            raw_sub = np.sin(self.phase_sub) * sub_env[i] * SUB_GAIN_BASE
            
            self.phase_1 = (self.phase_1 + 2 * np.pi * self.freq / SAMPLERATE) % (2 * np.pi)
            self.phase_2 = (self.phase_2 + 2 * np.pi * freq2 / SAMPLERATE) % (2 * np.pi)
            self.phase_sub = (self.phase_sub + 2 * np.pi * self.sub_freq / SAMPLERATE) % (2 * np.pi)
            
            # 立體聲寬廣化 (將兩顆振盪器稍微拉開)
            fm_L = (raw_osc1 * 0.7 + raw_osc2 * 0.3) * (1.0 - self.pan)
            fm_R = (raw_osc1 * 0.3 + raw_osc2 * 0.7) * self.pan
            
            # Echo 運算
            read_ptr = (self.delay_ptr - self.current_delay_samples) % self.max_delay_samples
            delayed_signal = self.delay_buffer[read_ptr]
            
            mix_L = fm_L * 0.7 + delayed_signal[0] * 0.4
            mix_R = fm_R * 0.7 + delayed_signal[1] * 0.4
            
            # 將隱形低頻與高頻混合輸出
            out_stereo[i, 0] = mix_L + raw_sub
            out_stereo[i, 1] = mix_R + raw_sub
            
            tail_val = gui_params[self.id]['tail']
            feedback = (0.1 + (tail_val * 0.6)) + random.uniform(-0.02, 0.02)
            
            self.delay_buffer[self.delay_ptr, 0] = fm_L + delayed_signal[1] * feedback
            self.delay_buffer[self.delay_ptr, 1] = fm_R + delayed_signal[0] * feedback
            self.delay_ptr = (self.delay_ptr + 1) % self.max_delay_samples

        current_vol = gui_params[self.id]['vol']
        return out_stereo * current_vol

# 使用新的 SciFiVoice
poly_voices = {1: SciFiVoice(1), 2: SciFiVoice(2), 3: SciFiVoice(3), 4: SciFiVoice(4)}

def audio_callback(outdata, frames, time, status):
    mixed_all = np.zeros((frames, 2))
    for v in poly_voices.values():
        mixed_all += v.next_block(frames)
    
    # 柔和限幅器，確保不管多少聲音疊加都不會數位破音
    outdata[:] = np.tanh(mixed_all)

def handle_imu_data(imu_id, data):
    if len(data) < 20 or data[0] != 0x55 or data[1] != 0x61: return
    vals = struct.unpack('<hhhhhhhhh', data[2:20])
    ax, ay, az = [v / 32768.0 * 16 for v in vals[0:3]]
    current_g = (ax**2 + ay**2 + az**2)**0.5
    roll = vals[6] / 32768.0 * 180 
    pitch = vals[7] / 32768.0 * 180

    v = poly_voices[imu_id]
    
    # 高頻科技連續滑音
    v.target_freq = v.base_freq + (pitch + 90) * 8.0 
    
    # 專注於 35Hz ~ 45Hz 的隱形推動力
    normalized_pitch = max(0.0, min(1.0, (pitch + 90) / 180.0))
    v.target_sub_freq = 35.0 + (normalized_pitch * 10.0)
    
    now = time.time()
    prev_g = last_accel.get(imu_id, 1.0)
    delta_g = abs(current_g - prev_g)
    last_accel[imu_id] = current_g
    
    if (current_g > 1.8 or delta_g > 0.8) and (now - last_trigger.get(imu_id, 0) > 0.15):
        v.trigger(current_g) 
        last_trigger[imu_id] = now
        print(f"🛸 IMU {imu_id} Pure Sci-Fi! High:{v.target_freq:.0f}Hz | Sub:{v.target_sub_freq:.1f}Hz")

async def connect_imu(device, imu_id):
    WRITE_CHAR = "0000ffe9-0000-1000-8000-00805f9a34fb"
    NOTIFY_CHAR = "0000ffe4-0000-1000-8000-00805f9a34fb"
    try:
        async with BleakClient(device) as client:
            active_ids.add(imu_id)
            reserved_ids.discard(imu_id)
            
            poly_voices[imu_id].current_amp = 0.0
            poly_voices[imu_id].state = 'IDLE'
            
            print(f"✅ IMU {imu_id} 純淨科幻引擎就緒")
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
    
    title = tk.Label(root, text="Sonic Squid - 純淨科幻版", font=("Helvetica", 16, "bold"))
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
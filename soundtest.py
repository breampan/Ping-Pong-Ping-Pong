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

# --- 音訊全域設定 ---
SAMPLERATE = 44100
CHANNELS = 2
poly_voices = {} 
last_accel = {}    
last_trigger = {}  

connected_addresses = set()
reserved_ids = set() 

# --- GUI 參數儲存區 ---
# 預設 IMU 2 音量較小，且尾音長度 (tail) 預設縮短以減少混濁感
gui_params = {
    1: {'vol': 0.8, 'tail': 0.3},
    2: {'vol': 0.4, 'tail': 0.3}, 
    3: {'vol': 0.8, 'tail': 0.3},
    4: {'vol': 0.8, 'tail': 0.3},
}

SUB_GAIN_BASE = 0.2  

class FMVoice:
    def __init__(self, voice_id):
        self.id = voice_id
        
        # 音色分配
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
        else: 
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
        self.pan = random.choice([random.uniform(0.1, 0.3), random.uniform(0.7, 0.9)])

        # --- 為每顆傳感器建立獨立的 Ping-Pong Echo 緩衝區 ---
        self.max_delay_samples = int(SAMPLERATE * 0.5)
        self.delay_buffer = np.zeros((self.max_delay_samples, 2))
        self.delay_ptr = 0
        self.current_delay_samples = int(SAMPLERATE * 0.3)

    def trigger(self, power):
        # 讀取 GUI 的尾音長度參數 (0.0 ~ 1.0)
        tail_val = gui_params[self.id]['tail']
        
        # 根據 GUI 參數決定 ADSR 衰減長短 (越小消失越快)
        # 短：0.9995 (約0.5秒) ~ 長：0.9999 (約3秒)
        self.decay_rate = 0.9995 + (tail_val * 0.0004)
        
        self.target_amp = min(0.8, power / 4.0 + 0.15)
        attack_sec = random.uniform(0.03, 0.07)
        self.attack_step = self.target_amp / (SAMPLERATE * attack_sec)
        self.state = 'ATTACK'  
        
        # 隨機延遲間隔 (Ping-pong 速度)
        random_sec = random.uniform(0.15, 0.40)
        self.current_delay_samples = int(SAMPLERATE * random_sec)
        self.ratio = self.base_ratio + random.uniform(-0.05, 0.05)
        self.pan = random.choice([random.uniform(0.1, 0.3), random.uniform(0.7, 0.9)])

    def next_block(self, frames):
        # 1. 產生波形
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
        
        # 2. 空間與獨立延遲運算 (Ping-Pong)
        tail_val = gui_params[self.id]['tail']
        # Echo 的反饋量也由 GUI 控制：0.1(只回聲一次) ~ 0.6(回聲多次)
        feedback_base = 0.1 + (tail_val * 0.5) 

        out_stereo = np.zeros((frames, 2))
        for i in range(frames):
            fm_L = filtered_fm[i] * (1.0 - self.pan)
            fm_R = filtered_fm[i] * self.pan
            
            read_ptr = (self.delay_ptr - self.current_delay_samples) % self.max_delay_samples
            delayed_signal = self.delay_buffer[read_ptr]
            
            # 乾濕混音
            mix_L = fm_L * 0.7 + delayed_signal[0] * 0.4
            mix_R = fm_R * 0.7 + delayed_signal[1] * 0.4
            
            # 將 FM 與 Sub-bass 結合
            out_stereo[i, 0] = mix_L + raw_sub[i]
            out_stereo[i, 1] = mix_R + raw_sub[i]
            
            # Ping-Pong 交叉回授
            feedback = feedback_base + random.uniform(-0.02, 0.02)
            self.delay_buffer[self.delay_ptr, 0] = fm_L + delayed_signal[1] * feedback
            self.delay_buffer[self.delay_ptr, 1] = fm_R + delayed_signal[0] * feedback
            self.delay_ptr = (self.delay_ptr + 1) % self.max_delay_samples

        # 3. 套用 GUI 的總音量
        current_vol = gui_params[self.id]['vol']
        return out_stereo * current_vol

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

async def connect_imu(device, imu_id):
    WRITE_CHAR = "0000ffe9-0000-1000-8000-00805f9a34fb"
    NOTIFY_CHAR = "0000ffe4-0000-1000-8000-00805f9a34fb"
    try:
        async with BleakClient(device) as client:
            poly_voices[imu_id] = FMVoice(imu_id)
            reserved_ids.discard(imu_id)
            print(f"✅ IMU {imu_id} 就緒")
            
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

async def manager():
    with sd.OutputStream(channels=2, callback=audio_callback, samplerate=SAMPLERATE):
        print("=== 系統運作中，請於彈出視窗調整參數 ===")
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

def run_audio_engine():
    """在背景執行音訊與藍牙連線"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(manager())

# --- GUI 介面建置 ---
def create_gui():
    root = tk.Tk()
    root.title("Ping Pong Ping Pong - Mixer")
    root.geometry("500x350")
    root.configure(padx=20, pady=20)
    
    title = tk.Label(root, text="Sonic Squid - 展演控制台", font=("Helvetica", 16, "bold"))
    title.grid(row=0, column=0, columnspan=4, pady=(0, 20))

    # 建立四個 IMU 的推桿
    for i in range(1, 5):
        frame = ttk.LabelFrame(root, text=f"IMU {i}")
        frame.grid(row=1, column=i-1, padx=10, sticky="n")
        
        # 音量推桿 (Volume)
        tk.Label(frame, text="音量").pack(pady=(5, 0))
        vol_slider = ttk.Scale(frame, from_=1.0, to=0.0, orient="vertical", length=120)
        vol_slider.set(gui_params[i]['vol'])
        vol_slider.pack(pady=5)
        # 綁定即時更新
        vol_slider.config(command=lambda val, idx=i: gui_params[idx].update({'vol': float(val)}))
        
        # 尾音長度推桿 (Tail / Echo Length)
        tk.Label(frame, text="尾音長度").pack(pady=(10, 0))
        tail_slider = ttk.Scale(frame, from_=1.0, to=0.0, orient="vertical", length=80)
        tail_slider.set(gui_params[i]['tail'])
        tail_slider.pack(pady=5)
        tail_slider.config(command=lambda val, idx=i: gui_params[idx].update({'tail': float(val)}))

    root.mainloop()

if __name__ == "__main__":
    # 啟動音訊與藍牙背景執行緒
    audio_thread = threading.Thread(target=run_audio_engine, daemon=True)
    audio_thread.start()
    
    # 啟動主執行緒的 GUI 介面
    try:
        create_gui()
    except KeyboardInterrupt:
        print("\n已停止。")
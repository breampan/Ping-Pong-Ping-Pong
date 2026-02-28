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
    1: {'vol': 0.8, 'tail': 0.5, 'sens': 5.0},
    2: {'vol': 0.4, 'tail': 0.5, 'sens': 5.0}, 
    3: {'vol': 0.8, 'tail': 0.5, 'sens': 5.0},
    4: {'vol': 0.8, 'tail': 0.5, 'sens': 5.0},
}

SUB_GAIN_BASE = 0.65  

class SciFiVoice:
    def __init__(self, voice_id):
        self.id = voice_id
        base_freqs = {1: 600.0, 2: 400.0, 3: 800.0, 4: 500.0}
        self.base_freq = base_freqs[voice_id]
        self.freq = self.base_freq
        self.target_freq = self.base_freq
        self.sub_freq = 40.0  
        self.target_sub_freq = 40.0
        
        self.phase_1 = 0; self.phase_2 = 0; self.phase_sub = 0
        self.state = 'IDLE'; self.current_amp = 0.0; self.target_amp = 0.0
        self.attack_step = 0.0; self.decay_rate = 0.9998
        self.sub_state = 'IDLE'; self.current_sub_amp = 0.0; self.sub_target = 0.0
        self.sub_attack_step = 0.0; self.sub_decay_rate = 0.9998
        
        self.pan = random.choice([0.0, 1.0])
        self.max_delay_samples = int(SAMPLERATE * 0.6)
        self.delay_buffer = np.zeros((self.max_delay_samples, 2))
        self.delay_ptr = 0
        delay_times = {1: 0.3, 2: 0.35, 3: 0.4, 4: 0.25}
        self.current_delay_samples = int(SAMPLERATE * delay_times[voice_id])

    def trigger(self, power, threshold):
        if self.current_amp < 0.005 and self.current_sub_amp < 0.005:
            self.phase_1 = 0; self.phase_2 = 0; self.phase_sub = 0

        # 計算拍擊力度 (Overshoot of delta_g)
        overshoot = power - threshold
        intensity = np.clip(overshoot / 2.0 + 0.15, 0.1, 1.0) 

        tail_val = gui_params[self.id]['tail']
        decay_time_sec = 0.2 + (tail_val * 3.8)
        self.decay_rate = 0.001 ** (1.0 / (decay_time_sec * SAMPLERATE))
        
        self.target_amp = intensity * 0.6  
        attack_sec = 0.05 - (intensity * 0.04)
        self.attack_step = self.target_amp / (SAMPLERATE * attack_sec)
        self.state = 'ATTACK'  
        
        self.sub_target = intensity * 1.0  
        sub_attack_sec = 0.08 - (intensity * 0.05) 
        self.sub_attack_step = self.sub_target / (SAMPLERATE * sub_attack_sec)
        self.sub_decay_rate = 0.001 ** (1.0 / (0.3 * SAMPLERATE))
        self.sub_state = 'ATTACK'
        
        self.pan = random.choice([0.0, 1.0])

    def next_block(self, frames):
        env = np.zeros(frames); sub_env = np.zeros(frames)
        for i in range(frames):
            if self.state == 'ATTACK':
                self.current_amp += self.attack_step
                if self.current_amp >= self.target_amp:
                    self.current_amp = self.target_amp; self.state = 'DECAY'
            elif self.state == 'DECAY':
                self.current_amp *= self.decay_rate
                if self.current_amp < 0.0001:
                    self.current_amp = 0.0; self.state = 'IDLE'
            
            if self.sub_state == 'ATTACK':
                self.current_sub_amp += self.sub_attack_step
                if self.current_sub_amp >= self.sub_target:
                    self.current_sub_amp = self.sub_target; self.sub_state = 'DECAY'
            elif self.sub_state == 'DECAY':
                self.current_sub_amp *= self.sub_decay_rate
                if self.current_sub_amp < 0.0001:
                    self.current_sub_amp = 0.0; self.sub_state = 'IDLE'

            env[i] = self.current_amp; sub_env[i] = self.current_sub_amp

        glide_speed = 0.015 
        out_stereo = np.zeros((frames, 2))
        
        for i in range(frames):
            self.freq += (self.target_freq - self.freq) * glide_speed
            self.sub_freq += (self.target_sub_freq - self.sub_freq) * glide_speed
            
            freq2 = self.freq * 1.008 
            raw_osc1 = np.sin(self.phase_1) * env[i]
            raw_osc2 = np.sin(self.phase_2) * env[i]
            raw_sub = np.sin(self.phase_sub) * sub_env[i] * SUB_GAIN_BASE
            
            self.phase_1 = (self.phase_1 + 2 * np.pi * self.freq / SAMPLERATE) % (2 * np.pi)
            self.phase_2 = (self.phase_2 + 2 * np.pi * freq2 / SAMPLERATE) % (2 * np.pi)
            self.phase_sub = (self.phase_sub + 2 * np.pi * self.sub_freq / SAMPLERATE) % (2 * np.pi)
            
            fm_L = (raw_osc1 * 0.7 + raw_osc2 * 0.3) * (1.0 - self.pan)
            fm_R = (raw_osc1 * 0.3 + raw_osc2 * 0.7) * self.pan
            sub_L = raw_sub * (1.0 - self.pan)
            sub_R = raw_sub * self.pan
            
            read_ptr = (self.delay_ptr - self.current_delay_samples) % self.max_delay_samples
            delayed_signal = self.delay_buffer[read_ptr]
            
            out_stereo[i, 0] = (fm_L * 0.7 + sub_L) + delayed_signal[0]
            out_stereo[i, 1] = (fm_R * 0.7 + sub_R) + delayed_signal[1]
            
            tail_val = gui_params[self.id]['tail']
            feedback = 0.15 + (tail_val * 0.6) 
            self.delay_buffer[self.delay_ptr, 0] = (fm_L * 0.7 + sub_L) + delayed_signal[1] * feedback
            self.delay_buffer[self.delay_ptr, 1] = (fm_R * 0.7 + sub_R) + delayed_signal[0] * feedback
            self.delay_ptr = (self.delay_ptr + 1) % self.max_delay_samples

        return out_stereo * gui_params[self.id]['vol']

poly_voices = {1: SciFiVoice(1), 2: SciFiVoice(2), 3: SciFiVoice(3), 4: SciFiVoice(4)}

def audio_callback(outdata, frames, time, status):
    mixed_all = np.zeros((frames, 2))
    for v in poly_voices.values(): mixed_all += v.next_block(frames)
    outdata[:] = np.tanh(mixed_all)

def handle_imu_data(imu_id, data):
    if len(data) < 20 or data[0] != 0x55 or data[1] != 0x61: return
    vals = struct.unpack('<hhhhhhhhh', data[2:20])
    ax, ay, az = [v / 32768.0 * 16 for v in vals[0:3]]
    current_g = (ax**2 + ay**2 + az**2)**0.5
    pitch = vals[7] / 32768.0 * 180

    v = poly_voices[imu_id]
    v.target_freq = v.base_freq + (pitch + 90) * 8.0 
    normalized_pitch = max(0.0, min(1.0, (pitch + 90) / 180.0))
    v.target_sub_freq = 35.0 + (normalized_pitch * 10.0)
    
    now = time.time()
    prev_g = last_accel.get(imu_id, 1.0)
    
    # 計算瞬間震波 (Impulse Jerk)
    delta_g = abs(current_g - prev_g)
    
    impulse_threshold = 2.0 - (gui_params[imu_id]['sens'] * 0.17)
    
    # 判斷瞬間變化率 (Slap)
    if delta_g > impulse_threshold and (now - last_trigger.get(imu_id, 0) > 0.25):
        # 將 delta_g 作為力度傳給合成器
        v.trigger(delta_g, impulse_threshold) 
        last_trigger[imu_id] = now
        print(f"💥 水球 {imu_id} 拍擊! 震波: {delta_g:.1f}g (門檻: {impulse_threshold:.1f}g)")
        
    last_accel[imu_id] = current_g

async def connect_imu(device, imu_id):
    WRITE_CHAR = "0000ffe9-0000-1000-8000-00805f9a34fb"
    NOTIFY_CHAR = "0000ffe4-0000-1000-8000-00805f9a34fb"
    try:
        async with BleakClient(device) as client:
            active_ids.add(imu_id); reserved_ids.discard(imu_id)
            poly_voices[imu_id].current_amp = 0.0; poly_voices[imu_id].state = 'IDLE'
            print(f"✅ IMU {imu_id} 水球拍擊引擎就緒")
            await client.start_notify(NOTIFY_CHAR, lambda s, d: handle_imu_data(imu_id, d))
            await client.write_gatt_char(WRITE_CHAR, bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5]))
            while client.is_connected: await asyncio.sleep(1)
    except Exception as e: print(f"❌ IMU {imu_id} 斷線")
    finally: 
        active_ids.discard(imu_id); reserved_ids.discard(imu_id); connected_addresses.discard(device.address)

async def manager():
    with sd.OutputStream(channels=2, callback=audio_callback, samplerate=SAMPLERATE):
        print("=== Phase 2: 水球拍擊版 (Impulse) ===")
        while True:
            used_ids = active_ids.union(reserved_ids)
            if len(used_ids) < 4:
                devices = await BleakScanner.discover(timeout=1.0)
                for d in devices:
                    if d.name and d.name.startswith("WT") and d.address not in connected_addresses:
                        used_ids = active_ids.union(reserved_ids)
                        if len(used_ids) >= 4: break
                        new_id = next(i for i in range(1, 5) if i not in used_ids)
                        connected_addresses.add(d.address); reserved_ids.add(new_id)
                        asyncio.create_task(connect_imu(d, new_id))
                await asyncio.sleep(0.2)
            else: await asyncio.sleep(2.0)

def run_audio_engine():
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop); loop.run_until_complete(manager())

def create_gui():
    root = tk.Tk(); root.title("Ping Pong Ping Pong - Mixer (SLAP)"); root.geometry("600x500"); root.configure(padx=20, pady=20)
    tk.Label(root, text="第二階段：水球拍擊版", font=("Helvetica", 16, "bold")).grid(row=0, column=0, columnspan=4, pady=(0, 20))
    for i in range(1, 5):
        frame = ttk.LabelFrame(root, text=f"IMU {i}"); frame.grid(row=1, column=i-1, padx=10, sticky="n")
        
        tk.Label(frame, text="總音量").pack(pady=(5, 0))
        vol = tk.Scale(frame, from_=1.0, to=0.0, orient="vertical", length=100, resolution=0.01, showvalue=False); vol.set(gui_params[i]['vol']); vol.pack(pady=5)
        vol.config(command=lambda val, idx=i: gui_params[idx].update({'vol': float(val)}))
        
        tk.Label(frame, text="尾音長度").pack(pady=(10, 0))
        tail = tk.Scale(frame, from_=1.0, to=0.0, orient="vertical", length=100, resolution=0.01, showvalue=False); tail.set(gui_params[i]['tail']); tail.pack(pady=5)
        tail.config(command=lambda val, idx=i: gui_params[idx].update({'tail': float(val)}))
        
        tk.Label(frame, text="拍擊靈敏").pack(pady=(10, 0))
        sens = tk.Scale(frame, from_=10.0, to=1.0, orient="vertical", length=100, resolution=0.1, showvalue=False); sens.set(gui_params[i]['sens']); sens.pack(pady=5)
        sens.config(command=lambda val, idx=i: gui_params[idx].update({'sens': float(val)}))
        tk.Label(frame, text="(上:輕摸 / 下:重拍)", font=("Helvetica", 10), fg="gray").pack(pady=(0, 5))
    root.mainloop()

if __name__ == "__main__":
    threading.Thread(target=run_audio_engine, daemon=True).start()
    try: create_gui()
    except KeyboardInterrupt: print("\n已停止。")
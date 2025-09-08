import pyaudio
import numpy as np
import webrtcvad
import wave
import time

# 设置PyAudio相关参数
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1  # 单声道
RATE = 16000  # 采样率
CHUNK = 3200  # 每个块的大小
SILENCE_DURATION = 2  # 假定的静默时间（秒）

# 初始化PyAudio
p = pyaudio.PyAudio()

# 打开音频流
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# 创建VAD对象
vad = webrtcvad.Vad(3)  # 3 是最严格的模式，适用于大多数情况

print("正在监听...")


def detect_speech_vad(data):
    """使用VAD检测语音活动"""
    # VAD需要16位音频数据，每个块为160个采样点 (10ms)
    audio_data = np.frombuffer(data, dtype=np.int16)
    return vad.is_speech(audio_data.tobytes(), RATE)


def save_audio(frames):
    """保存录音文件"""
    if frames:
        with wave.open('../user_audio.wav', 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        print("录音保存成功！")


def record_audio():
    """录制音频"""
    frames = []
    silence_counter = 0  # 静默计数器
    while True:
        data = stream.read(CHUNK)
        if detect_speech_vad(data):
            print("检测到语音...")
            silence_counter = 0
            frames.append(data)  # 继续录音
        else:
            silence_counter += 1
            if silence_counter > RATE / CHUNK * SILENCE_DURATION:
                print("检测到静默，停止录音")
                break  # 超过静默时间后停止录音
    print(frames)
    if frames:
        save_audio(frames)


# 主逻辑：根据语音活动开始和停止录音
while True:
    record_audio()
    time.sleep(1)  # 每次完成录音后稍作等待

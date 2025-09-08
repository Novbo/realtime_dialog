import pyaudio
import wave

# 设置音频参数
CHUNK = 1024  # 数据块大小
FORMAT = pyaudio.paInt16  # 采样格式（16位整型）
CHANNELS = 1  # 单声道
RATE = 44100  # 采样率（每秒采样次数）
RECORD_SECONDS = 5  # 录制时长（秒）
WAVE_OUTPUT_FILENAME = "../output.wav"  # 输出文件名

# 创建PyAudio对象
p = pyaudio.PyAudio()

# 打开音频流
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,  # 启用输入（录音）
                frames_per_buffer=CHUNK)

print("开始录音...")

frames = []  # 用于存储音频数据块

# 循环读取音频数据
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("录音结束.")

# 停止和关闭流
stream.stop_stream()
stream.close()
p.terminate()  # 终止PyAudio对象

# 保存录音为WAV文件
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))  # 将所有数据块连接并写入
wf.close()

print(f"音频已保存为: {WAVE_OUTPUT_FILENAME}")

import os
import queue
import threading

import torch

import ChatTTS
import pyaudio
import time
import wave
import numpy as np
import pyttsx3
import requests
import sounddevice as sd
import config

from dataclasses import dataclass
from typing import Optional
from loguru import logger

# 初始化录音数据
audio_datas = []
num = 0
start_record = False
end_record = False
zero_num = -10000


class ChatTts:
    """文字转语音模型"""

    def start(self):
        self.chat = ChatTTS.Chat()
        # 初始化模型
        self.chat.load(compile=False)  # Set to True for better performance
        spk = config.SPK
        speaker_spk = torch.tensor([float(x) for x in spk.split(',')])

        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=speaker_spk,  # add sampled speaker
            temperature=.3,  # using custom temperature
            top_P=0.7,  # top P decode
            top_K=20,  # top K decode
        )

        ###################################
        # For sentence level manual control.

        # use oral_(0-9), laugh_(0-2), break_(0-7)
        # to generate special token in text to synthesize.
        self.params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_2][laugh_0][break_6]',
        )


@dataclass
class AudioConfig:
    """音频配置数据类"""
    format: str
    bit_size: int
    channels: int
    sample_rate: int
    chunk: int


class AudioDeviceManager:
    """音频设备管理类，处理音频输入输出"""

    def __init__(self, input_config: AudioConfig, output_config: AudioConfig):
        self.input_config = input_config
        self.output_config = output_config
        self.pyaudio = pyaudio.PyAudio()
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None

    def open_input_stream(self) -> pyaudio.Stream:
        """打开音频输入流"""
        # p = pyaudio.PyAudio()
        self.input_stream = self.pyaudio.open(
            format=self.input_config.bit_size,
            channels=self.input_config.channels,
            rate=self.input_config.sample_rate,
            input=True,
            frames_per_buffer=self.input_config.chunk
        )
        return self.input_stream

    def open_output_stream(self) -> pyaudio.Stream:
        """打开音频输出流"""
        self.output_stream = self.pyaudio.open(
            format=self.output_config.bit_size,
            channels=self.output_config.channels,
            rate=self.output_config.sample_rate,
            output=True,
            frames_per_buffer=self.output_config.chunk
        )
        return self.output_stream

    def cleanup(self) -> None:
        """清理音频设备资源"""
        for stream in [self.input_stream, self.output_stream]:
            if stream:
                stream.stop_stream()
                stream.close()
        self.pyaudio.terminate()


class DialogSession:
    """对话会话管理类"""
    is_audio_file_input: bool

    def __init__(self, output_audio_format: str = "pcm", audio_file_path: str = ""):
        self.audio_file_path = audio_file_path
        self.is_audio_file_input = self.audio_file_path != ""

        if output_audio_format == "pcm_s16le":
            config.output_audio_config["format"] = "pcm_s16le"
            config.output_audio_config["bit_size"] = pyaudio.paInt16

        self.is_running = True
        self.is_session_finished = False
        self.is_user_querying = False
        self.is_sending_chat_tts_text = False
        self.audio_buffer = b''

        self.audio_queue = queue.Queue()
        if not self.is_audio_file_input:
            self.audio_device = AudioDeviceManager(
                AudioConfig(**config.input_audio_config),
                AudioConfig(**config.output_audio_config)
            )
            # 初始化音频队列和输出流
            self.output_stream = self.audio_device.open_output_stream()
            # 启动播放线程
            self.is_recording = True
            self.is_playing = True
            self.player_thread = threading.Thread(target=self._audio_player_thread)
            self.player_thread.daemon = True
            self.player_thread.start()

    def _audio_player_thread(self):
        """音频播放线程"""
        count = 0
        while self.is_playing:
            try:
                # 从队列获取音频数据
                audio_data = self.audio_queue.get(timeout=1.0)
                if audio_data is not None:
                    self.output_stream.write(audio_data)
            except queue.Empty:
                # 队列为空时等待一小段时间
                time.sleep(0.1)
            except Exception as e:
                print(f"音频播放错误: {e}")
                time.sleep(0.1)
            finally:
                count += 1

    # def start(self):
    #     """启动"""
    #     # 初始化模型
    #     self.chat_tts = ChatTts()
    #     self.chat_tts.start()
    #     print(1111, id(self.chat_tts))
    #     wavs = self.chat_tts.chat.infer(self.say_hello)
    #     audio_data = wavs[0].astype(np.float32).tobytes()
    #     self.audio_queue.put(audio_data)


def catch_voice(voice_path):
    def save_audio():
        global audio_datas
        # 保存为WAV文件
        if audio_datas:
            audio_np = np.concatenate(audio_datas)

            # 转换为16位PCM格式（WAV标准）
            audio_np = (audio_np * 32767).astype(np.int16)
            with wave.open(voice_path, 'wb') as wf:
                wf.setnchannels(config.CHANNELS)
                wf.setsampwidth(2)  # 16bit=2bytes
                wf.setframerate(config.SAMPLE_RATE)
                wf.writeframes(audio_np.tobytes())

            return voice_path

    def audio_callback(indata, frames, _time, status):
        """录音回调函数"""
        global start_record, audio_datas, end_record, zero_num
        audio_chunk = indata.copy()
        current_volume = np.sqrt(np.mean(audio_chunk ** 2)) * 1000  # 放大便于阈值比较
        threshold_value = 20  # 阈值，声音超过阈值则开始录制
        # print(current_volume)
        if current_volume > threshold_value and start_record is False:
            print("开始录制...")
            start_record = True
            zero_num = 0
            config.share_data['flag'] = False

        if start_record:
            audio_datas.append(indata.copy())

        if current_volume < threshold_value:
            zero_num += 1
            time.sleep(0.03)

        if zero_num > 100:
            print("录制结束...")
            start_record = False
            if len(audio_datas) > 120:
                logger.info(f"本次录制了{len(audio_datas)}个片段")
                audio_datas = audio_datas[0:-100]
                save_audio()
                end_record = True
                config.share_data['flag'] = True
            audio_datas = []
            zero_num = -10000

    print("开始录音... (按Ctrl+C停止)")

    # 启动录音流
    with sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype=config.DTYPE,
            callback=audio_callback
    ):
        while True:
            sd.sleep(1000)  # 保持录音状态


def say_text(text):
    logger.info(f"将开始朗读：{text}")

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    # 选择特定的语音（例如，选择第2个发音人）
    engine.setProperty('voice', voices[0].id)  # 索引可能因系统而异
    engine.setProperty('rate', 200)  # 语速 (默认200)
    engine.setProperty('volume', 0.9)  # 音量 (0-1)

    engine.say(text)
    engine.runAndWait()


def voice_to_text(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise Exception(f"Error: File {file_path} not found")

    # 读取音频文件
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'audio/wav')}

        try:
            # 发送POST请求
            response = requests.post(
                'http://192.168.30.100:5001/upload',
                files=files,
                timeout=30  # 超时设置
            )

            if response.status_code == 200:
                return response.json().get("text")
            else:
                raise Exception(f"Error {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")

# catch_voice()

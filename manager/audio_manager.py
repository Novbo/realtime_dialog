import asyncio
import json
import os
import queue
import signal
import threading
import time
import wave
from dataclasses import dataclass
from typing import Optional, Dict, Any

import aiohttp
import numpy as np
import pyaudio
import requests
import webrtcvad
from loguru import logger

from manager import config
from manager.realtime_dialog_client import ChatTts, FasterWhisper


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
        # 创建VAD对象
        self.vad = webrtcvad.Vad(3)  # 3 是最严格的模式，适用于大多数情况

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

    def detect_speech_vad(self, data):
        """使用VAD检测语音活动"""
        # VAD需要16位音频数据，每个块为160个采样点 (10ms)
        audio_data = np.frombuffer(data, dtype=np.int16)
        return self.vad.is_speech(audio_data.tobytes(), self.input_config.sample_rate)

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
        # 初始化模型
        self.chat_tts = ChatTts()
        self.faster_whisper = FasterWhisper()

        self.audio_file_path = audio_file_path
        self.is_audio_file_input = False  # self.audio_file_path != ""
        if self.is_audio_file_input:
            self.quit_event = asyncio.Event()

        if output_audio_format == "pcm_s16le":
            config.output_audio_config["format"] = "pcm_s16le"
            config.output_audio_config["bit_size"] = pyaudio.paInt16

        self.is_running = True
        self.is_session_finished = False
        self.is_user_querying = False
        self.is_sending_chat_tts_text = False
        self.audio_buffer = b''

        signal.signal(signal.SIGINT, self._keyboard_signal)
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

    def _keyboard_signal(self, sig, frame):
        print(f"receive keyboard Ctrl+C")
        self.is_recording = False
        self.is_playing = False
        self.is_running = False

    async def large_model_server_response(self, input_text):
        logger.info(f"用户提问：{input_text}")

        # 追加用户消息到历史
        config.messages.append({"role": "user", "content": input_text})

        # 流式请求
        url = config.server_config['large_model_server']
        data = {
            "model": "Qwen3:14B",
            "messages": config.messages,
            "stream": True  # 启用流式
        }

        # 记录所有的回答
        full_reply = ""
        flag = False
        with requests.post(url, json=data, stream=True) as response:
            queue_content = ""

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    content = chunk.get("message", {}).get("content", "").replace("\n", "").replace("\r", "")
                    if not content:
                        continue

                    if "</think>" not in content and not flag:
                        continue
                    else:
                        flag = True
                        if "</think>" in content:
                            continue

                    for symbol in config.SYMBOLS:
                        if symbol in content:
                            print(queue_content)
                            audio_data = self.chat_tts.infer(queue_content)
                            self.audio_queue.put(audio_data)
                            queue_content = ""
                            break
                    else:
                        queue_content += content

                    # 记录所有内容
                    full_reply += content
                    # await asyncio.sleep(0.001)

            # 将完整回复追加到历史
            config.messages.append({"role": "assistant", "content": full_reply})
            self.is_session_finished = True

    async def server_response(self) -> None:
        file_path = config.INPUT_WAV_PATH
        while True:
            # 检查文件是否存在
            while not os.path.exists(file_path):
                await asyncio.sleep(0.01)

            # 用户提问前，先清除历史语音
            while not self.audio_queue.empty():
                self.audio_queue.get()
                await asyncio.sleep(0.001)

            # 语音转文字
            input_text = self.faster_whisper.infer(file_path)

            # 删除输入音
            os.remove(config.INPUT_WAV_PATH)

            if not input_text:
                audio_data = self.chat_tts.infer("声音识别错误，麻烦您再讲一次")
                self.audio_queue.put(audio_data)
                continue

            # 等待大模型回答
            await self.large_model_server_response(input_text)
            await asyncio.sleep(0.01)

    async def say_hello(self):
        say_hello = config.SAY_HELLO
        audio_data = self.chat_tts.infer(say_hello)
        self.audio_queue.put(audio_data)

    async def record_audio(self, stream):
        """录制音频"""
        frames = []
        silence_counter = 0  # 静默计数器

        while True:
            data = stream.read(config.input_audio_config['chunk'])
            if self.audio_device.detect_speech_vad(data):
                print("检测到语音...")
                silence_counter = 0
                frames.append(data)  # 继续录音
            else:
                silence_counter += 1
                if silence_counter > config.input_audio_config['sample_rate'] / config.input_audio_config[
                    'chunk'] * config.SILENCE_DURATION:
                    # print("讲话完毕")
                    break  # 超过静默时间后停止录音
        if frames:
            save_audio_data_to_wav(frames, config.INPUT_WAV_PATH)

    async def process_microphone_input(self) -> None:
        await self.say_hello()
        """处理麦克风输入"""
        print("已打开麦克风，请讲话...")

        stream = self.audio_device.open_input_stream()
        while self.is_recording:
            try:
                await self.record_audio(stream)
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"读取麦克风数据出错: {e}")
                await asyncio.sleep(0.1)  # 给系统一些恢复时间

    async def start(self) -> None:
        """启动对话会话"""
        try:
            if self.is_audio_file_input:
                # asyncio.create_task(self.process_audio_file())
                # await self.receive_loop()
                self.quit_event.set()
                await asyncio.sleep(0.1)
            else:
                asyncio.create_task(self.server_response())
                asyncio.create_task(self.process_microphone_input())
                while self.is_running:
                    await asyncio.sleep(0.1)

            while not self.is_session_finished:
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.1)
            save_output_to_file(self.audio_buffer, "output.pcm")
        except Exception as e:
            print(f"会话错误: {e}")
        finally:
            if not self.is_audio_file_input:
                self.audio_device.cleanup()


def save_input_pcm_to_wav(pcm_data: bytes, filename: str) -> None:
    """保存PCM数据为WAV文件"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(config.input_audio_config["channels"])
        wf.setsampwidth(2)  # paInt16 = 2 bytes
        wf.setframerate(config.input_audio_config["sample_rate"])
        wf.writeframes(pcm_data)


def save_output_to_file(audio_data: bytes, filename: str) -> None:
    """保存原始PCM音频数据到文件"""
    if not audio_data:
        print("No audio data to save.")
        return
    try:
        with open(filename, 'wb') as f:
            f.write(audio_data)
    except IOError as e:
        print(f"Failed to save pcm file: {e}")


def save_audio_data_to_wav(frames: list, filename: str):
    """保存列表形式的audio_data为wav格式"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(config.input_audio_config["channels"])
        wf.setsampwidth(2)  # paInt16 = 2 bytes
        wf.setframerate(config.input_audio_config["sample_rate"])
        wf.writeframes(b''.join(frames))
    print("录音保存成功！")
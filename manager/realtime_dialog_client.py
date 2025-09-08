import time

import numpy as np
import torch
from faster_whisper import WhisperModel

import ChatTTS
from manager import config


class ChatTts:
    """文本转语音模型"""

    def __init__(self):
        """初始化"""
        self.chat = ChatTTS.Chat()
        # 初始化模型
        self.chat.load(compile=False, source="custom",
                       custom_path=config.PROJECT_DIR)  # Set to True for better performance
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

    def infer(self, context):
        """文本转语音"""
        wavs = self.chat.infer(text=context, skip_refine_text=True, params_infer_code=self.params_infer_code,
                               params_refine_text=self.params_refine_text)
        try:
            audio_data = wavs[0].astype(np.float32).tobytes()
        except Exception as e:
            return b'\x00\x00' * config.input_audio_config['channels'] * int(
                1000 * config.input_audio_config['sample_rate'] / 1000)
        return audio_data


class FasterWhisper:
    """语音转文本模型"""

    def __init__(self):
        """初始化"""
        model_path = config.FASTER_WHISPER_PATH
        self.model = None
        if self.model is None:
            self.model = WhisperModel(
                model_path,  # 直接指向快照目录
                device="cuda",
                compute_type="int8"
            )
            self.infer(config.FIRST_WAV_PATH)

    def infer(self, wav_path):
        """语音转文子"""
        segments, _ = self.model.transcribe(wav_path)  # 替换为你的中文音频文件
        text = "".join([seg.text for seg in segments])
        return text

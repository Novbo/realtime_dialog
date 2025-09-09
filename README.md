# 语音实时交互

## 1. 介绍

通过3个开源模型 + pyaduio模块实现语音实时交互“类豆包”功能。3个模型为：

- Faster Whisper语音转文字模型
- Qween3:14B通义千问大模型
- ChatTTS文字转语音模型

## 2.环境配置

### 1. 安装Faster Whisper模型所需要的权重文件

链接：https://huggingface.co/Systran/faster-whisper-large-v3/tree/main

### 2. 通过ollama本地部署Qween大模型

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama -v

# 拉取代码
ollama pull qwen2.5:14b

# 本地运行测试
ollama run qwen2.5:14b
```

### 3.安装配置环境

```bash
git clone https://github.com/Novbo/realtime_dialog
cd realtime_dialog
```

安装python

```bash
conda create -n realtime_dialog python=3.11
conda activate realtime_dialog
pip install -r requirements.txt
```

## 3. 运行

注意：运行之前请先修改配置文件信息`config.py`

```bash
python main.py
```

## 4. 代码配置文件

> manager/config.py
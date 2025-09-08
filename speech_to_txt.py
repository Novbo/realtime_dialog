import os
import time
from datetime import datetime

# model = WhisperModel("large-v2", device="cuda", compute_type="int8")  # 选择模型大小
# model = WhisperModel("large", device="cpu", compute_type="int8")  # 选择模型大小
from faster_whisper import WhisperModel
from flask import Flask, request, jsonify, abort
from werkzeug.utils import secure_filename

from manager import config

model_path = "/home/rm/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
model = WhisperModel(
    model_path,  # 直接指向快照目录
    device="cuda",
    compute_type="int8"
)

print(model)

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB限制

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    # 检查是否包含文件
    if 'file' not in request.files:
        abort(400, description="No file part")

    file = request.files['file']

    # 检查是否选择了文件
    if file.filename == '':
        abort(400, description="No selected file")

    # 验证文件类型
    if not allowed_file(file.filename):
        abort(400, description="Only WAV files are allowed")

    # 安全保存文件
    if file:
        filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # 这里可以添加音频处理逻辑
        start_time = time.time()
        segments, _ = model.transcribe(save_path)  # 替换为你的中文音频文件
        text = "".join([seg.text for seg in segments])
        print(f"use time is {time.time() - start_time}s")
        print(text)  # 输出中文文本
        # 例如调用语音识别API等

        return jsonify({
            "status": "success",
            "text": text
        })


@app.route('/local', methods=['POST'])
def local():
    save_path = config.INPUT_WAV_PATH
    # 这里可以添加音频处理逻辑
    start_time = time.time()
    segments, _ = model.transcribe(save_path)  # 替换为你的中文音频文件
    text = "".join([seg.text for seg in segments])
    print(f"use time is {time.time() - start_time}s")
    print(text)  # 输出中文文本
    # 例如调用语音识别API等

    return jsonify({
        "status": "success",
        "text": text
    })


if __name__ == '__main__':
    # 创建上传目录
    # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5001, debug=True)

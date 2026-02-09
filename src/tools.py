import os
import sys
import subprocess
import numpy as np
import torch
import yaml
import warnings
import soundfile as sf  # 替代librosa.output，兼容新版librosa
from ml_collections import ConfigDict

# ===================== 关键配置：提前初始化变量+添加路径 =====================
# 过滤无关警告
warnings.filterwarnings("ignore")
# 添加项目根路径，确保能导入模型模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# 提前初始化temp_wav，避免NameError
temp_wav = None


# ===================== 1. 手动初始化MelBandRoformer模型（无复杂依赖） =====================
def init_mel_band_roformer(config_path, weights_path, device):
    """
    直接初始化模型，绕过utils/utils的复杂导入
    """
    # 1. 导入模型（确保modules/bs_roformer路径正确）
    try:
        from modules.bs_roformer.mel_band_roformer import MelBandRoformer
    except ImportError as e:
        raise ImportError(f"导入模型失败，请检查路径：{e}\n确认路径：modules/bs_roformer/mel_band_roformer.py")

    # 2. 加载配置文件
    with open(config_path, encoding="utf-8") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = ConfigDict(config_dict)

    # 3. 初始化模型（仅保留核心参数）
    model = MelBandRoformer(
        dim=config.model.dim,
        depth=config.model.depth,
        heads=config.model.heads,
        dim_head=config.model.dim_head,
        attn_dropout=config.model.attn_dropout,
        ff_dropout=config.model.ff_dropout,
        num_bands=config.model.num_bands,
        dim_freqs_in=config.model.dim_freqs_in,
        stft_n_fft=config.model.stft_n_fft,
        stft_hop_length=config.model.stft_hop_length,
        stft_win_length=config.model.stft_win_length,
        stft_normalized=config.model.stft_normalized,
        stereo=config.model.stereo,
        flash_attn=config.model.flash_attn,
        time_transformer_depth=config.model.time_transformer_depth,
        freq_transformer_depth=config.model.freq_transformer_depth,
        mask_estimator_depth=config.model.mask_estimator_depth,
        num_stems=config.model.num_stems
    )

    # 4. 加载权重
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        # 兼容不同格式的权重文件（有的包含state_dict键）
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # 忽略不匹配的参数（避免权重加载报错）
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 权重加载成功：{weights_path}")
    else:
        raise FileNotFoundError(f"权重文件不存在：{weights_path}")

    # 5. 模型部署到设备并设为推理模式
    model = model.to(device)
    model.eval()
    return model, config


# ===================== 2. 音频预处理（极简版） =====================
def check_ffmpeg():
    """检查ffmpeg是否安装"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        raise RuntimeError("请先安装ffmpeg并添加到环境变量！")


def convert_audio_to_wav(input_path, temp_wav="temp_standard.wav"):
    """任意格式转44100Hz/2声道/16bit WAV"""
    check_ffmpeg()
    # 兼容Windows路径
    input_path = input_path.replace("\\", "/")
    temp_wav = temp_wav.replace("\\", "/")

    cmd = (
        f'ffmpeg -i "{input_path}" '
        f'-loglevel error -ar 44100 -ac 2 -c:a pcm_s16le '
        f'-y "{temp_wav}"'
    )
    subprocess.run(cmd, shell=True, check=True)
    return temp_wav


def load_audio(input_path):
    """加载音频并返回[channels, samples]的float32数组"""
    global temp_wav
    # 1. 转WAV（非WAV格式）
    if not input_path.lower().endswith(".wav"):
        temp_wav = "temp_standard.wav"
        input_path = convert_audio_to_wav(input_path, temp_wav)

    # 2. 用soundfile加载（比librosa更稳定）
    audio, sr = sf.read(input_path, dtype="float32")
    # 确保采样率是44100
    if sr != 44100:
        import librosa
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=44100).T
        sr = 44100

    # 3. 维度标准化：[channels, samples]
    if audio.ndim == 1:  # 单声道转立体声
        audio = np.repeat(audio[:, np.newaxis], 2, axis=1).T
    else:  # [samples, channels] → [channels, samples]
        audio = audio.T

    # 4. 幅值归一化
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    return audio, sr


def split_audio(audio, chunk_size=485100, num_overlap=4):
    """音频分块，返回[1,2,485100]的Tensor列表"""
    hop_size = chunk_size // num_overlap
    total_samples = audio.shape[1]

    # 补零
    if total_samples < chunk_size:
        pad_len = chunk_size - total_samples
        audio = np.pad(audio, ((0, 0), (0, pad_len)), mode="constant")
    elif (total_samples - chunk_size) % hop_size != 0:
        pad_len = hop_size - ((total_samples - chunk_size) % hop_size)
        audio = np.pad(audio, ((0, 0), (0, pad_len)), mode="constant")

    # 分块并转Tensor
    chunks = []
    for start in range(0, audio.shape[1] - chunk_size + 1, hop_size):
        chunk = audio[:, start:start + chunk_size]
        # 转Tensor：[2,485100] → [1,2,485100]
        chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).float()
        chunks.append(chunk_tensor)

    return chunks, audio.shape[1]


# ===================== 3. 分块合并+音频保存 =====================
def merge_audio_chunks(chunks, original_length, chunk_size=485100, num_overlap=4):
    """重叠相加合并分块音频"""
    hop_size = chunk_size // num_overlap
    # 初始化输出
    output = np.zeros((2, original_length))
    weight = np.zeros((2, original_length))

    for i, chunk in enumerate(chunks):
        start = i * hop_size
        end = start + chunk_size

        # 截断超出原始长度的部分
        if end > original_length:
            end = original_length
            chunk = chunk[:, :end - start]

        # 叠加chunk和权重
        chunk_np = chunk.squeeze(0).cpu().numpy()
        output[:, start:end] += chunk_np
        weight[:, start:end] += 1

    # 权重归一化
    weight[weight == 0] = 1
    output = output / weight
    # 限制幅值在[-1,1]
    output = np.clip(output, -1.0, 1.0)

    return output


def save_wav(audio, output_path, sr=44100):
    """保存音频：[channels, samples] → [samples, channels]"""
    audio = audio.T  # 转soundfile要求的格式
    sf.write(output_path, audio, sr)
    print(f"✅ 音频已保存：{output_path}")


# ===================== 4. 主流程（无复杂依赖） =====================
if __name__ == "__main__":
    # ********** 请根据你的路径修改 **********
    CONFIG_PATH = "../configs/inst_v1e.ckpt.yaml"  # 配置文件
    WEIGHTS_PATH = "../models/inst_v1e.ckpt"  # 权重文件
    INPUT_AUDIO = "../input/qqqq.m4a"  # 输入音频
    OUTPUT_AUDIO = "../output/separated.wav"  # 输出音频
    # ****************************************

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备：{device}")

    try:
        # Step1: 初始化模型（核心修复：手动加载，无复杂依赖）
        model, config = init_mel_band_roformer(CONFIG_PATH, WEIGHTS_PATH, device)
        print(f"✅ 模型初始化完成")

        # Step2: 加载并分块音频
        audio, sr = load_audio(INPUT_AUDIO)
        audio_chunks, original_len = split_audio(audio)
        print(f"✅ 音频加载完成：")
        print(f"   - 原始形状：{audio.shape} | 分块数：{len(audio_chunks)}")

        # Step3: 分块推理（禁用梯度，节省显存）
        print(f"\n🚀 开始推理...")
        output_chunks = []
        with torch.no_grad():
            for i, chunk in enumerate(audio_chunks):
                chunk = chunk.to(device)
                out = model(chunk)  # 直接传入时域Tensor，模型内部转频谱
                output_chunks.append(out)
                if (i + 1) % 10 == 0:  # 每10块打印一次进度
                    print(f"   进度：{i + 1}/{len(audio_chunks)}")

        # Step4: 合并+保存
        merged_audio = merge_audio_chunks(output_chunks, original_len)
        os.makedirs(os.path.dirname(OUTPUT_AUDIO), exist_ok=True)
        save_wav(merged_audio, OUTPUT_AUDIO)

    except Exception as e:
        print(f"\n❌ 错误：{str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        # 清理临时文件
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
            print(f"\n🗑️ 清理临时文件：{temp_wav}")
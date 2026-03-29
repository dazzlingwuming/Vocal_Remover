import os
import subprocess
import tempfile
import soundfile as sf
import librosa
import numpy as np
import torch
import yaml
from ml_collections import ConfigDict
from modules.bs_roformer import MelBandRoformer


def get_model_from_config(config_path,weights_path=None, device=None):

	# 1. 加载配置并初始化模型结构
	with open(config_path) as f:
		config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
		model = MelBandRoformer(**dict(config.model))
	# 2. 加载权重（如果指定权重路径）
	if weights_path and os.path.exists(weights_path):
		# 加载权重文件（兼容带/不带state_dict外层key的情况）
		state_dict = torch.load(weights_path, map_location=device)
		if "state_dict" in state_dict:  # 权重文件包含state_dict外层key
			state_dict = state_dict["state_dict"]
		# 加载权重（strict=False兼容少量key命名差异）
		model.load_state_dict(state_dict, strict=False)
		print(f"✅ 权重文件 {weights_path} 加载成功！")

		# 3. 部署模型到指定设备（CPU/GPU）
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	model.eval()  # 推理模式（禁用Dropout等训练层）
	print(f"✅ 模型已部署到设备：{device}")

	return model, config


def model_run(model, config, device):
	"""构造测试输入，验证模型前向传播是否正常"""
	# 从config中提取音频参数，确保输入形状匹配
	sample_rate = config.audio.sample_rate  # 44100（固定）
	num_channels = config.audio.num_channels  # 2（立体声）
	# 构造测试输入：[batch_size, num_channels, audio_length]
	# audio_length参考config.inference.dim_t * config.audio.hop_length（适配模型输入维度）
	test_audio_length = config.inference.dim_t * config.audio.hop_length  # 比如 1101*441=485541
	test_input = torch.randn(1, num_channels, test_audio_length).to(device)  # batch_size=1

	# 前向传播测试（禁用梯度计算，提升速度）
	with torch.no_grad():
		try:
			output = model(test_input)
			print(f"✅ 模型前向传播成功！")
			print(f"输入形状：{test_input.shape}")
			print(f"输出形状：{output.shape}")  # 输出形状应和输入一致（分离后音频）
			return True
		except Exception as e:
			print(f"❌ 前向传播失败：{str(e)}")
			return False

# 用 ffmpeg 将任何音频文件解码为 wav 并读取为 numpy
# ------------------------------------------------------------
def load_audio_ffmpeg(audio_path, target_sr=44100):
    """
    使用 ffmpeg 解码音频，返回 (audio_np: (channels, samples), sample_rate)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"文件不存在: {audio_path}")

    # 临时 wav 文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmp_wav = tmpfile.name

    # 调用 ffmpeg 转换为目标采样率的 wav（立体声）
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-ar", str(target_sr),
        "-ac", "2",
        "-f", "wav",
        "-y", tmp_wav
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 转换失败: {e.stderr.decode()}")

    # 读取 wav 文件
    data, sr = sf.read(tmp_wav, always_2d=True)   # data shape: (samples, channels)
    # 转置为 (channels, samples)
    audio_np = data.T
    os.unlink(tmp_wav)  # 删除临时文件
    return audio_np, sr

# 分块推理：避免长音频显存不足
# ------------------------------------------------------------
def process_chunks(model, audio_tensor, device, chunk_duration=10.0, sr=44100):
    total_samples = audio_tensor.shape[-1]
    chunk_samples = int(chunk_duration * sr)
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    instrumental_chunks = []
    vocal_chunks = []

    with torch.no_grad():
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, total_samples)
            chunk = audio_tensor[:, :, start:end].to(device)
            inst_chunk = model(chunk)
            # 对齐长度：取两个张量长度的最小值
            min_len = min(chunk.shape[-1], inst_chunk.shape[-1])
            chunk = chunk[:, :, :min_len]          # 裁剪输入
            inst_chunk = inst_chunk[:, :, :min_len] # 裁剪输出
            voc_chunk = chunk - inst_chunk
            instrumental_chunks.append(inst_chunk.cpu())
            vocal_chunks.append(voc_chunk.cpu())

    # 拼接所有块
    instrumental = torch.cat(instrumental_chunks, dim=-1)
    vocal = torch.cat(vocal_chunks, dim=-1)
    # 最终长度可能略小于原始 total_samples（每块丢弃了尾部几个样本），不影响使用
    return instrumental, vocal

# 新增：模型推理得到伴奏，然后相减得到人声
# ------------------------------------------------------------
def extract_vocal_by_subtraction(model, audio_tensor, device):
    """
    audio_tensor: (1, channels, samples)
    返回: instrumental_tensor, vocal_tensor (形状相同)
    """
    with torch.no_grad():
        audio_tensor = audio_tensor.to(device)
        instrumental = model(audio_tensor)          # 模型输出伴奏
        vocal = audio_tensor - instrumental         # 原始减伴奏 = 人声
    return instrumental.cpu(), vocal.cpu()



# 主流程
# ------------------------------------------------------------
if __name__ == "__main__":
    # 配置路径
    weights_path = "../models/inst_v1e.ckpt"
    config_path = "../configs/inst_v1e.ckpt.yaml"
    input_audio = r"D:\github\Vocal_Remover\Bili_video_audio\output\《不凡2024》4K动画韩立结婴曲 王铮亮_1774767238\audio.m4s"
    output_dir = os.path.dirname(input_audio)

    # 输出文件
    instrumental_path = os.path.join(output_dir, "instrumental.wav")
    vocal_path = os.path.join(output_dir, "vocal.wav")

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = get_model_from_config(config_path, weights_path=weights_path, device=device)

    # 加载音频 (用 ffmpeg)
    print(f"加载音频: {input_audio}")
    audio_np, sr = load_audio_ffmpeg(input_audio, target_sr=config.audio.sample_rate)
    print(f"音频形状 (channels, samples): {audio_np.shape}, 采样率: {sr}")

    # 转换为 torch tensor
    audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)  # (1, channels, samples)

    # 分离（分块，每块10秒，可根据显存调整）
    print("正在分离伴奏（分块推理）...")
    instrumental, vocal = process_chunks(model, audio_tensor, device, chunk_duration=10.0, sr=sr)

    # 保存
    instrumental_np = instrumental.squeeze(0).numpy().T  # (samples, channels)
    vocal_np = vocal.squeeze(0).numpy().T

    sf.write(instrumental_path, instrumental_np, sr)
    sf.write(vocal_path, vocal_np, sr)

    print(f"✅ 伴奏已保存: {instrumental_path}")
    print(f"✅ 人声已保存: {vocal_path}")

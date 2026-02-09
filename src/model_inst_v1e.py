import os
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
if __name__ == "__main__":
	weights_path = "../models/inst_v1e.ckpt"
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model, config = get_model_from_config("../configs/inst_v1e.ckpt.yaml",weights_path=weights_path,device=device)
	print(model)
	model_run(model, config, device)

# Vocal Remover KTV (Bilibili LAN KTV)

基于 Bilibili 搜索 + 本地提取 + 人声/伴奏分离 的家庭 KTV 项目。  
支持 TV 页面播放、手机扫码点歌、队列管理、原唱/伴奏切换和音量控制。

## 功能特性

- B 站歌曲搜索（分页）
- 提取音视频并入本地媒体库
- 自动分离伴奏与人声（并缓存结果）
- 手机端点歌 + 任务进度条
- TV 端播放队列 + 自动下一首
- 局域网二维码入房（房间模式）
- 本地媒体库浏览/筛选/点歌

## 项目结构

- `Bili_video_audio/falsk_reseach/flask_search.py`：Flask 主服务
- `Bili_video_audio/falsk_reseach/templates/tv.html`：TV 端页面
- `Bili_video_audio/falsk_reseach/templates/mobile.html`：手机点歌页面
- `Bili_video_audio/output/`：音视频、封面、分离结果统一存放目录
- `models/inst_v1e.ckpt`：分离模型权重
- `configs/inst_v1e.ckpt.yaml`：模型配置

## 环境要求

- Python 3.10+（建议 3.10/3.11）
- FFmpeg（需加入 PATH）
- Windows / Linux 均可（当前主要在 Windows 场景验证）
- PyTorch（CPU 或 CUDA 版本，见下方安装说明）

## 安装步骤

### 1) 安装 PyTorch

请按你的机器环境选择其一：

- `CPU`：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

- `NVIDIA CUDA 12.1`：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2) 安装项目依赖

```bash
pip install -r requirements.txt
```

### 3) 准备模型权重（必须）

本仓库默认**不包含大模型权重文件**（通常不会上传到 GitHub）。

请自行下载并放到以下路径：

- 权重文件：`models/inst_v1e.ckpt`
- 配置文件：`configs/inst_v1e.ckpt.yaml`（仓库已提供）

如果 `models/inst_v1e.ckpt` 缺失，分离伴奏/人声功能无法工作。

### 4) 启动服务

```bash
python Bili_video_audio/falsk_reseach/flask_search.py
```

默认监听：`0.0.0.0:5000`

## 使用方式

1. 打开 TV 页面：`http://<电脑IP>:5000/tv?room=ktv001`
2. 手机扫码进入：`http://<电脑IP>:5000/mobile?room=ktv001`
3. 手机端搜索歌曲，点击“提取并点歌”
4. TV 端自动进入队列播放

## 首次运行自检

启动后建议先做这 3 项检查：

1. 打开 `http://127.0.0.1:5000/api/ping`，确认服务在线
2. 在网页执行一次“提取并点歌”，确认音视频提取正常
3. 执行一次“分离伴奏/人声”，确认模型加载和推理正常

## 重要说明

- 手机连电脑热点时，通常应使用 `192.168.137.1` 网段地址。
- 服务端默认动态探测局域网 IP 生成二维码。
- 如需强制指定对外 IP（可选）：

```powershell
$env:KTV_LAN_IP="192.168.137.1"
python Bili_video_audio/falsk_reseach/flask_search.py
```

## 常见问题

### 1) 手机能打开 TV 页面但扫码进不去

- 检查手机和电脑是否在同一网段
- 检查防火墙是否放行 5000 端口
- 尝试直接访问 `http://<IP>:5000/api/ping`

### 2) 分离时报 shape/broadcast 错误

历史版本可能出现音频长度不一致导致异常；当前版本已做长度对齐修复。

### 3) 切歌后声音异常

建议先确认当前模式（原唱/伴奏）和人声/伴奏音量状态是否符合预期。

### 4) 分离功能不可用 / 报模型相关错误

优先检查：

- `models/inst_v1e.ckpt` 是否存在且文件完整
- `configs/inst_v1e.ckpt.yaml` 是否存在
- `torch` 是否正确安装（CPU/CUDA 版本是否匹配你的环境）

## Roadmap

- PWA 手机安装
- Android TV/机顶盒壳应用
- 鉴权与房间管理
- 生产部署（Nginx + WSGI）

## License

仅供学习与个人研究使用。请遵守 Bilibili 及相关内容平台的服务条款与版权规范。

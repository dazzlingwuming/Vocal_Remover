import asyncio

from bilibili_api import video, Credential, HEADERS, get_client
import os

SESSDATA = ""
BILI_JCT = ""
BUVID3 = ""

# FFMPEG 路径，查看：http://ffmpeg.org/
FFMPEG_PATH = "ffmpeg"


async def download(url: str, out: str, intro: str):
    dwn_id = await get_client().download_create(url, HEADERS)
    bts = 0
    tot = get_client().download_content_length(dwn_id)
    with open(out, "wb") as file:
        while True:
            bts += file.write(await get_client().download_chunk(dwn_id))
            print(f"{intro} - {out} [{bts} / {tot}]", end="\r")
            if bts == tot:
                break
    print()


async def main():
    # 实例化 Credential 类
    credential = Credential(sessdata=SESSDATA, bili_jct=BILI_JCT, buvid3=BUVID3)
    # 实例化 Video 类
    v = video.Video(bvid="BV1R1UkBaEUQ", credential=credential)
    name = (await v.get_info())["title"]
    #创建文件夹
    if not os.path.exists(f"output/{name}"):
        os.mkdir(f"output/{name}")
    # 获取视频下载链接
    download_url_data = await v.get_download_url(0)
    # 解析视频下载信息
    detecter = video.VideoDownloadURLDataDetecter(data=download_url_data)
    streams = detecter.detect_best_streams()
    # 有 MP4 流 / FLV 流两种可能
    if detecter.check_flv_mp4_stream() == True:
        # FLV 流下载
        await download(streams[0].url, f"output/{name}/{name}.flv", "下载 FLV 音视频流")
        # 转换文件格式
        os.system(f"{FFMPEG_PATH} -i output/{name}/{name}.flv output/{name}/{name}.mp4")
        # 删除临时文件
        os.remove(f"{name}.flv")
    else:
        # MP4 流下载
        await download(streams[0].url, f"output/{name}/video.m4s", "下载视频流")
        await download(streams[1].url, f"output/{name}/audio.m4s", "下载音频流")
        # 混流
        # os.system(
        #     f"{FFMPEG_PATH} -i video_temp.m4s -i audio_temp.m4s -vcodec copy -acodec copy video.mp4"
        # )
        # 删除临时文件
        # os.remove("video_temp.m4s")
        # os.remove("audio_temp.m4s")

    print(f"已下载为：{name}.mp4")


if __name__ == "__main__":
    # 主入口
    asyncio.run(main())

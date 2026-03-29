from bilibili_api import audio, sync, get_session,Credential
import os

from bilibili_api.video import Video, AudioStreamDownloadURL, VideoDownloadURLDataDetecter
from curl_cffi import requests

credential = Credential(sessdata="e181e290%2C1786178882%2Ccdb01%2A21CjCtjNYiylroxRsgt8SkXm59K2JkPd5eMIL_-eSf_I5U_9PQDlp1JS9WQ6uFvXavmkYSVnVSaWtKX184eEFydkpNTzRfdVBVSURmYjZTNmpkR1piNUwzWnhrbWMzVkxrVHBKVE84aXJlRldCSVFyN2diaExkXzlYM013MHRLM2FrNGNXWDFubTF3IIEC", bili_jct="7bc50afec6215d63d71e928c62ea8926")

AUDIO_LIST_ID = 30232


async def main1():
    a = Video(bvid="BV1R1UkBaEUQ", aid=115604983976735,credential=credential)
    info = await a.get_info()
    print(info)
    name = info["title"]
    cid = info["pages"][0]["cid"]
    print(name)
    dict = await a.get_download_url(cid=cid)

    b = VideoDownloadURLDataDetecter(dict)
    b = b.detect_best_streams()
    print(b)
    c = AudioStreamDownloadURL(b[1].url,b[1].audio_quality)
    print(c)
    #下载
    sess: requests.AsyncSession = (
        get_session()
    )
    # 下载歌曲
    file = f"output/{name}.m4a"
    print(f"下载 {name}")
    resp: requests.Response = await sess.get(
        c.url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
            "Referer": "https://www.bilibili.com/",
        },
        stream=True,
    )
    with open(file, "wb") as f:
        async for chunk in resp.aiter_content():
            if not chunk:
                break
            f.write(chunk)

sync(main1())

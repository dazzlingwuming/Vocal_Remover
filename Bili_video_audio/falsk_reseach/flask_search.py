import asyncio
import hashlib
import io
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image, ImageDraw
from bilibili_api import search, sync
from bilibili_api.search import SearchObjectType
from bilibili_api.video import Video, VideoDownloadURLDataDetecter
from flask import Flask, jsonify, make_response, redirect, render_template, request, send_file

import src.tools as separator_tools

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "*")
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Range"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Range, Accept-Ranges"
    response.headers["Access-Control-Allow-Credentials"] = "false"
    return response

REPO_ROOT = Path(__file__).resolve().parents[2]
BILI_ROOT = Path(__file__).resolve().parents[1]
KTV_OUTPUT_ROOT = BILI_ROOT / "output"
SEPARATE_OUTPUT_ROOT = REPO_ROOT / "output"
CONFIG_PATH = REPO_ROOT / "configs" / "inst_v1e.ckpt.yaml"
WEIGHTS_PATH = REPO_ROOT / "models" / "inst_v1e.ckpt"
MEDIA_INDEX_PATH = KTV_OUTPUT_ROOT / "media_index.json"

KTV_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
SEPARATE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

TASKS = {}
TASKS_LOCK = threading.Lock()
MODEL_CACHE = {"model": None, "config": None, "device": None}
MODEL_LOCK = threading.Lock()
MIGRATION_DONE = False
ROOMS = {}
ROOMS_LOCK = threading.Lock()
QR_CACHE = {}
QR_CACHE_LOCK = threading.Lock()


class BiliSearch:
    def __init__(self):
        self.video_type = SearchObjectType("video")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                "Referer": "https://www.bilibili.com/",
            }
        )

    def search_videos(self, keyword, page=1):
        try:
            return sync(search.search_by_type(keyword, search_type=self.video_type, page=page))
        except Exception as exc:
            print(f"search failed: {exc}")
            return None


bili_search = BiliSearch()


def now_ms() -> int:
    return int(time.time() * 1000)


def room_song_key(song: dict) -> str:
    if not song:
        return ""
    bvid = song.get("bvid", "")
    if bvid:
        return f"bvid:{bvid}"
    audio = normalize_path(song.get("audio_path", ""))
    if audio:
        return f"audio:{audio}"
    title = safe_name(song.get("title", "")).lower()
    author = (song.get("author", "") or "").lower()
    if title:
        return f"title:{title}|author:{author}"
    return hashlib.md5(json.dumps(song, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def get_room(room_id: str) -> dict:
    room_id = (room_id or "ktv001").strip() or "ktv001"
    with ROOMS_LOCK:
        room = ROOMS.get(room_id)
        if room is None:
            room = {
                "room_id": room_id,
                "queue": [],
                "current_idx": -1,
                "mode": "accomp",
                "accomp_volume": 1.0,
                "vocal_volume": 0.0,
                "updated_at": now_ms(),
            }
            ROOMS[room_id] = room
        return room


def room_state_payload(room_id: str) -> dict:
    room = get_room(room_id)
    queue = room.get("queue", [])
    cur_idx = int(room.get("current_idx", -1))
    current = queue[cur_idx] if 0 <= cur_idx < len(queue) else None
    return {
        "room_id": room_id,
        "queue": queue,
        "queue_count": len(queue),
        "current_idx": cur_idx,
        "current_song": current,
        "mode": room.get("mode", "accomp"),
        "accomp_volume": float(room.get("accomp_volume", 1.0)),
        "vocal_volume": float(room.get("vocal_volume", 0.0)),
        "updated_at": room.get("updated_at", now_ms()),
    }


def detect_lan_ip() -> str:
    forced = (os.environ.get("KTV_LAN_IP") or "").strip()
    if forced:
        try:
            ipaddress.ip_address(forced)
            if not forced.startswith("127."):
                return forced
        except Exception:
            pass

    def parse_ipconfig_ipv4() -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        try:
            proc = subprocess.run(
                ["ipconfig"],
                capture_output=True,
                text=True,
                encoding="gbk",
                errors="ignore",
            )
            out = proc.stdout or ""
            blocks = re.split(r"\r?\n\r?\n+", out)
            for block in blocks:
                header = (block.splitlines() or [""])[0].strip().rstrip(":")
                if not header:
                    continue
                m = re.search(r"IPv4[^\n:：]*[:：]\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", block)
                if not m:
                    continue
                ip = (m.group(1) or "").strip()
                if not ip or ip.startswith("127.") or ip == "0.0.0.0" or ip.startswith("169.254."):
                    continue
                try:
                    ip_obj = ipaddress.ip_address(ip)
                    if not ip_obj.is_private:
                        continue
                except Exception:
                    continue
                name_lower = header.lower()
                if any(bad in name_lower for bad in ["meta", "vmware", "virtualbox", "hyper-v", "loopback", "bluetooth"]):
                    continue
                if ip.startswith("198.18."):
                    continue
                pair = (header, ip)
                if pair not in pairs:
                    pairs.append(pair)
        except Exception:
            pass
        return pairs

    cfg_pairs = parse_ipconfig_ipv4()
    cfg_ips = [ip for _name, ip in cfg_pairs]
    # Windows hotspot usually binds 192.168.137.1; prioritize this for mobile scan/join.
    for ip in cfg_ips:
        if ip.startswith("192.168.137."):
            return ip
    for name, ip in cfg_pairs:
        name_lower = name.lower()
        if ("本地连接*" in name or "local connection*" in name_lower or "wlan" in name_lower or "无线局域网" in name) and ip.startswith("192.168."):
            return ip
    for ip in cfg_ips:
        if ip.startswith("192.168."):
            return ip
    for ip in cfg_ips:
        if ip.startswith("10."):
            return ip

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127.") and ip in cfg_ips:
            return ip
    except Exception:
        pass
    if cfg_ips:
        return cfg_ips[0]
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    return "127.0.0.1"


def mobile_join_url(room_id: str) -> str:
    host = request.host or "127.0.0.1:5000"
    scheme = request.scheme or "http"
    h = host
    if host.startswith("127.0.0.1") or host.startswith("localhost"):
        port = ""
        if ":" in host:
            port = ":" + host.split(":")[-1]
        h = f"{detect_lan_ip()}{port}"
    return f"{scheme}://{h}/mobile?room={room_id}"


def fallback_qr_image(text: str) -> bytes:
    img = Image.new("RGB", (512, 512), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((8, 8, 503, 503), outline=(30, 30, 30), width=3)
    d.text((20, 20), "Scan Failed - Open URL manually:", fill=(20, 20, 20))
    lines = [text[i : i + 36] for i in range(0, len(text), 36)]
    y = 56
    for ln in lines[:10]:
        d.text((20, y), ln, fill=(40, 40, 40))
        y += 24
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def create_task(task_type: str, payload: dict) -> str:
    task_id = uuid.uuid4().hex
    with TASKS_LOCK:
        TASKS[task_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "status": "queued",
            "progress": 0,
            "message": "task created",
            "payload": payload,
            "result": None,
            "error": None,
            "created_at": now_ms(),
            "updated_at": now_ms(),
        }
    return task_id


def update_task(task_id: str, **kwargs):
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task:
            return
        task.update(kwargs)
        task["updated_at"] = now_ms()


def get_task(task_id: str):
    with TASKS_LOCK:
        return TASKS.get(task_id)


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", (name or "untitled").strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:80] if cleaned else "untitled"


def parse_bvid(text: str) -> str:
    if not text:
        return ""
    matched = re.search(r"BV[0-9A-Za-z]{10}", text)
    return matched.group(0) if matched else ""


def run_ffmpeg(cmd: list[str]) -> bool:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode == 0
    except Exception:
        return False


def read_media_index() -> list[dict]:
    if not MEDIA_INDEX_PATH.exists():
        return []
    try:
        data = json.loads(MEDIA_INDEX_PATH.read_text(encoding="utf-8"))
        entries = data.get("entries", [])
        return entries if isinstance(entries, list) else []
    except Exception:
        return []


def write_media_index(entries: list[dict]):
    MEDIA_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEDIA_INDEX_PATH.write_text(
        json.dumps({"entries": entries, "updated_at": now_ms()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def merge_media_records(entries: list[dict]) -> list[dict]:
    """Merge duplicated media records by bvid/path keys."""
    merged: list[dict] = []
    key_to_idx: dict[str, int] = {}

    def keys_of(item: dict) -> list[str]:
        keys = []
        bvid = item.get("bvid", "")
        audio = normalize_path(item.get("audio_path", ""))
        video = normalize_path(item.get("video_path", ""))
        work_dir = normalize_path(item.get("work_dir", ""))
        if bvid:
            keys.append(f"b:{bvid}")
        if audio:
            keys.append(f"a:{audio}")
        if video:
            keys.append(f"v:{video}")
        if work_dir:
            keys.append(f"w:{work_dir}")
        if not keys:
            title = safe_name(item.get("title", "")).lower()
            if title:
                keys.append(f"t:{title}")
        return keys

    for item in sorted(entries, key=lambda x: x.get("updated_at", 0), reverse=True):
        item = {**item}
        hit_idx = -1
        for k in keys_of(item):
            if k in key_to_idx:
                hit_idx = key_to_idx[k]
                break

        if hit_idx < 0:
            merged.append(item)
            idx = len(merged) - 1
        else:
            old = merged[hit_idx]
            cur = {**old, **item}
            for field in ["bvid", "title", "source", "audio_path", "video_path", "separated_path", "vocal_path", "cover_path", "work_dir", "id"]:
                if old.get(field) and not item.get(field):
                    cur[field] = old.get(field)
            cur["created_at"] = old.get("created_at") or item.get("created_at", 0)
            cur["updated_at"] = max(int(old.get("updated_at", 0)), int(item.get("updated_at", 0)))
            merged[hit_idx] = cur
            idx = hit_idx

        for k in keys_of(merged[idx]):
            key_to_idx[k] = idx

    return merged


def normalize_path(raw_path: str) -> str:
    if not raw_path:
        return ""
    return str(Path(raw_path).resolve())


def file_exists(raw_path: str) -> bool:
    if not raw_path:
        return False
    try:
        path = resolve_allowed_file(raw_path)
        return path.exists() and path.is_file()
    except Exception:
        return False


def guess_separated_path(audio_path: str, title: str = "") -> str:
    candidates = []
    if audio_path:
        audio_p = Path(audio_path)
        stem = audio_p.stem
        digest = hashlib.md5(str(audio_p.resolve()).encode("utf-8")).hexdigest()[:8]
        if audio_p.parent.exists():
            candidates.append(audio_p.parent / f"{safe_name(stem)}_{digest}_separated.wav")
            candidates.append(audio_p.parent / f"{safe_name(stem)}_separated.wav")
        candidates.append(SEPARATE_OUTPUT_ROOT / f"{safe_name(stem)}_{digest}_separated.wav")
        candidates.append(SEPARATE_OUTPUT_ROOT / f"{safe_name(stem)}_separated.wav")
    if title:
        candidates.append(SEPARATE_OUTPUT_ROOT / f"{safe_name(title)}_separated.wav")
    for item in candidates:
        if item.exists() and item.is_file():
            return str(item.resolve())
    return ""


def guess_vocal_path(audio_path: str) -> str:
    if not audio_path:
        return ""
    audio_p = Path(audio_path)
    stem = safe_name(audio_p.stem)
    digest = hashlib.md5(str(audio_p.resolve()).encode("utf-8")).hexdigest()[:8]
    candidates = []
    if audio_p.parent.exists():
        candidates.append(audio_p.parent / f"{stem}_{digest}_vocal.wav")
        candidates.append(audio_p.parent / f"{stem}_vocal.wav")
    candidates.append(SEPARATE_OUTPUT_ROOT / f"{stem}_{digest}_vocal.wav")
    candidates.append(SEPARATE_OUTPUT_ROOT / f"{stem}_vocal.wav")
    for p in candidates:
        if p.exists() and p.is_file():
            return str(p.resolve())
    return ""


def guess_cover_path(work_dir: str = "") -> str:
    if not work_dir:
        return ""
    base = Path(work_dir)
    if not base.exists() or not base.is_dir():
        return ""
    for name in ["cover.jpg", "cover.jpeg", "cover.png", "cover.webp"]:
        p = base / name
        if p.exists() and p.is_file():
            return str(p.resolve())
    return ""


def download_cover_image(url: str, run_dir: Path) -> str:
    if not url:
        return ""
    try:
        if url.startswith("//"):
            url = f"https:{url}"
        elif not url.startswith("http"):
            url = f"https://{url}"
        resp = bili_search.session.get(url, timeout=20)
        resp.raise_for_status()
        ctype = (resp.headers.get("Content-Type", "") or "").lower()
        ext = ".jpg"
        if "png" in ctype:
            ext = ".png"
        elif "webp" in ctype:
            ext = ".webp"
        elif "jpeg" in ctype or "jpg" in ctype:
            ext = ".jpg"
        out = run_dir / f"cover{ext}"
        out.write_bytes(resp.content)
        return str(out.resolve())
    except Exception:
        return ""


def migrate_legacy_stems():
    """Move old stems from outer output directory into each song work_dir."""
    global MIGRATION_DONE
    if MIGRATION_DONE:
        return
    entries = read_media_index()
    changed = False

    for item in entries:
        work_dir = item.get("work_dir", "")
        if not work_dir:
            continue
        wd = Path(work_dir)
        if not wd.exists() or not wd.is_dir() or not path_under(wd, KTV_OUTPUT_ROOT):
            continue

        for field in ["separated_path", "vocal_path"]:
            raw = item.get(field, "")
            if not raw:
                continue
            p = Path(raw)
            try:
                p = p.resolve()
            except Exception:
                continue
            if not p.exists() or not p.is_file():
                continue
            if path_under(p, KTV_OUTPUT_ROOT):
                continue
            target = wd / p.name
            try:
                if not target.exists():
                    shutil.move(str(p), str(target))
                    changed = True
                item[field] = str(target.resolve())
                changed = True
            except Exception:
                continue

    if changed:
        write_media_index(merge_media_records(entries))
    MIGRATION_DONE = True


def upsert_media_entry(entry: dict):
    entries = read_media_index()
    entry = {**entry}
    entry["audio_path"] = normalize_path(entry.get("audio_path", ""))
    entry["video_path"] = normalize_path(entry.get("video_path", ""))
    entry["separated_path"] = normalize_path(entry.get("separated_path", ""))
    entry["vocal_path"] = normalize_path(entry.get("vocal_path", ""))
    entry["cover_path"] = normalize_path(entry.get("cover_path", ""))
    entry["work_dir"] = normalize_path(entry.get("work_dir", ""))
    entry["title"] = safe_name(entry.get("title", "untitled"))
    entry["updated_at"] = now_ms()
    entry["created_at"] = entry.get("created_at") or now_ms()

    hit_idx = -1
    for idx, old in enumerate(entries):
        old_bvid = old.get("bvid", "")
        old_audio = normalize_path(old.get("audio_path", ""))
        old_video = normalize_path(old.get("video_path", ""))
        if entry.get("bvid") and old_bvid == entry["bvid"]:
            hit_idx = idx
            break
        if entry.get("audio_path") and old_audio == entry["audio_path"]:
            hit_idx = idx
            break
        if entry.get("video_path") and old_video == entry["video_path"]:
            hit_idx = idx
            break

    if hit_idx >= 0:
        merged = {**entries[hit_idx], **entry}
        entries[hit_idx] = merged
    else:
        entries.append(entry)

    write_media_index(merge_media_records(entries))


def find_cached_by_bvid(bvid: str) -> dict:
    if not bvid:
        return {}
    entries = read_media_index()
    for item in sorted(entries, key=lambda x: x.get("updated_at", 0), reverse=True):
        if item.get("bvid") != bvid:
            continue
        if file_exists(item.get("audio_path", "")) or file_exists(item.get("video_path", "")):
            if not item.get("separated_path"):
                sep = guess_separated_path(item.get("audio_path", ""), item.get("title", ""))
                if sep:
                    item["separated_path"] = sep
            if not item.get("vocal_path"):
                voc = guess_vocal_path(item.get("audio_path", ""))
                if voc:
                    item["vocal_path"] = voc
            if not item.get("cover_path"):
                cover = guess_cover_path(item.get("work_dir", ""))
                if cover:
                    item["cover_path"] = cover
            return item
    return {}


def scan_local_library() -> list[dict]:
    migrate_legacy_stems()
    entries = []
    seen = set()

    indexed = read_media_index()
    for item in sorted(indexed, key=lambda x: x.get("updated_at", 0), reverse=True):
        audio_path = item.get("audio_path", "")
        video_path = item.get("video_path", "")
        separated_path = item.get("separated_path", "") or guess_separated_path(audio_path, item.get("title", ""))
        vocal_path = item.get("vocal_path", "") or guess_vocal_path(audio_path)
        cover_path = item.get("cover_path", "") or guess_cover_path(item.get("work_dir", ""))
        if not (file_exists(audio_path) or file_exists(video_path)):
            continue
        key = f"{normalize_path(audio_path)}|{normalize_path(video_path)}"
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "id": item.get("bvid") or hashlib.md5(key.encode("utf-8")).hexdigest()[:10],
                "bvid": item.get("bvid", ""),
                "title": item.get("title", "untitled"),
                "source": item.get("source", "indexed"),
                "audio_path": audio_path,
                "video_path": video_path,
                "separated_path": separated_path if file_exists(separated_path) else "",
                "vocal_path": vocal_path if file_exists(vocal_path) else "",
                "cover_path": cover_path if file_exists(cover_path) else "",
                "work_dir": item.get("work_dir", ""),
                "created_at": item.get("created_at", 0),
                "updated_at": item.get("updated_at", 0),
            }
        )

    for run_dir in sorted(KTV_OUTPUT_ROOT.glob("*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if not run_dir.is_dir():
            continue
        video_candidates = [run_dir / "merged.mp4", run_dir / "video.mp4", run_dir / "video.flv", run_dir / "video.m4s"]
        audio_candidates = [run_dir / "audio.m4a", run_dir / "audio.m4s", run_dir / "audio.mp3", run_dir / "audio.wav"]
        video_path = next((str(p.resolve()) for p in video_candidates if p.exists() and p.is_file()), "")
        audio_path = next((str(p.resolve()) for p in audio_candidates if p.exists() and p.is_file()), "")
        if not (audio_path or video_path):
            continue
        key = f"{normalize_path(audio_path)}|{normalize_path(video_path)}"
        if key in seen:
            continue
        seen.add(key)
        title = re.sub(r"_[0-9]{8,}$", "", run_dir.name)
        separated_path = guess_separated_path(audio_path, title)
        cover_path = guess_cover_path(str(run_dir))
        entries.append(
            {
                "id": hashlib.md5(str(run_dir.resolve()).encode("utf-8")).hexdigest()[:10],
                "bvid": "",
                "title": title or run_dir.name,
                "source": "local_scan",
                "audio_path": audio_path,
                "video_path": video_path,
                "separated_path": separated_path if file_exists(separated_path) else "",
                "vocal_path": "",
                "cover_path": cover_path if file_exists(cover_path) else "",
                "work_dir": str(run_dir.resolve()),
                "created_at": int(run_dir.stat().st_mtime * 1000),
                "updated_at": int(run_dir.stat().st_mtime * 1000),
            }
        )

    return merge_media_records(entries)


def resolve_allowed_file(raw_path: str) -> Path:
    if not raw_path:
        raise ValueError("missing file path")
    path = Path(raw_path)
    path = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    roots = [REPO_ROOT.resolve(), BILI_ROOT.resolve()]
    if not any(str(path).startswith(str(root)) for root in roots):
        raise ValueError("path outside allowed roots")
    return path


def path_under(path: Path, root: Path) -> bool:
    return str(path.resolve()).startswith(str(root.resolve()))


def resolve_existing_path(raw_path: str) -> Path:
    if not raw_path:
        raise ValueError("missing path")
    path = Path(raw_path)
    path = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"path not found: {path}")
    roots = [REPO_ROOT.resolve(), BILI_ROOT.resolve()]
    if not any(path_under(path, root) for root in roots):
        raise ValueError("path outside allowed roots")
    return path


def delete_media_item(payload: dict) -> dict:
    audio_path = payload.get("audio_path", "")
    video_path = payload.get("video_path", "")
    separated_path = payload.get("separated_path", "")
    vocal_path = payload.get("vocal_path", "")
    cover_path = payload.get("cover_path", "")
    work_dir = payload.get("work_dir", "")
    removed_files = []
    removed_dirs = []

    if work_dir:
        try:
            dir_path = resolve_existing_path(work_dir)
            if dir_path.is_dir() and path_under(dir_path, KTV_OUTPUT_ROOT):
                shutil.rmtree(dir_path)
                removed_dirs.append(str(dir_path))
        except FileNotFoundError:
            pass

    for raw in [audio_path, video_path, separated_path, vocal_path, cover_path]:
        if not raw:
            continue
        try:
            file_path = resolve_existing_path(raw)
            if not file_path.is_file():
                continue
            allowed = path_under(file_path, KTV_OUTPUT_ROOT) or path_under(file_path, SEPARATE_OUTPUT_ROOT)
            if not allowed:
                raise ValueError(f"cannot delete outside media folders: {file_path}")
            file_path.unlink(missing_ok=True)
            removed_files.append(str(file_path))
        except FileNotFoundError:
            continue

    target_paths = {normalize_path(x) for x in [audio_path, video_path, separated_path, vocal_path, cover_path, work_dir] if x}
    if target_paths:
        kept = []
        for entry in read_media_index():
            cands = {
                normalize_path(entry.get("audio_path", "")),
                normalize_path(entry.get("video_path", "")),
                normalize_path(entry.get("separated_path", "")),
                normalize_path(entry.get("vocal_path", "")),
                normalize_path(entry.get("cover_path", "")),
                normalize_path(entry.get("work_dir", "")),
            }
            if cands.intersection(target_paths):
                continue
            kept.append(entry)
        write_media_index(kept)

    return {"removed_files": removed_files, "removed_dirs": removed_dirs}


def download_binary(url: str, out_file: Path, task_id: str, stage_base: int, stage_span: int, stage_name: str):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": bili_search.session.headers.get("User-Agent"),
        "Referer": "https://www.bilibili.com/",
    }
    with bili_search.session.get(url, stream=True, timeout=60, headers=headers) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        downloaded = 0
        with open(out_file, "wb") as fp:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                fp.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    progress = stage_base + int((downloaded / total) * stage_span)
                    update_task(task_id, progress=min(progress, stage_base + stage_span), message=f"{stage_name}...")


def get_separator_model():
    with MODEL_LOCK:
        if MODEL_CACHE["model"] is not None:
            return MODEL_CACHE["model"], MODEL_CACHE["config"], MODEL_CACHE["device"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, config = separator_tools.init_mel_band_roformer(str(CONFIG_PATH), str(WEIGHTS_PATH), device)
        MODEL_CACHE["model"] = model
        MODEL_CACHE["config"] = config
        MODEL_CACHE["device"] = device
        return model, config, device


def worker_extract(task_id: str, payload: dict):
    update_task(task_id, status="running", progress=5, message="prepare extract")
    try:
        bvid = payload.get("bvid") or parse_bvid(payload.get("arcurl", ""))
        if not bvid:
            raise ValueError("missing bvid")
        cached = find_cached_by_bvid(bvid)
        if cached:
            update_task(
                task_id,
                status="done",
                progress=100,
                message="extract skipped, use local cache",
                result={
                    "bvid": bvid,
                    "title": cached.get("title", ""),
                    "work_dir": cached.get("work_dir", ""),
                    "video_path": cached.get("video_path", ""),
                    "audio_path": cached.get("audio_path", ""),
                    "separated_path": cached.get("separated_path", ""),
                    "vocal_path": cached.get("vocal_path", ""),
                    "cover_path": cached.get("cover_path", ""),
                    "from_cache": True,
                },
            )
            return

        video_obj = Video(bvid=bvid)
        update_task(task_id, progress=12, message="load video info")
        info = asyncio.run(video_obj.get_info())
        title = safe_name(info.get("title") or payload.get("title") or bvid)
        pic_url = info.get("pic") or payload.get("pic", "")
        cid = info["pages"][0]["cid"]

        update_task(task_id, progress=20, message="fetch stream urls")
        download_data = asyncio.run(video_obj.get_download_url(cid=cid))
        detector = VideoDownloadURLDataDetecter(data=download_data)
        streams = detector.detect_best_streams()

        run_dir = KTV_OUTPUT_ROOT / f"{title}_{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cover_path = download_cover_image(pic_url, run_dir)

        if detector.check_flv_mp4_stream():
            raw_flv = run_dir / "video.flv"
            download_binary(streams[0].url, raw_flv, task_id, 25, 50, "download video")
            mp4_path = run_dir / "video.mp4"
            audio_m4a = run_dir / "audio.m4a"
            run_ffmpeg(["ffmpeg", "-y", "-i", str(raw_flv), str(mp4_path)])
            run_ffmpeg(["ffmpeg", "-y", "-i", str(raw_flv), "-vn", str(audio_m4a)])
            video_path = mp4_path if mp4_path.exists() else raw_flv
            audio_path = audio_m4a if audio_m4a.exists() else raw_flv
        else:
            if len(streams) < 2:
                raise RuntimeError("cannot find separate audio/video streams")
            video_m4s = run_dir / "video.m4s"
            audio_m4s = run_dir / "audio.m4s"
            download_binary(streams[0].url, video_m4s, task_id, 25, 30, "download video stream")
            download_binary(streams[1].url, audio_m4s, task_id, 56, 30, "download audio stream")
            merged_mp4 = run_dir / "merged.mp4"
            run_ffmpeg(["ffmpeg", "-y", "-i", str(video_m4s), "-i", str(audio_m4s), "-c:v", "copy", "-c:a", "aac", str(merged_mp4)])
            video_path = merged_mp4 if merged_mp4.exists() else video_m4s
            audio_path = audio_m4s

        update_task(
            task_id,
            status="done",
            progress=100,
            message="extract done",
            result={
                "bvid": bvid,
                "title": title,
                "work_dir": str(run_dir),
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "cover_path": cover_path,
            },
        )
        upsert_media_entry(
            {
                "bvid": bvid,
                "title": title,
                "source": "bilibili_extract",
                "work_dir": str(run_dir),
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "separated_path": guess_separated_path(str(audio_path), title),
                "cover_path": cover_path,
            }
        )
    except Exception as exc:
        update_task(task_id, status="failed", progress=100, message="extract failed", error=str(exc))


def worker_separate(task_id: str, payload: dict):
    update_task(task_id, status="running", progress=5, message="prepare separate")
    try:
        input_path = resolve_allowed_file(payload.get("audio_path", ""))
        digest = hashlib.md5(str(input_path.resolve()).encode("utf-8")).hexdigest()[:8]
        bvid = payload.get("bvid", "")
        vocal_only = bool(payload.get("vocal_only", False))

        # Try to recover bvid/video/work_dir context from index/local paths.
        indexed = read_media_index()
        matched = None
        input_norm = normalize_path(str(input_path))
        for item in sorted(indexed, key=lambda x: x.get("updated_at", 0), reverse=True):
            if bvid and item.get("bvid", "") == bvid:
                matched = item
                break
            if input_norm and normalize_path(item.get("audio_path", "")) == input_norm:
                matched = item
                break

        if matched and not bvid:
            bvid = matched.get("bvid", "")

        model = None
        device = torch.device("cpu")
        chunks = []

        parent_dir = input_path.parent if input_path.parent.exists() else None
        inferred_video = ""
        inferred_work_dir = ""
        inferred_cover = ""
        output_root = KTV_OUTPUT_ROOT / "_separated"
        if parent_dir and path_under(parent_dir, KTV_OUTPUT_ROOT):
            inferred_work_dir = str(parent_dir.resolve())
            video_candidates = [parent_dir / "merged.mp4", parent_dir / "video.mp4", parent_dir / "video.flv", parent_dir / "video.m4s"]
            inferred_video = next((str(p.resolve()) for p in video_candidates if p.exists() and p.is_file()), "")
            inferred_cover = guess_cover_path(str(parent_dir))
            # Prefer saving separated stems next to extracted media.
            output_root = parent_dir

        if matched:
            inferred_video = inferred_video or matched.get("video_path", "")
            inferred_work_dir = inferred_work_dir or matched.get("work_dir", "")
            inferred_cover = inferred_cover or matched.get("cover_path", "")

        output_name = f"{safe_name(input_path.stem)}_{digest}_separated.wav"
        vocal_output_name = f"{safe_name(input_path.stem)}_{digest}_vocal.wav"
        output_path = output_root / output_name
        vocal_output_path = output_root / vocal_output_name

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Fast path: human-vocal extraction from existing accompaniment (no model run)
        if vocal_only:
            if not file_exists(str(output_path)):
                if matched and file_exists(matched.get("separated_path", "")):
                    output_path = Path(matched.get("separated_path", ""))
                else:
                    raise RuntimeError("未找到伴奏文件，请先执行一次分离伴奏")

            src_audio, _ = separator_tools.load_audio(str(input_path))
            inst_audio, _ = separator_tools.load_audio(str(output_path))
            mix_len = min(src_audio.shape[1], inst_audio.shape[1])
            vocal_audio = np.clip(src_audio[:, :mix_len] - inst_audio[:, :mix_len], -1.0, 1.0)
            separator_tools.save_wav(vocal_audio, str(vocal_output_path))
            chunks = [0]
        else:
            # Reuse existing accompaniment if present; only build missing vocal.
            if file_exists(str(output_path)):
                src_audio, _ = separator_tools.load_audio(str(input_path))
                inst_audio, _ = separator_tools.load_audio(str(output_path))
                mix_len = min(src_audio.shape[1], inst_audio.shape[1])
                vocal_audio = np.clip(src_audio[:, :mix_len] - inst_audio[:, :mix_len], -1.0, 1.0)
                separator_tools.save_wav(vocal_audio, str(vocal_output_path))
                chunks = [0]
                device = "cache"
            else:
                model, _config, device = get_separator_model()
                update_task(task_id, progress=20, message="model loaded")
                audio, _ = separator_tools.load_audio(str(input_path))
                raw_len = audio.shape[1]
                chunks, original_len = separator_tools.split_audio(audio)
                update_task(task_id, progress=35, message=f"infer chunks {len(chunks)}")

                out_chunks = []
                with torch.no_grad():
                    for idx, chunk in enumerate(chunks, start=1):
                        out = model(chunk.to(device))
                        out_chunks.append(out)
                        p = 35 + int((idx / max(len(chunks), 1)) * 55)
                        update_task(task_id, progress=min(p, 92), message=f"infer {idx}/{len(chunks)}")

                merged = separator_tools.merge_audio_chunks(out_chunks, original_len)
                mix_len = min(raw_len, merged.shape[1], audio.shape[1])
                merged = merged[:, :mix_len]
                src_audio = audio[:, :mix_len]
                separator_tools.save_wav(merged, str(output_path))
                vocal_audio = np.clip(src_audio - merged, -1.0, 1.0)
                separator_tools.save_wav(vocal_audio, str(vocal_output_path))

        temp_wav = getattr(separator_tools, "temp_wav", None)
        if temp_wav and Path(temp_wav).exists():
            Path(temp_wav).unlink(missing_ok=True)

        update_task(
            task_id,
            status="done",
            progress=100,
            message="vocal done" if vocal_only else "separate done",
            result={
                "bvid": bvid,
                "input_audio": str(input_path),
                "output_audio": str(output_path),
                "vocal_audio": str(vocal_output_path),
                "device": str(device),
                "chunks": len(chunks),
            },
        )

        upsert_media_entry(
            {
                "bvid": bvid,
                "title": payload.get("title", input_path.stem),
                "source": "separate",
                "work_dir": inferred_work_dir,
                "video_path": inferred_video,
                "cover_path": inferred_cover,
                "audio_path": str(input_path),
                "separated_path": str(output_path),
                "vocal_path": str(vocal_output_path),
            }
        )
    except Exception as exc:
        update_task(task_id, status="failed", progress=100, message="separate failed", error=str(exc))


def worker_separate_from_bvid(task_id: str, payload: dict):
    update_task(task_id, status="running", progress=3, message="prepare bvid separate")
    internal_task_ids: list[str] = []
    try:
        bvid = payload.get("bvid") or parse_bvid(payload.get("arcurl", ""))
        if not bvid:
            raise ValueError("missing bvid")

        cached = find_cached_by_bvid(bvid)
        if cached.get("separated_path") and file_exists(cached.get("separated_path", "")) and cached.get("vocal_path") and file_exists(cached.get("vocal_path", "")):
            update_task(
                task_id,
                status="done",
                progress=100,
                message="separate cached",
                result={
                    "bvid": bvid,
                    "title": cached.get("title", payload.get("title", "")),
                    "audio_path": cached.get("audio_path", ""),
                    "video_path": cached.get("video_path", ""),
                    "cover_path": cached.get("cover_path", ""),
                    "accompaniment_path": cached.get("separated_path", ""),
                    "vocal_path": cached.get("vocal_path", ""),
                    "from_cache": True,
                },
            )
            return

        extract_payload = {
            "bvid": bvid,
            "arcurl": payload.get("arcurl", ""),
            "title": payload.get("title", ""),
            "pic": payload.get("pic", ""),
        }
        extract_task_id = create_task("extract_internal", extract_payload)
        internal_task_ids.append(extract_task_id)
        update_task(task_id, progress=8, message="extract start")
        worker_extract(extract_task_id, extract_payload)
        extract_task = get_task(extract_task_id) or {}
        if extract_task.get("status") != "done":
            raise RuntimeError(extract_task.get("error") or "extract failed")

        extract_result = extract_task.get("result") or {}
        audio_path = extract_result.get("audio_path", "")
        if not audio_path:
            raise RuntimeError("extract audio missing")

        separate_payload = {
            "bvid": bvid,
            "title": extract_result.get("title", payload.get("title", "")),
            "audio_path": audio_path,
        }
        separate_task_id = create_task("separate_internal", separate_payload)
        internal_task_ids.append(separate_task_id)
        update_task(task_id, progress=62, message="separate start")
        worker_separate(separate_task_id, separate_payload)
        separate_task = get_task(separate_task_id) or {}
        if separate_task.get("status") != "done":
            raise RuntimeError(separate_task.get("error") or "separate failed")

        separate_result = separate_task.get("result") or {}
        update_task(
            task_id,
            status="done",
            progress=100,
            message="bvid separate done",
            result={
                "bvid": bvid,
                "title": extract_result.get("title", payload.get("title", "")),
                "audio_path": audio_path,
                "video_path": extract_result.get("video_path", ""),
                "cover_path": extract_result.get("cover_path", ""),
                "accompaniment_path": separate_result.get("output_audio", ""),
                "vocal_path": separate_result.get("vocal_audio", ""),
                "from_cache": False,
            },
        )
    except Exception as exc:
        update_task(task_id, status="failed", progress=100, message="bvid separate failed", error=str(exc))


def start_task(task_id: str, target, payload: dict):
    thread = threading.Thread(target=target, args=(task_id, payload), daemon=True)
    thread.start()


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/desktop")
def desktop_page():
    return render_template("index.html")


@app.route("/tv")
def tv_page():
    room = request.args.get("room", "ktv001")
    return render_template("tv.html", room_id=room)


@app.route("/mobile")
def mobile_page():
    room = request.args.get("room", "ktv001")
    return render_template("mobile.html", room_id=room, page_name="search")


@app.route("/mobile/search")
def mobile_search_page():
    room = request.args.get("room", "ktv001")
    return render_template("mobile.html", room_id=room, page_name="search")


@app.route("/mobile/player")
def mobile_player_page():
    room = request.args.get("room", "ktv001")
    return redirect(f"/mobile/search?room={room}")


@app.route("/mobile/queue")
def mobile_queue_page():
    room = request.args.get("room", "ktv001")
    return render_template("mobile.html", room_id=room, page_name="queue")


@app.route("/mobile/library")
def mobile_library_page():
    room = request.args.get("room", "ktv001")
    return render_template("mobile.html", room_id=room, page_name="library")


@app.route("/api/search")
def api_search():
    keyword = request.args.get("keyword", "")
    page = request.args.get("page", 1, type=int)
    if not keyword:
        return jsonify({"success": False, "message": "missing keyword"})

    result = bili_search.search_videos(keyword, page)
    if not result or "result" not in result:
        return jsonify({"success": False, "message": "search failed"})

    videos = []
    for item in result.get("result", []):
        if item.get("type") == "video":
            videos.append(
                {
                    "id": item.get("bvid") or item.get("aid"),
                    "bvid": item.get("bvid", ""),
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "pic": f"https:{item.get('pic', '')}" if item.get("pic") else "",
                    "duration": item.get("duration", "0:00"),
                    "author": item.get("author", ""),
                    "play": item.get("play", 0),
                    "video_review": item.get("video_review", 0),
                    "favorites": item.get("favorites", 0),
                    "pubdate": item.get("pubdate", 0),
                    "arcurl": item.get("arcurl", ""),
                    "badgepay": item.get("badgepay", False),
                    "like": item.get("like", 0),
                }
            )

    return jsonify(
        {
            "success": True,
            "keyword": keyword,
            "current_page": page,
            "total_pages": result.get("numPages", 1),
            "total_results": result.get("numResults", 0),
            "videos": videos,
            "page_size": result.get("pagesize", 20),
        }
    )


@app.route("/api/room/state")
def api_room_state():
    room_id = request.args.get("room", "ktv001")
    return jsonify({"success": True, "room": room_state_payload(room_id)})


@app.route("/api/server-info")
def api_server_info():
    room_id = request.args.get("room", "ktv001")
    join = mobile_join_url(room_id)
    ping = join.split("/mobile?")[0] + "/api/ping"
    return jsonify(
        {
            "success": True,
            "room": room_id,
            "mobile_join_url": join,
            "ping_url": ping,
            "lan_ip": detect_lan_ip(),
            "host": request.host,
        }
    )


@app.route("/api/qr")
def api_qr():
    text = request.args.get("text", "").strip()
    if not text:
        return "", 400

    with QR_CACHE_LOCK:
        cached = QR_CACHE.get(text)
    if cached:
        resp = make_response(cached)
        resp.headers.set("Content-Type", "image/png")
        resp.headers.set("Cache-Control", "public, max-age=86400")
        return resp

    img_bytes = b""
    try:
        qurl = "https://api.qrserver.com/v1/create-qr-code/"
        r = requests.get(qurl, params={"size": "512x512", "data": text}, timeout=15)
        if r.status_code == 200 and r.content:
            img_bytes = r.content
    except Exception:
        img_bytes = b""

    if not img_bytes:
        img_bytes = fallback_qr_image(text)

    with QR_CACHE_LOCK:
        QR_CACHE[text] = img_bytes
    resp = make_response(img_bytes)
    resp.headers.set("Content-Type", "image/png")
    resp.headers.set("Cache-Control", "public, max-age=86400")
    return resp


@app.route("/api/ping")
def api_ping():
    return jsonify({"success": True, "ts": now_ms(), "host": request.host})


@app.route("/api/room/queue/add", methods=["POST"])
def api_room_queue_add():
    payload = request.get_json(silent=True) or {}
    room_id = payload.get("room", "ktv001")
    song = payload.get("song") or {}
    if not isinstance(song, dict):
        return jsonify({"success": False, "message": "invalid song payload"}), 400

    room = get_room(room_id)
    key = room_song_key(song)
    with ROOMS_LOCK:
        queue = room["queue"]
        for item in queue:
            if item.get("song_key") == key:
                room["updated_at"] = now_ms()
                return jsonify({"success": True, "room": room_state_payload(room_id), "duplicate": True})

        wrapped = {**song}
        wrapped["song_key"] = key
        wrapped["added_at"] = now_ms()
        queue.append(wrapped)
        if room["current_idx"] < 0:
            room["current_idx"] = 0
        room["updated_at"] = now_ms()
    return jsonify({"success": True, "room": room_state_payload(room_id)})


@app.route("/api/room/queue/remove", methods=["POST"])
def api_room_queue_remove():
    payload = request.get_json(silent=True) or {}
    room_id = payload.get("room", "ktv001")
    song_key = payload.get("song_key", "")
    idx = payload.get("idx", None)

    room = get_room(room_id)
    with ROOMS_LOCK:
        queue = room["queue"]
        remove_idx = -1
        if isinstance(idx, int) and 0 <= idx < len(queue):
            remove_idx = idx
        elif song_key:
            for i, item in enumerate(queue):
                if item.get("song_key") == song_key:
                    remove_idx = i
                    break
        if remove_idx < 0:
            return jsonify({"success": False, "message": "song not found"}), 404

        queue.pop(remove_idx)
        cur = room["current_idx"]
        if not queue:
            room["current_idx"] = -1
        elif remove_idx < cur:
            room["current_idx"] = max(cur - 1, 0)
        elif remove_idx == cur:
            room["current_idx"] = min(cur, len(queue) - 1)
        room["updated_at"] = now_ms()
    return jsonify({"success": True, "room": room_state_payload(room_id)})


@app.route("/api/room/queue/pin", methods=["POST"])
def api_room_queue_pin():
    payload = request.get_json(silent=True) or {}
    room_id = payload.get("room", "ktv001")
    song_key = payload.get("song_key", "")
    room = get_room(room_id)
    with ROOMS_LOCK:
        queue = room["queue"]
        idx = -1
        for i, item in enumerate(queue):
            if item.get("song_key") == song_key:
                idx = i
                break
        if idx <= 0:
            return jsonify({"success": True, "room": room_state_payload(room_id)})
        item = queue.pop(idx)
        queue.insert(0, item)
        room["current_idx"] = 0 if room["current_idx"] >= 0 else -1
        room["updated_at"] = now_ms()
    return jsonify({"success": True, "room": room_state_payload(room_id)})


@app.route("/api/room/play", methods=["POST"])
def api_room_play():
    payload = request.get_json(silent=True) or {}
    room_id = payload.get("room", "ktv001")
    idx = int(payload.get("idx", 0))
    room = get_room(room_id)
    with ROOMS_LOCK:
        queue = room["queue"]
        if not queue:
            room["current_idx"] = -1
        else:
            room["current_idx"] = min(max(idx, 0), len(queue) - 1)
        room["updated_at"] = now_ms()
    return jsonify({"success": True, "room": room_state_payload(room_id)})


@app.route("/api/room/next", methods=["POST"])
def api_room_next():
    payload = request.get_json(silent=True) or {}
    room_id = payload.get("room", "ktv001")
    room = get_room(room_id)
    with ROOMS_LOCK:
        queue = room["queue"]
        if queue:
            queue.pop(0)
        room["current_idx"] = 0 if queue else -1
        room["updated_at"] = now_ms()
    return jsonify({"success": True, "room": room_state_payload(room_id)})


@app.route("/api/room/mode", methods=["POST"])
def api_room_mode():
    payload = request.get_json(silent=True) or {}
    room_id = payload.get("room", "ktv001")
    mode = payload.get("mode", "accomp")
    if mode not in ["original", "accomp"]:
        return jsonify({"success": False, "message": "invalid mode"}), 400
    room = get_room(room_id)
    with ROOMS_LOCK:
        room["mode"] = mode
        room["updated_at"] = now_ms()
    return jsonify({"success": True, "room": room_state_payload(room_id)})


@app.route("/api/room/volume", methods=["POST"])
def api_room_volume():
    payload = request.get_json(silent=True) or {}
    room_id = payload.get("room", "ktv001")
    room = get_room(room_id)
    acc = float(payload.get("accomp_volume", room.get("accomp_volume", 1.0)))
    voc = float(payload.get("vocal_volume", room.get("vocal_volume", 0.0)))
    with ROOMS_LOCK:
        room["accomp_volume"] = max(0.0, min(1.0, acc))
        room["vocal_volume"] = max(0.0, min(1.0, voc))
        room["updated_at"] = now_ms()
    return jsonify({"success": True, "room": room_state_payload(room_id)})


@app.route("/api/extract", methods=["POST"])
def api_extract():
    payload = request.get_json(silent=True) or {}
    bvid = payload.get("bvid") or parse_bvid(payload.get("arcurl", ""))
    if not bvid:
        return jsonify({"success": False, "message": "missing bvid"}), 400
    payload["bvid"] = bvid
    task_id = create_task("extract", payload)
    start_task(task_id, worker_extract, payload)
    return jsonify({"success": True, "task_id": task_id})


@app.route("/api/separate-from-bvid", methods=["POST"])
def api_separate_from_bvid():
    payload = request.get_json(silent=True) or {}
    bvid = payload.get("bvid") or parse_bvid(payload.get("arcurl", ""))
    if not bvid:
        return jsonify({"success": False, "message": "missing bvid"}), 400
    payload["bvid"] = bvid
    task_id = create_task("separate_from_bvid", payload)
    start_task(task_id, worker_separate_from_bvid, payload)
    return jsonify({"success": True, "task_id": task_id})


@app.route("/api/separate", methods=["POST"])
def api_separate():
    payload = request.get_json(silent=True) or {}
    if not payload.get("audio_path"):
        return jsonify({"success": False, "message": "missing audio_path"}), 400
    task_id = create_task("separate", payload)
    start_task(task_id, worker_separate, payload)
    return jsonify({"success": True, "task_id": task_id})


@app.route("/api/task/<task_id>")
def api_task(task_id: str):
    task = get_task(task_id)
    if not task:
        return jsonify({"success": False, "message": "task not found"}), 404
    return jsonify({"success": True, "task": task})


@app.route("/api/result/<task_id>/meta")
def api_result_meta(task_id: str):
    task = get_task(task_id)
    if not task:
        return jsonify({"success": False, "message": "task not found"}), 404
    if task.get("status") != "done":
        return jsonify({"success": False, "message": "task not done"}), 409
    return jsonify({"success": True, "result": task.get("result") or {}})


@app.route("/api/result/<task_id>/<kind>")
def api_result_file(task_id: str, kind: str):
    task = get_task(task_id)
    if not task:
        return jsonify({"success": False, "message": "task not found"}), 404
    if task.get("status") != "done":
        return jsonify({"success": False, "message": "task not done"}), 409
    result = task.get("result") or {}
    field_map = {
        "accompaniment": "accompaniment_path",
        "vocal": "vocal_path",
        "audio": "audio_path",
        "video": "video_path",
    }
    field = field_map.get(kind)
    if not field:
        return jsonify({"success": False, "message": "invalid result kind"}), 400
    raw_path = result.get(field, "")
    if not raw_path:
        return jsonify({"success": False, "message": "result file missing"}), 404
    try:
        path = resolve_allowed_file(raw_path)
        return send_file(path, as_attachment=False, conditional=True)
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 404


@app.route("/api/local-file")
def api_local_file():
    raw_path = request.args.get("path", "")
    try:
        path = resolve_allowed_file(raw_path)
        return send_file(path, as_attachment=False, conditional=True)
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 404


@app.route("/api/local-library")
def api_local_library():
    entries = scan_local_library()
    return jsonify({"success": True, "count": len(entries), "items": entries})


@app.route("/api/local-delete", methods=["POST"])
def api_local_delete():
    payload = request.get_json(silent=True) or {}
    try:
        result = delete_media_item(payload)
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 400


@app.route("/proxy/image")
def proxy_image():
    url = request.args.get("url", "")
    if not url:
        return "", 404
    try:
        if url.startswith("//"):
            url = f"https:{url}"
        elif not url.startswith("http"):
            url = f"https://{url}"
        response = bili_search.session.get(url, timeout=15)
        if response.status_code != 200:
            return "", 404
        img_response = make_response(response.content)
        img_response.headers.set("Content-Type", response.headers.get("Content-Type", "image/jpeg"))
        img_response.headers.set("Cache-Control", "public, max-age=86400")
        return img_response
    except Exception:
        return "", 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass(frozen=True)
class Base64Result:
    base64: str
    mime_type: str
    data_url: str
    length_bytes: int
    source: str


def _infer_mime_from_path(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def _infer_mime_from_headers(
    content_type: Optional[str], fallback: str = "application/octet-stream"
) -> str:
    if content_type:
        return content_type.split(";")[0].strip()
    return fallback


def _to_data_url(b64: str, mime: str) -> str:
    return f"data:{mime};base64,{b64}"


def _encode_bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def file_to_base64(file_path: str, mime_type: Optional[str] = None) -> Base64Result:
    with open(file_path, "rb") as f:
        data = f.read()
    mime = mime_type or _infer_mime_from_path(file_path)
    b64 = _encode_bytes_to_b64(data)
    return Base64Result(
        base64=b64,
        mime_type=mime,
        data_url=_to_data_url(b64, mime),
        length_bytes=len(data),
        source=f"file:{file_path}",
    )


def url_to_base64(
    url: str, mime_type: Optional[str] = None, timeout: float = 30.0
) -> Base64Result:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.content
    content_type = resp.headers.get("Content-Type")
    mime = mime_type or _infer_mime_from_headers(
        content_type, fallback=_infer_mime_from_path(url)
    )
    b64 = _encode_bytes_to_b64(data)
    return Base64Result(
        base64=b64,
        mime_type=mime,
        data_url=_to_data_url(b64, mime),
        length_bytes=len(data),
        source=f"url:{url}",
    )


def bytes_to_base64(
    data: bytes, mime_type: str = "application/octet-stream", source: str = "bytes"
) -> Base64Result:
    b64 = _encode_bytes_to_b64(data)
    return Base64Result(
        base64=b64,
        mime_type=mime_type,
        data_url=_to_data_url(b64, mime_type),
        length_bytes=len(data),
        source=source,
    )

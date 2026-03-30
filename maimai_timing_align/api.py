from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

try:
    from .models import AlignConfig
except ImportError:  # pragma: no cover
    from models import AlignConfig


@dataclass(slots=True)
class RemoteAlignResult:
    anchor_a_sec: float
    anchor_b_sec: float
    offset_sec: float
    start_a_sec: float
    start_b_sec: float
    overlap_duration_sec: float
    confidence: float
    method: str
    warnings: list[str]


class OtogeAlignClient:
    def __init__(self, config: AlignConfig):
        self.base_url = config.otoge_base_url.rstrip("/")
        self.token = (config.otoge_developer_token or "").strip()
        self.timeout = float(config.otoge_timeout_sec)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.token:
            headers["x-developer-token"] = self.token
        return headers

    def align_audio(self, audio_a_path: Path, audio_b_path: Path, config: AlignConfig) -> RemoteAlignResult:
        endpoint = f"{self.base_url}/audalign/audalign/align"
        data = {
            "sample_rate": str(int(config.audio_sr)),
            "hop_length": str(int(config.audio_hop_length)),
            "n_fft": str(int(config.audio_n_fft)),
            "search_range_sec": str(float(config.audio_search_range_sec)),
            "min_overlap_sec": str(float(config.audio_min_overlap_sec)),
            "confidence_floor": str(float(config.audio_confidence_floor)),
            "max_duration_sec": str(float(config.audio_max_duration_sec)),
        }

        with (
            audio_a_path.open("rb") as fa,
            audio_b_path.open("rb") as fb,
            httpx.Client(timeout=self.timeout, follow_redirects=True) as client,
        ):
            files = {
                "audio_a": (audio_a_path.name, fa, "audio/wav"),
                "audio_b": (audio_b_path.name, fb, "audio/wav"),
            }
            resp = client.post(endpoint, headers=self._headers(), data=data, files=files)

        if resp.status_code >= 400:
            body = resp.text
            raise RuntimeError(f"otoge-service 对齐请求失败: HTTP {resp.status_code}, {body[:500]}")

        payload = resp.json()
        if not isinstance(payload, dict):
            raise RuntimeError("otoge-service 响应格式错误：根对象不是 JSON 对象")

        if int(payload.get("code", -1)) != 200:
            raise RuntimeError(
                f"otoge-service 对齐失败: code={payload.get('code')} msg={payload.get('message')}"
            )

        data_payload = payload.get("data")
        if not isinstance(data_payload, dict):
            raise RuntimeError("otoge-service 响应格式错误：data 字段缺失或不是对象")

        return RemoteAlignResult(
            anchor_a_sec=float(data_payload.get("anchor_a_sec", 0.0)),
            anchor_b_sec=float(data_payload.get("anchor_b_sec", 0.0)),
            offset_sec=float(data_payload.get("offset_sec", 0.0)),
            start_a_sec=float(data_payload.get("start_a_sec", 0.0)),
            start_b_sec=float(data_payload.get("start_b_sec", 0.0)),
            overlap_duration_sec=float(data_payload.get("overlap_duration_sec", 0.0)),
            confidence=float(data_payload.get("confidence", 0.0)),
            method=str(data_payload.get("method") or "audio_remote"),
            warnings=[str(x) for x in (data_payload.get("warnings") or [])],
        )

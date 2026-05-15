from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


MAX_PACKETS = 128
NET_TIMESTEPS = 24
PKT_FEAT_DIM = 16
NET_FEAT_DIM = 8
APP_EMB_DIM = 256
NET_EMB_DIM = 128
FLOW_EMB_DIM = 256


SERVICE_LABELS = [
    "streaming",
    "gaming",
    "conferencing",
    "bulk_transfer",
    "browsing",
    "xr_interactive",
    "iot_security",
    "unknown",
]


DEFAULT_APP_TO_SERVICE = {
    "youtube": "streaming",
    "youtube_live": "streaming",
    "netflix": "streaming",
    "prime_video": "streaming",
    "amazon_prime": "streaming",
    "twitch": "streaming",
    "video_streaming": "streaming",
    "streaming": "streaming",
    "valorant": "gaming",
    "league_of_legends": "gaming",
    "teamfight_tactics": "gaming",
    "roblox": "gaming",
    "minecraft": "gaming",
    "cloud_gaming": "gaming",
    "gaming": "gaming",
    "zoom": "conferencing",
    "teams": "conferencing",
    "google_meet": "conferencing",
    "discord": "conferencing",
    "voip": "conferencing",
    "conferencing": "conferencing",
    "download": "bulk_transfer",
    "file_download": "bulk_transfer",
    "ftp": "bulk_transfer",
    "bulk": "bulk_transfer",
    "bulk_transfer": "bulk_transfer",
    "web": "browsing",
    "web_browsing": "browsing",
    "browsing": "browsing",
    "vr": "xr_interactive",
    "ar": "xr_interactive",
    "xr": "xr_interactive",
    "metaverse": "xr_interactive",
    "cloud_vr": "xr_interactive",
    "vr_video": "xr_interactive",
    "xr_interactive": "xr_interactive",
    "benign": "iot_security",
    "ddos": "iot_security",
    "dos": "iot_security",
    "recon": "iot_security",
    "web_attack": "iot_security",
    "bruteforce": "iot_security",
    "brute_force": "iot_security",
    "spoofing": "iot_security",
    "mirai": "iot_security",
}


APP_ALIASES = {
    "yt": "youtube",
    "youtube live": "youtube_live",
    "amazon prime": "prime_video",
    "prime": "prime_video",
    "lol": "league_of_legends",
    "tft": "teamfight_tactics",
    "ms teams": "teams",
    "microsoft teams": "teams",
    "meet": "google_meet",
    "http download": "file_download",
    "http_download": "file_download",
    "web browsing": "web_browsing",
    "cloudvr": "cloud_vr",
    "cloud vr": "cloud_vr",
    "vr video": "vr_video",
    "brute force": "brute_force",
}


CONDITION_LABELS = [
    "good",
    "moderate",
    "degraded",
    "bad",
    "unknown",
]


def normalize_label(value: object) -> str:
    text = str(value or "unknown").strip().lower()
    text = text.replace("/", "_").replace("-", "_").replace(" ", "_")
    text = text.replace("__", "_")
    return APP_ALIASES.get(text, text)


def infer_service(app_or_label: object) -> str:
    token = normalize_label(app_or_label)
    if token in SERVICE_LABELS:
        return token
    if token in DEFAULT_APP_TO_SERVICE:
        return DEFAULT_APP_TO_SERVICE[token]
    for key, service in DEFAULT_APP_TO_SERVICE.items():
        if key in token:
            return service
    return "unknown"


@dataclass
class FlowConXConfig:

    max_packets: int = MAX_PACKETS
    net_timesteps: int = NET_TIMESTEPS
    pkt_feat_dim: int = PKT_FEAT_DIM
    net_feat_dim: int = NET_FEAT_DIM
    app_emb_dim: int = APP_EMB_DIM
    net_emb_dim: int = NET_EMB_DIM
    flow_emb_dim: int = FLOW_EMB_DIM
    services: List[str] = field(default_factory=lambda: list(SERVICE_LABELS))
    conditions: List[str] = field(default_factory=lambda: list(CONDITION_LABELS))

    @property
    def service_to_id(self) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(self.services)}

    @property
    def condition_to_id(self) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(self.conditions)}
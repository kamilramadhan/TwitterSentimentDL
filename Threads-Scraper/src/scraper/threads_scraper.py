from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from .utils.error_handler import retry
from .utils.logger import get_logger
from .utils.proxy_manager import ProxyManager

logger = get_logger(__name__)


class ThreadsScraper:
    """Fetch Threads posts either from live public pages or offline fixtures."""

    def __init__(self, settings: Dict[str, Any], config_dir: Path, data_dir: Path):
        self.settings = settings or {}
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        self.base_url = str(self.settings.get("base_url", "https://www.threads.net")).rstrip("/")
        self.timeout = int(self.settings.get("timeout", 15))

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": self.base_url,
            }
        )

        cookie = os.getenv("THREADS_COOKIE") or str(self.settings.get("cookie", "")).strip()
        if cookie:
            self.session.headers["Cookie"] = cookie

        proxies_path = self.data_dir / "raw" / "proxies.json"
        legacy_proxies_path = self.data_dir / "raw" / "proxies"
        if legacy_proxies_path.exists() and not proxies_path.exists():
            proxies_path = legacy_proxies_path
        self.proxy_manager = ProxyManager(proxies_path=proxies_path) if bool(self.settings.get("use_proxies", False)) else None
        self._profile_cache: Dict[str, Dict[str, Any]] = {}

    def fetch_user_threads(self, username: str, limit: int = 50) -> List[Dict[str, Any]]:
        safe_limit = max(1, int(limit))

        if bool(self.settings.get("use_offline", False)):
            return self._load_offline_items(username=username, limit=safe_limit)

        try:
            items = self._fetch_user_threads_from_profile_api(username=username, limit=safe_limit)
            if items:
                self._persist_raw(username=username, items=items)
                return items
            logger.warning("No items returned from live profile API for @%s.", username)
        except Exception as exc:
            logger.warning("Live profile API fetch failed for @%s (%s). Trying HTML fallback.", username, exc)

        try:
            html = self._fetch_profile_html(username=username)
            items = self._extract_items_from_html(html=html, username=username)
            if items:
                limited = items[:safe_limit]
                self._persist_raw(username=username, items=limited)
                return limited
            logger.warning("No online items parsed for @%s.", username)
        except Exception as exc:
            logger.warning("Online fetch failed for @%s (%s).", username, exc)

        if bool(self.settings.get("allow_offline_fallback", False)):
            logger.warning("Using offline fallback for @%s.", username)
            return self._load_offline_items(username=username, limit=safe_limit)

        return []

    @retry((requests.RequestException,), tries=3, delay=1.0, backoff=2.0)
    def _fetch_profile_html(self, username: str) -> str:
        url = f"{self.base_url}/@{username}"
        proxies = self.proxy_manager.get_proxy() if self.proxy_manager else None
        response = self.session.get(url, timeout=self.timeout, proxies=proxies)
        response.raise_for_status()
        return response.text

    @retry((requests.RequestException,), tries=3, delay=1.0, backoff=2.0)
    def _fetch_user_threads_from_profile_api(self, username: str, limit: int) -> List[Dict[str, Any]]:
        user = self._fetch_profile_user_payload(username=username)
        return self._extract_items_from_profile_user(user=user, default_username=username, limit=limit)

    def fetch_related_usernames(self, username: str, limit: int = 25) -> List[str]:
        safe_limit = max(0, int(limit))
        if safe_limit == 0:
            return []

        try:
            user = self._profile_cache.get(username) or self._fetch_profile_user_payload(username=username)
        except Exception as exc:
            logger.warning("Failed to fetch related usernames for @%s (%s).", username, exc)
            return []

        edges = ((user.get("edge_related_profiles") or {}).get("edges") or [])
        if not isinstance(edges, list):
            return []

        related: List[str] = []
        seen = set()
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            node = edge.get("node")
            if not isinstance(node, dict):
                continue

            candidate = str(node.get("username") or "").strip().lower()
            if not candidate or candidate == username.lower() or candidate in seen:
                continue

            seen.add(candidate)
            related.append(candidate)
            if len(related) >= safe_limit:
                break

        return related

    @retry((requests.RequestException,), tries=3, delay=1.0, backoff=2.0)
    def _fetch_profile_user_payload(self, username: str) -> Dict[str, Any]:
        cached = self._profile_cache.get(username)
        if isinstance(cached, dict):
            return cached

        url = f"{self.base_url}/api/v1/users/web_profile_info/"
        proxies = self.proxy_manager.get_proxy() if self.proxy_manager else None
        response = self.session.get(
            url,
            params={"username": username},
            headers={
                "Accept": "*/*",
                "X-IG-App-ID": "238260118697367",
                "Referer": f"{self.base_url}/@{username}",
            },
            timeout=self.timeout,
            proxies=proxies,
        )
        response.raise_for_status()

        body = response.text.strip()
        if body.startswith("for (;;);"):
            body = body[len("for (;;);") :]

        payload = json.loads(body)
        user = (payload.get("data") or {}).get("user") if isinstance(payload, dict) else None
        if not isinstance(user, dict):
            return {}

        self._profile_cache[username] = user
        return user

    def _extract_items_from_profile_user(self, user: Dict[str, Any], default_username: str, limit: int) -> List[Dict[str, Any]]:
        if not isinstance(user, dict):
            return []

        edges = ((user.get("edge_owner_to_timeline_media") or {}).get("edges") or [])
        if not isinstance(edges, list):
            return []

        items: List[Dict[str, Any]] = []
        safe_limit = max(1, int(limit))
        resolved_username = str(user.get("username") or default_username)

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            node = edge.get("node")
            if not isinstance(node, dict):
                continue

            post_id = node.get("id") or node.get("shortcode")
            shortcode = str(node.get("shortcode") or "")
            if not post_id:
                continue

            caption_edges = ((node.get("edge_media_to_caption") or {}).get("edges") or [])
            caption = ""
            if isinstance(caption_edges, list) and caption_edges:
                first_caption = caption_edges[0]
                if isinstance(first_caption, dict):
                    caption = str(((first_caption.get("node") or {}).get("text") or "")).strip()
            if not caption:
                caption = str(node.get("accessibility_caption") or "").strip()
            if not caption:
                continue

            like_count = (
                ((node.get("edge_liked_by") or {}).get("count"))
                or ((node.get("edge_media_preview_like") or {}).get("count"))
                or 0
            )
            reply_count = ((node.get("edge_media_to_comment") or {}).get("count")) or 0
            created_at = node.get("taken_at_timestamp")
            post_url = f"{self.base_url}/@{resolved_username}/post/{shortcode or post_id}"

            items.append(
                {
                    "id": str(post_id),
                    "username": resolved_username,
                    "text": caption,
                    "like_count": int(like_count),
                    "reply_count": int(reply_count),
                    "repost_count": 0,
                    "created_at": created_at,
                    "url": post_url,
                }
            )

            if len(items) >= safe_limit:
                break

        return items

    def _extract_items_from_html(self, html: str, username: str) -> List[Dict[str, Any]]:
        match = re.search(
            r'<script[^>]*id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
            html,
            flags=re.DOTALL,
        )
        if not match:
            return []

        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return []

        results: List[Dict[str, Any]] = []
        seen_ids = set()

        for node in self._walk_dicts(payload):
            item = self._coerce_item(node, default_username=username)
            if not item:
                continue
            if item["id"] in seen_ids:
                continue
            seen_ids.add(item["id"])
            results.append(item)

        return results

    def _walk_dicts(self, value: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from self._walk_dicts(child)
        elif isinstance(value, list):
            for item in value:
                yield from self._walk_dicts(item)

    def _coerce_item(self, raw: Dict[str, Any], default_username: str) -> Optional[Dict[str, Any]]:
        post = raw.get("post") if isinstance(raw.get("post"), dict) else raw

        pid = post.get("id") or post.get("pk") or post.get("code")
        text = self._extract_text(post)
        if not pid or not text:
            return None

        like_count = post.get("like_count") or post.get("likes") or 0
        reply_count = post.get("reply_count") or post.get("comment_count") or post.get("replies") or 0
        repost_count = post.get("repost_count") or post.get("reposts") or 0

        user_obj = post.get("user") if isinstance(post.get("user"), dict) else {}
        username = (
            post.get("username")
            or user_obj.get("username")
            or raw.get("username")
            or default_username
        )

        timestamp = post.get("created_at") or post.get("timestamp") or post.get("taken_at")
        created_at = self._to_iso_utc(timestamp)

        url = post.get("url") or f"{self.base_url}/@{username}/post/{pid}"

        return {
            "id": str(pid),
            "username": str(username),
            "text": str(text).strip(),
            "like_count": int(like_count or 0),
            "reply_count": int(reply_count or 0),
            "repost_count": int(repost_count or 0),
            "created_at": created_at,
            "url": str(url),
        }

    def _extract_text(self, post: Dict[str, Any]) -> str:
        direct_text = post.get("text")
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text

        caption = post.get("caption")
        if isinstance(caption, dict):
            caption_text = caption.get("text")
            if isinstance(caption_text, str) and caption_text.strip():
                return caption_text
        if isinstance(caption, str) and caption.strip():
            return caption

        content = post.get("content")
        if isinstance(content, dict):
            content_text = content.get("text")
            if isinstance(content_text, str) and content_text.strip():
                return content_text

        return ""

    def _to_iso_utc(self, value: Any) -> str:
        if value is None or value == "":
            return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        try:
            if isinstance(value, (int, float)):
                dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
                return dt.isoformat().replace("+00:00", "Z")

            text = str(value)
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except Exception:
            return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _load_offline_items(self, username: str, limit: int) -> List[Dict[str, Any]]:
        candidates = [
            self.data_dir / "raw" / f"{username}_threads.json",
            self.data_dir / "raw" / f"{username}.json",
            self.data_dir / "raw" / "sample_threads.json",
        ]

        for path in candidates:
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                rows = payload if isinstance(payload, list) else payload.get("items", [])
                if not isinstance(rows, list):
                    continue
                items = []
                for row in rows:
                    if isinstance(row, dict):
                        item = self._coerce_item(row, default_username=username)
                        if item:
                            items.append(item)
                if items:
                    return items[:limit]
            except Exception as exc:
                logger.warning("Failed loading offline file %s: %s", path, exc)

        generated = self._generate_synthetic_items(username=username, limit=limit)
        self._persist_raw(username=username, items=generated)
        return generated

    def _generate_synthetic_items(self, username: str, limit: int) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        items: List[Dict[str, Any]] = []
        for idx in range(limit):
            dt = now - timedelta(hours=idx)
            item_id = f"offline-{username}-{idx + 1}"
            items.append(
                {
                    "id": item_id,
                    "username": username,
                    "text": f"Sample offline Threads post #{idx + 1} for @{username}",
                    "like_count": max(0, 20 - idx),
                    "reply_count": max(0, 5 - (idx // 3)),
                    "repost_count": max(0, 3 - (idx // 6)),
                    "created_at": dt.isoformat().replace("+00:00", "Z"),
                    "url": f"{self.base_url}/@{username}/post/{item_id}",
                }
            )
        return items

    def _persist_raw(self, username: str, items: List[Dict[str, Any]]) -> None:
        raw_dir = self.data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        out_path = raw_dir / f"{username}_threads.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

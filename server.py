from __future__ import annotations

import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ai_engine import BossAI


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"
MAX_BODY_SIZE = 1_000_000
AI = BossAI()


class BossAIHandler(BaseHTTPRequestHandler):
    server_version = "BossAI/0.1"

    def do_HEAD(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            self._send_json({"error": "HEAD không hỗ trợ cho API."}, HTTPStatus.METHOD_NOT_ALLOWED)
            return
        self._serve_static(path, head_only=True)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send_json({"status": "ok"})
            return
        if path == "/api/status":
            if self._authorized():
                self._send_json(AI.status())
            return
        self._serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/") and not self._authorized():
            return
        try:
            payload = self._read_json()
            if path == "/api/chat":
                result = AI.respond(
                    str(payload.get("message", "")),
                    str(payload.get("session_id", "default")),
                )
                self._send_json(result)
                return
            if path == "/api/train":
                result = AI.train(
                    str(payload.get("instruction", "")),
                    str(payload.get("response", "")),
                    str(payload.get("intent", "")),
                )
                self._send_json({"message": "Đã học ví dụ mới.", "status": result})
                return
            self._send_json({"error": "Endpoint không tồn tại."}, HTTPStatus.NOT_FOUND)
        except ValueError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "JSON không hợp lệ."}, HTTPStatus.BAD_REQUEST)
        except OSError:
            self._send_json(
                {"error": "Không thể ghi dữ liệu runtime."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _authorized(self) -> bool:
        owner_token = os.environ.get("BOSS_AI_OWNER_TOKEN", "")
        if not owner_token:
            return True
        supplied = self.headers.get("X-Owner-Key", "")
        if supplied == owner_token:
            return True
        self._send_json(
            {"error": "Khóa chủ sở hữu không đúng."},
            HTTPStatus.UNAUTHORIZED,
        )
        return False

    def _read_json(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > MAX_BODY_SIZE:
            raise ValueError("Kích thước request không hợp lệ.")
        return json.loads(self.rfile.read(content_length).decode("utf-8"))

    def _serve_static(self, request_path: str, head_only: bool = False) -> None:
        relative = "index.html" if request_path == "/" else request_path.lstrip("/")
        candidate = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT.resolve() not in candidate.parents and candidate != STATIC_ROOT.resolve():
            self._send_json({"error": "Đường dẫn không hợp lệ."}, HTTPStatus.BAD_REQUEST)
            return
        if not candidate.is_file():
            candidate = STATIC_ROOT / "index.html"
        content_type, _ = mimetypes.guess_type(candidate.name)
        body = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            f"{content_type or 'application/octet-stream'}; charset=utf-8",
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; "
            "img-src 'self' data:; connect-src 'self'",
        )
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _send_json(
        self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[BossAI] {self.address_string()} - {format_string % args}")


def main() -> None:
    host = os.environ.get("BOSS_AI_HOST", "127.0.0.1")
    port = int(os.environ.get("BOSS_AI_PORT", "8000"))
    if host not in {"127.0.0.1", "localhost", "::1"} and not os.environ.get(
        "BOSS_AI_OWNER_TOKEN"
    ):
        raise SystemExit(
            "Từ chối bind ra mạng khi chưa đặt BOSS_AI_OWNER_TOKEN."
        )
    server = ThreadingHTTPServer((host, port), BossAIHandler)
    print(f"Sếp AI đang chạy tại http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐang dừng Sếp AI...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

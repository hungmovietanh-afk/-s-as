from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import RequestResponseEndpoint

from ai_engine import BossAI


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"
OWNER_HASH_PATH = ROOT / "data" / "owner_token.sha256"
MAX_BODY_SIZE = 1_000_000


def hash_owner_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def runtime_directory() -> Path:
    configured = os.environ.get("BOSS_AI_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    if os.environ.get("FLY_APP_NAME"):
        return Path("/data/runtime")
    return ROOT / "data" / "runtime"


def configured_owner_digest() -> str:
    owner_token = os.environ.get("BOSS_AI_OWNER_TOKEN", "")
    if owner_token:
        return hash_owner_token(owner_token)

    configured_digest = os.environ.get("BOSS_AI_OWNER_TOKEN_SHA256", "").strip().lower()
    if configured_digest:
        return _validate_digest(configured_digest)

    if os.environ.get("FLY_APP_NAME") and OWNER_HASH_PATH.is_file():
        return _validate_digest(OWNER_HASH_PATH.read_text(encoding="utf-8").strip().lower())
    return ""


def _validate_digest(digest: str) -> str:
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError("BOSS_AI_OWNER_TOKEN_SHA256 không hợp lệ.")
    return digest


def owner_token_matches(supplied: str, required_digest: str) -> bool:
    if not required_digest:
        return True
    return hmac.compare_digest(hash_owner_token(supplied), required_digest)


OWNER_TOKEN_DIGEST = configured_owner_digest()
AI = BossAI(runtime_dir=runtime_directory())
app = FastAPI(
    title="Sếp AI",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


class ChatPayload(BaseModel):
    message: str
    session_id: str = "default"


class TrainingPayload(BaseModel):
    instruction: str
    response: str
    intent: str


@app.middleware("http")
async def secure_responses(
    request: Request,
    call_next: RequestResponseEndpoint,
) -> Response:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_BODY_SIZE:
                return JSONResponse(
                    {"error": "Kích thước request không hợp lệ."},
                    status_code=413,
                )
        except ValueError:
            return JSONResponse(
                {"error": "Content-Length không hợp lệ."},
                status_code=400,
            )

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self'; script-src 'self'; "
        "img-src 'self' data:; connect-src 'self'"
    )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def require_owner(request: Request) -> None:
    supplied = request.headers.get("X-Owner-Key", "")
    if not owner_token_matches(supplied, OWNER_TOKEN_DIGEST):
        raise HTTPException(status_code=401, detail="Khóa chủ sở hữu không đúng.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status")
def status(request: Request) -> dict[str, object]:
    require_owner(request)
    return AI.status()


@app.post("/api/chat")
def chat(payload: ChatPayload, request: Request) -> dict[str, object]:
    require_owner(request)
    try:
        return AI.respond(payload.message, payload.session_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail="Không thể ghi dữ liệu runtime.",
        ) from error


@app.post("/api/train")
def train(payload: TrainingPayload, request: Request) -> dict[str, object]:
    require_owner(request)
    try:
        result = AI.train(payload.instruction, payload.response, payload.intent)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail="Không thể ghi dữ liệu runtime.",
        ) from error
    return {"message": "Đã học ví dụ mới.", "status": result}


app.mount("/", StaticFiles(directory=STATIC_ROOT, html=True), name="static")


def main() -> None:
    host = os.environ.get("BOSS_AI_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", os.environ.get("BOSS_AI_PORT", "8000")))
    if host not in {"127.0.0.1", "localhost", "::1"} and not OWNER_TOKEN_DIGEST:
        raise SystemExit("Từ chối bind ra mạng khi chưa đặt khóa chủ sở hữu.")
    print(f"Sếp AI đang chạy tại http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

from fastapi import FastAPI

from server import app as server_app


app = FastAPI(
    title="Sếp AI",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.mount("/", server_app)

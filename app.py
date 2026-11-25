from __future__ import annotations

import base64
import io
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from slop_engine import SLOP_MODES, VIDEO_MODES, SlopEngine

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Нейрослопер", version="1.1.0")
engine = SlopEngine()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> JSONResponse:
    modes = [
        {
            "id": m.id,
            "name": m.name,
            "description": m.description,
            "passes": m.passes,
            "use_caption": m.use_caption,
            "think_absurd": m.think_absurd,
        }
        for m in SLOP_MODES.values()
    ]
    video_modes = [
        {
            "id": vm.id,
            "name": vm.name,
            "description": vm.description,
            "engine": vm.engine,
            "default_frames": vm.default_frames,
            "default_fps": vm.default_fps,
        }
        for vm in VIDEO_MODES.values()
    ]
    return JSONResponse({"engine": engine.get_status(), "modes": modes, "video_modes": video_modes})


@app.post("/api/slopify")
async def slopify(
    file: UploadFile = File(...),
    mode: str = Form("total_bs"),
    slop_level: int = Form(65),
    custom_prompt: str = Form(""),
    seed: int = Form(-1),
    jpeg_fry: bool = Form(False),
    glitch_shift: bool = Form(False),
) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Пустой файл")

    try:
        image = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Не удалось распознать изображение") from exc

    result = engine.slopify(
        image=image,
        mode_id=mode,
        slop_level=slop_level,
        custom_prompt=custom_prompt,
        seed=seed if seed >= 0 else None,
        jpeg_fry=jpeg_fry,
        glitch_shift=glitch_shift,
    )

    buf = io.BytesIO()
    result["image"].save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return JSONResponse(
        {
            "image_data_url": f"data:image/png;base64,{b64}",
            "mode": result["mode"],
            "mode_name": result["mode_name"],
            "slop_level": result["slop_level"],
            "strength": result["strength"],
            "steps": result["steps"],
            "seed": result["seed"],
            "caption": result["caption"],
            "thoughts": result["thoughts"],
            "elapsed_ms": result["elapsed_ms"],
        }
    )


@app.post("/api/slopify-video")
async def slopify_video(
    file: UploadFile = File(...),
    video_mode: str = Form("morph_brainrot"),
    slop_level: int = Form(65),
    custom_prompt: str = Form(""),
    seed: int = Form(-1),
    num_frames: int = Form(20),
    fps: int = Form(10),
    jpeg_fry: bool = Form(False),
    glitch_shift: bool = Form(False),
) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Пустой файл")

    try:
        image = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Не удалось распознать изображение") from exc

    result = engine.slopify_video(
        image=image,
        video_mode_id=video_mode,
        slop_level=slop_level,
        custom_prompt=custom_prompt,
        seed=seed if seed >= 0 else None,
        num_frames=num_frames,
        fps=fps,
        jpeg_fry=jpeg_fry,
        glitch_shift=glitch_shift,
    )

    b64 = base64.b64encode(result["video_bytes"]).decode("ascii")
    return JSONResponse(
        {
            "video_data_url": f"data:video/mp4;base64,{b64}",
            "mode": result["mode"],
            "mode_name": result["mode_name"],
            "frames": result["frames"],
            "fps": result["fps"],
            "strength": result["strength"],
            "steps": result["steps"],
            "seed": result["seed"],
            "thoughts": result["thoughts"],
            "elapsed_ms": result["elapsed_ms"],
        }
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

"""
AutoSub-KZ — видеоларға автоматты қазақша субтитр жасайтын FastAPI сервері.

Негізгі ағын:
  1. /process эндпойнтіне видео URL жіберіледі
  2. Видео жүктеледі -> аудио бөлінеді -> Whisper транскрипциялайды
  3. Керек болса, мәтін қазақ тіліне аударылады
  4. .srt файлы құрастырылып, жауап ретінде қайтарылады
"""
import os
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.services.downloader import download_video
from app.services.audio_extractor import extract_audio
from app.services.transcriber import transcribe_audio
from app.services.translator import translate_segments
from app.services.subtitle_generator import generate_srt

app = FastAPI(
    title="AutoSub-KZ",
    description="Видеоларға автоматты қазақша субтитр жасайтын API (MVP)",
    version="0.1.0",
)

TEMP_DIR = "temp"

# Веб-интерфейс (static/index.html) осы арқылы қолжетімді болады: "/app"
app.mount("/app", StaticFiles(directory="static", html=True), name="static")


class ProcessRequest(BaseModel):
    video_url: str = Field(..., description="Видео сілтемесі (YouTube, т.б.)")
    source_language: str | None = Field(
        default=None,
        description="Бастапқы тіл коды, мысалы 'en', 'ru'. Бос қалдырса, автоанықталады.",
    )
    whisper_model: str = Field(
        default="base",
        description="Whisper моделінің өлшемі: tiny/base/small/medium/large",
    )


class ProcessResponse(BaseModel):
    task_id: str
    srt_download_url: str
    segment_count: int


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "AutoSub-KZ",
        "web_interface": "/app",
        "docs": "/docs",
    }


@app.post("/process", response_model=ProcessResponse)
def process_video(request: ProcessRequest):
    """
    Видео сілтемесін қабылдап, толық циклды орындайды:
    жүктеу -> аудио бөлу -> транскрипция -> аударма -> .srt жасау.

    ЕСКЕРТПЕ: MVP нұсқасында бұл синхронды орындалады (клиент күтеді).
    Өндірістік нұсқада бұл фондық тапсырма (Celery/RQ) ретінде істелуі керек,
    себебі ұзақ видеоларда бірнеше минутқа созылуы мүмкін.
    """
    task_id = str(uuid.uuid4())
    task_dir = os.path.join(TEMP_DIR, task_id)

    try:
        # 1. Видеоны жүктеу
        video_path = download_video(request.video_url, output_dir=task_dir)

        # 2. Аудионы бөліп алу
        audio_path = extract_audio(video_path, output_dir=task_dir)

        # 3. Whisper арқылы транскрипциялау
        segments = transcribe_audio(
            audio_path,
            model_size=request.whisper_model,
            source_language=request.source_language,
        )

        if not segments:
            raise HTTPException(status_code=422, detail="Видеодан мәтін анықталмады.")

        # 4. Қазақ тіліне аудару (бастапқы тіл қазақша болмаса)
        detected_language = request.source_language or "en"
        translated_segments = translate_segments(
            segments,
            source_language=detected_language,
            target_language="kk",
        )

        # 5. .srt файлын құрастыру
        srt_path = os.path.join(task_dir, "subtitles_kk.srt")
        generate_srt(translated_segments, srt_path)

        return ProcessResponse(
            task_id=task_id,
            srt_download_url=f"/download/{task_id}",
            segment_count=len(translated_segments),
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Өңдеу кезінде қате пайда болды: {exc}")


@app.get("/download/{task_id}")
def download_subtitles(task_id: str):
    """Дайын .srt файлын жүктеп алу эндпойнті."""
    srt_path = os.path.join(TEMP_DIR, task_id, "subtitles_kk.srt")
    if not os.path.exists(srt_path):
        raise HTTPException(status_code=404, detail="Файл табылмады немесе әлі дайын емес.")

    return FileResponse(
        path=srt_path,
        media_type="application/x-subrip",
        filename="subtitles_kk.srt",
    )

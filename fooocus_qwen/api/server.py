"""Приложение FastAPI и запуск сервера API (``--api``).

FastAPI и uvicorn — те же, на которых стоит Gradio: новых зависимостей нет.
Описание ``/docs`` (Swagger) и ``/openapi.json`` строит сам FastAPI.

Доступ — по Bearer-токену (``--api-token`` или ``FOOOCUS_QWEN_API_TOKEN``):
каждый маршрут, кроме ``/api/v1/health``, требует заголовок
``Authorization: Bearer <токен>``. Без токена API открыт — так удобно на
своей машине, а при прослушивании не только петли запуск предупреждает в
журнале. Токен сравнивается за постоянное время (``secrets.compare_digest``).
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .. import __version__, config
from . import images, operations, schemas
from .jobs import CANCELLED, DONE, Job, JobQueue

LOGGER = logging.getLogger(__name__)

PREFIX = operations.API_PREFIX
LOOPBACK = ("127.0.0.1", "localhost", "::1")

_bearer = HTTPBearer(auto_error=False, description="the token given by --api-token")


def _auth(token: str | None):
    """Зависимость маршрутов: пропускает всех без токена, с токеном — только его владельца."""

    def check(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
        if token is None:
            return
        given = credentials.credentials if credentials else ""
        if not secrets.compare_digest(given.encode("utf-8"), token.encode("utf-8")):
            raise HTTPException(401, "a valid Bearer token is required", headers={"WWW-Authenticate": "Bearer"})

    return check


def _finish(job: Job, queue: JobQueue, wait: bool) -> JSONResponse:
    """Ответ на работу: дождаться и отдать результат, либо 202 с номером задачи."""
    location = {"Location": f"{PREFIX}/jobs/{job.id}"}
    if not wait:
        return JSONResponse(job.describe(queue.position(job)), status_code=202, headers=location)
    job.done.wait()
    if job.status == DONE:
        return JSONResponse({"job": job.id, **(job.result or {})})
    status = 409 if job.status == CANCELLED else 500
    return JSONResponse({"job": job.id, "status": job.status, "detail": job.error or job.status},
                        status_code=status, headers=location)


def create_app(studio, token: str | None = None, queue: JobQueue | None = None) -> FastAPI:
    """Приложение API над ``Studio``. ``queue`` подменяют тесты."""
    queue = queue or JobQueue(lambda: studio.generator.interrupt() if studio.model_loaded else None)
    app = FastAPI(
        title="Fooocus-Qwen-Image-2.1 API",
        version=__version__,
        description="Generation, editing and outpainting with Qwen-Image-2.1. See docs/API.md.",
    )
    app.state.studio, app.state.queue = studio, queue

    @app.exception_handler(operations.RequestError)
    async def request_error(_request: Request, error: operations.RequestError):
        return JSONResponse({"detail": str(error)}, status_code=404 if error.not_found else 400)

    @app.exception_handler(operations.UpstreamError)
    async def upstream_error(_request: Request, error: operations.UpstreamError):
        return JSONResponse({"detail": str(error)}, status_code=502)

    @app.exception_handler(operations.EngineUnavailable)
    async def engine_unavailable(_request: Request, error: operations.EngineUnavailable):
        return JSONResponse({"detail": str(error)}, status_code=503)

    @app.get(f"{PREFIX}/health", tags=["service"])
    def health() -> Any:
        """Жив ли сервер — без токена: для проверок готовности и балансировщиков.

        После сбоя CUDA (``engine/faults.py``) сервер отвечает, но изображений
        уже не сделает до перезапуска: это 503, чтобы надзор его перезапустил.
        """
        if studio.gpu_fault is not None:
            return JSONResponse({"status": "gpu_fault", "detail": studio.gpu_fault}, status_code=503)
        return {"status": "ok"}

    api = APIRouter(prefix=PREFIX, dependencies=[Depends(_auth(token))])

    # --- сведения ---

    @api.get("/info", tags=["service"])
    def info() -> dict[str, Any]:
        return operations.info(studio)

    @api.get("/presets", tags=["catalogues"])
    def preset_list() -> list[dict[str, Any]]:
        return operations.preset_list()

    @api.get("/styles", tags=["catalogues"])
    def style_list() -> list[str]:
        return sorted(studio.catalogue)

    @api.get("/loras", tags=["catalogues"])
    def lora_list() -> list[dict[str, Any]]:
        return operations.lora_list(studio)

    # --- работы видеокарты ---

    @api.post("/generate", tags=["images"])
    def generate(body: schemas.GenerateRequest, wait: bool = True) -> JSONResponse:
        """Текст (и референсы) в изображение."""
        return _finish(queue.submit("generate", operations.prepare_generate(studio, body)), queue, wait)

    @api.post("/edit", tags=["images"])
    def edit(body: schemas.EditRequest, wait: bool = True) -> JSONResponse:
        """Правка промтом: по маске, по аннотации или всего кадра."""
        return _finish(queue.submit("edit", operations.prepare_edit(studio, body)), queue, wait)

    @api.post("/outpaint", tags=["images"])
    def outpaint(body: schemas.OutpaintRequest, wait: bool = True) -> JSONResponse:
        """Расширение кадра одной операцией (лора outpaint), без маски."""
        return _finish(queue.submit("outpaint", operations.prepare_outpaint(studio, body)), queue, wait)

    # --- видео (MiniMax H3 через FreeVideo) ---

    @api.post("/video", tags=["video"])
    def video(body: schemas.VideoGenerateRequest, wait: bool = True) -> JSONResponse:
        """Ролик: текст, кадры или референсы → MP4 в галерее; продление и доводка черновика."""
        work = operations.prepare_video(studio, body)
        return _finish(queue.submit("video", work, interrupt=studio.stop_video), queue, wait)

    @api.post("/video/storyboard/run", tags=["video"])
    def video_storyboard_run(body: schemas.StoryboardRunRequest, wait: bool = True) -> JSONResponse:
        """Все шоты подряд (с цепочкой кадров) и склейка с переходами."""
        work = operations.prepare_storyboard_run(studio, body)
        return _finish(queue.submit("video_storyboard", work, interrupt=studio.stop_video), queue, wait)

    @api.post("/video/boost", tags=["video"])
    def video_boost(body: schemas.VideoBoostRequest) -> dict[str, Any]:
        """Сценарий — в промт H3 языковой моделью."""
        return operations.video_boost(studio, body)

    @api.post("/video/storyboard", tags=["video"])
    def video_storyboard(body: schemas.StoryboardRequest) -> dict[str, Any]:
        """Бриф — в шоты языковой моделью."""
        return operations.storyboard_plan(studio, body)

    @api.get("/video/status", tags=["video"])
    def video_status() -> dict[str, Any]:
        """Готов ли движок видео; установка — только из командной строки (--setup-video)."""
        return operations.video_status(studio)

    @api.get("/video/loras", tags=["video"])
    def video_loras() -> list[str]:
        return operations.video_lora_list()

    @api.get("/jobs", tags=["jobs"])
    def job_list() -> list[dict[str, Any]]:
        return [job.describe(queue.position(job)) for job in queue.list()]

    @api.get("/jobs/{job_id}", tags=["jobs"])
    def job_get(job_id: str) -> dict[str, Any]:
        job = _job(job_id)
        return job.describe(queue.position(job))

    @api.delete("/jobs/{job_id}", tags=["jobs"])
    def job_cancel(job_id: str) -> dict[str, Any]:
        """Снять из очереди или прервать идущую (после текущего шага)."""
        job = _job(job_id)
        if not queue.cancel(job):
            raise HTTPException(409, f"the job is already {job.status}")
        return job.describe(queue.position(job))

    @api.post("/interrupt", tags=["jobs"])
    def interrupt() -> dict[str, Any]:
        """Прервать работу, которая идёт сейчас, — как кнопка «Стоп» (и картинку, и ролик)."""
        if studio.model_loaded:
            studio.generator.interrupt()
        studio.stop_video()
        return {"interrupted": studio.model_loaded}

    def _job(job_id: str) -> Job:
        job = queue.get(job_id)
        if job is None:
            raise HTTPException(404, f"no such job: {job_id}")
        return job

    # --- языковая модель ---

    @api.post("/boost", tags=["language model"])
    def boost(body: schemas.BoostRequest) -> dict[str, Any]:
        """Переписать промт языковой моделью (AI буст)."""
        return operations.boost(studio, body)

    @api.post("/describe", tags=["language model"])
    def describe(body: schemas.DescribeRequest) -> dict[str, Any]:
        """Описать картинку словами — промтом для правки или повтора."""
        return operations.describe(studio, body)

    @api.get("/llm", tags=["language model"])
    def llm_get(check: bool = False) -> dict[str, Any]:
        """Адрес языковой модели (токен не отдаётся); ``check`` — достучаться до неё."""
        return operations.llm_status(studio, check)

    @api.put("/llm", tags=["language model"])
    def llm_put(
        address: str = Body(..., embed=True),
        token: str | None = Body(None, embed=True),
        forget_token: bool = Body(False, embed=True),
    ) -> dict[str, Any]:
        """Сохранить адрес; пустой ``token`` оставляет прежний, ``forget_token`` удаляет."""
        return operations.store_llm(studio, address, token, forget_token)

    @api.get("/system-prompts", tags=["language model"])
    def system_prompt_list() -> list[str]:
        return list(operations.SYSTEM_PROMPTS)

    @api.get("/system-prompts/{name}", tags=["language model"])
    def system_prompt_get(name: str) -> dict[str, str]:
        return {"name": name, "text": operations.read_system_prompt(name)}

    @api.put("/system-prompts/{name}", tags=["language model"])
    def system_prompt_put(name: str, text: str = Body(..., embed=True)) -> dict[str, str]:
        operations.write_system_prompt(name, text)
        return {"name": name, "text": text}

    # --- позы ---

    @api.get("/poses", tags=["poses"])
    def pose_list() -> list[dict[str, Any]]:
        return operations.list_poses()

    @api.post("/poses", tags=["poses"])
    def pose_add(body: schemas.PoseRequest) -> dict[str, Any]:
        """Своя поза с фотографии (DWPose). Скелет сразу годится в ``references``."""
        return operations.add_pose(studio, body)

    @api.patch("/poses/{name}", tags=["poses"])
    def pose_update(
        name: str, title: str | None = Body(None, embed=True), schematic: bool | None = Body(None, embed=True),
    ) -> dict[str, Any]:
        return operations.update_pose(name, title, schematic)

    @api.delete("/poses/{name}", tags=["poses"])
    def pose_delete(name: str) -> dict[str, Any]:
        operations.delete_pose(name)
        return {"deleted": name}

    @api.post("/poses/{name}/cover", tags=["poses"])
    def pose_cover(name: str, wait: bool = True) -> JSONResponse:
        """Перерисовать обложку позы моделью."""
        return _finish(queue.submit("pose_cover", operations.prepare_pose_cover(studio, name)), queue, wait)

    @api.get("/poses/{name}/skeleton", tags=["poses"], response_class=FileResponse)
    def pose_skeleton(name: str):
        return FileResponse(operations.pose(name).skeleton, media_type="image/png")

    @api.get("/poses/{name}/preview", tags=["poses"], response_class=FileResponse)
    def pose_preview(name: str):
        return FileResponse(operations.pose(name).preview())

    # --- галерея и файлы ---

    @api.get("/gallery", tags=["gallery"])
    def gallery_list(limit: int = Query(60, ge=1, le=1000)) -> list[dict[str, Any]]:
        """Последние картинки и ролики, новые первыми, с параметрами (PNG или соседний JSON ролика)."""
        return operations.gallery_list(limit)

    @api.get("/files/{path:path}", tags=["gallery"], response_class=FileResponse)
    def file_get(path: str):
        try:
            found = images.output_path(path)
            return FileResponse(found, media_type=images.media_type(found))
        except images.ImageError as error:
            raise HTTPException(404 if error.not_found else 400, str(error)) from error

    # --- сохранённые промты ---

    @api.get("/prompts", tags=["prompts"])
    def prompt_list() -> list[str]:
        from ..prompting import library

        return library.list_prompts(config.PROMPT_DIR)

    @api.get("/prompts/{name}", tags=["prompts"])
    def prompt_get(name: str) -> dict[str, Any]:
        return operations.prompt_get(name)

    @api.put("/prompts/{name}", tags=["prompts"])
    def prompt_put(name: str, payload: schemas.PromptPayload) -> dict[str, Any]:
        operations.prompt_save(name, payload)
        return operations.prompt_get(name)

    @api.delete("/prompts/{name}", tags=["prompts"])
    def prompt_delete(name: str) -> dict[str, Any]:
        from ..prompting import library

        if not library.delete_prompt(name, config.PROMPT_DIR):
            raise HTTPException(404, f"no saved prompt {name!r}")
        return {"deleted": name}

    # --- настройки ---

    @api.get("/settings", tags=["settings"])
    def settings_get() -> dict[str, Any]:
        return operations.settings_state(studio)

    @api.patch("/settings", tags=["settings"])
    def settings_patch(body: schemas.SettingsPatch, wait: bool = True) -> JSONResponse:
        """Точность, SageAttention, профиль памяти. Смена точности докачивает веса и перезагружает модель."""
        return _finish(queue.submit("settings", operations.prepare_settings(studio, body)), queue, wait)

    @api.post("/model/load", tags=["settings"])
    def model_load() -> dict[str, Any]:
        """Начать загрузку модели в фоне, не дожидаясь первой генерации."""
        studio.preload_in_background()
        return {"model_loaded": studio.model_loaded}

    @api.post("/model/unload", tags=["settings"])
    def model_unload() -> dict[str, Any]:
        """Выгрузить модель и освободить видеопамять (после текущей работы)."""
        studio.unload()
        return {"model_loaded": studio.model_loaded}

    app.include_router(api)
    return app


def run(cfg: config.AppConfig) -> None:
    """Поднимает только API — без интерфейса — и держит поток до Ctrl+C."""
    import uvicorn

    from ..ui.state import Studio

    if cfg.api_token is None and cfg.host not in LOOPBACK:
        LOGGER.warning(
            "The API listens on %s without a token: anyone on the network can use it. "
            "Set --api-token (or FOOOCUS_QWEN_API_TOKEN), or listen on 127.0.0.1.", cfg.host,
        )
    studio = Studio(cfg)
    app = create_app(studio, cfg.api_token)
    shown = cfg.host if cfg.host not in ("0.0.0.0", "::", "") else "127.0.0.1"
    LOGGER.info("API at http://%s:%s%s, interactive docs at http://%s:%s/docs",
                shown, cfg.port, PREFIX, shown, cfg.port)
    if cfg.preload:
        studio.preload_in_background()
    # Журнал — наш (``logging_setup``): ``log_config=None`` не даёт uvicorn
    # переписать настройку корневого журнала своей.
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_config=None, access_log=cfg.verbose)


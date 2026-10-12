"""真实的世界 Web 版（v0.16 云端化第一阶段）：FastAPI 后端 + 静态单页。
运行:  pip install -r requirements-web.txt && python3 -m uvicorn web.server:app --host 0.0.0.0 --port 8000
物理内核直接复用 rw 包（纯 Python，无 Qt）；桌面版 app.py 不受影响。
任务模型：POST /api/sim/{module} 提交 → 后台线程跑 → GET /api/job/{id} 轮询 → 结果（文字 + SVG 图）。"""
import os, time, traceback, uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .modules import MODULES
from .jobstore import JobStore
from rw.version import VERSION

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="真实的世界 Web 版", version=VERSION)
_pool = ThreadPoolExecutor(max_workers=2)     # 仿真吃 CPU，限制并发防雪崩
STORE = JobStore()                            # v0.22 任务表持久化（SQLite）
_interrupted = STORE.mark_interrupted()       # 上次进程退出时没跑完的任务：标记 interrupted，参数保留可重提


class RunReq(BaseModel):
    params: dict = {}


@app.get("/api/modules")
def list_modules():
    return [{k: m[k] for k in ("key", "name", "desc", "fields")} for m in MODULES]


@app.post("/api/sim/{key}")
def start_sim(key: str, req: RunReq):
    mod = next((m for m in MODULES if m["key"] == key), None)
    if mod is None:
        raise HTTPException(404, "未知模块")
    jid = uuid.uuid4().hex[:12]
    STORE.create(jid, key, req.params)
    last = [0.0, -1.0]                                # (上次写入的进度, 上次写入时间)：进度落盘节流

    def on_progress(f):
        f = round(float(f), 3)
        if f >= 1.0 or f - last[0] >= 0.01 or time.time() - last[1] >= 1.0:
            STORE.set_progress(jid, f); last[0], last[1] = f, time.time()

    def job():
        try:
            res = mod["run"](req.params, on_progress)
            STORE.set_done(jid, res)
        except Exception:
            STORE.set_error(jid, traceback.format_exc(limit=6))

    _pool.submit(job)
    return {"job": jid}


@app.get("/api/job/{jid}")
def get_job(jid: str):
    j = STORE.get(jid)
    if j is None:
        raise HTTPException(404, "任务不存在")
    return j


@app.get("/api/meta")
def meta():
    return {"version": VERSION, "modules": len(MODULES), "interrupted_on_boot": _interrupted}


@app.get("/api/jobs")
def list_jobs():
    """最近任务（不含结果体，供历史面板）。"""
    return {"jobs": STORE.recent(20), "interrupted_on_boot": _interrupted}


app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")

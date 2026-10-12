"""真实的世界 Web 版（v0.16 云端化第一阶段）：FastAPI 后端 + 静态单页。
运行:  pip install -r requirements-web.txt && python3 -m uvicorn web.server:app --host 0.0.0.0 --port 8000
物理内核直接复用 rw 包（纯 Python，无 Qt）；桌面版 app.py 不受影响。
任务模型：POST /api/sim/{module} 提交 → 后台线程跑 → GET /api/job/{id} 轮询 → 结果（文字 + SVG 图）。"""
import os, traceback, uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .modules import MODULES

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="真实的世界 Web 版", version="0.16")
_pool = ThreadPoolExecutor(max_workers=2)     # 仿真吃 CPU，限制并发防雪崩
JOBS = {}     # v0.16 任务表在内存里；长时连续运行(100 h 级)是中期目标，届时换持久化任务表 + 断点续算


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
    JOBS[jid] = {"status": "running", "progress": 0.0, "module": key}

    def job():
        try:
            res = mod["run"](req.params, lambda f: JOBS[jid].__setitem__("progress", round(float(f), 3)))
            JOBS[jid].update(status="done", progress=1.0, result=res)
        except Exception:
            JOBS[jid].update(status="error", error=traceback.format_exc(limit=6))

    _pool.submit(job)
    return {"job": jid}


@app.get("/api/job/{jid}")
def get_job(jid: str):
    j = JOBS.get(jid)
    if j is None:
        raise HTTPException(404, "任务不存在")
    return j


app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")

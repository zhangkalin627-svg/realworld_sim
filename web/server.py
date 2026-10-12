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


def _validate_params(mod, params):
    """按模块表单 schema 校验/收口 API 参数：数字解析+按 min/max 收口；select 必须在选项内；
    textarea 必须是字符串；check 归一成 bool。未声明的参数原样透传（runner 自己处理）。"""
    out = dict(params)
    for fd in mod["fields"]:
        if "name" not in fd:
            continue
        name, typ, lab = fd["name"], fd.get("type", "number"), fd.get("label", fd["name"])
        if name not in out:
            continue
        v = out[name]
        if typ == "number":
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise HTTPException(400, f"参数「{lab}」需要数字，收到 {v!r}")
            if fd.get("min") is not None:
                v = max(float(fd["min"]), v)
            if fd.get("max") is not None:
                v = min(float(fd["max"]), v)
            out[name] = v
        elif typ == "select":
            opts = [o["v"] if isinstance(o, dict) else o for o in fd.get("options", [])]
            if opts and not any(str(v) == str(o) for o in opts):
                raise HTTPException(400, f"参数「{lab}」的取值 {v!r} 不在可选范围内")
        elif typ == "check":
            out[name] = bool(v)
        elif typ == "textarea" and not isinstance(v, str):
            raise HTTPException(400, f"参数「{lab}」需要文本")
    return out


@app.get("/api/modules")
def list_modules():
    return [{k: m[k] for k in ("key", "name", "desc", "fields")} for m in MODULES]


@app.post("/api/sim/{key}")
def start_sim(key: str, req: RunReq):
    mod = next((m for m in MODULES if m["key"] == key), None)
    if mod is None:
        raise HTTPException(404, "未知模块")
    params = _validate_params(mod, req.params)
    jid = uuid.uuid4().hex[:12]
    STORE.create(jid, key, params)
    last = [0.0, -1.0]                                # (上次写入的进度, 上次写入时间)：进度落盘节流

    def on_progress(f):
        f = round(float(f), 3)
        if f >= 1.0 or f - last[0] >= 0.01 or time.time() - last[1] >= 1.0:
            STORE.set_progress(jid, f); last[0], last[1] = f, time.time()

    def job():
        try:
            res = mod["run"](params, on_progress)
            STORE.set_done(jid, res)
        except Exception as e:
            traceback.print_exc()                       # 完整堆栈只留在服务器日志，不回给客户端
            STORE.set_error(jid, f"{type(e).__name__}: {e}")

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

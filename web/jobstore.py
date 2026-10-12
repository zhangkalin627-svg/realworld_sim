"""Web 版任务持久化 (v0.22)：SQLite 任务表，替代 v0.16 的内存字典。
- 任务（参数/进度/结果）落盘：服务器重启不丢历史，重启时把"running"标记为 interrupted（参数保留，可一键重提）
- 断点续算（跑到一半接着算）需要各模块 runner 支持检查点，仍在路线图
DB 路径：环境变量 RW_JOBS_DB，默认 ./realworld_jobs.db（uvicorn 启动目录）。"""
import json, os, sqlite3, threading, time


class JobStore:
    def __init__(self, path=None):
        self.path = path or os.environ.get("RW_JOBS_DB") or os.path.abspath("realworld_jobs.db")
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, module TEXT, params TEXT, status TEXT,
                progress REAL, result TEXT, error TEXT, created REAL, updated REAL)""")
            c.execute("PRAGMA journal_mode=WAL")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    def create(self, jid, module, params):
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)",
                      (jid, module, json.dumps(params, ensure_ascii=False), "running", 0.0, None, None,
                       time.time(), time.time()))
            c.execute("DELETE FROM jobs WHERE id NOT IN (SELECT id FROM jobs ORDER BY created DESC LIMIT 200)")

    def set_progress(self, jid, frac):
        with self._lock, self._conn() as c:
            c.execute("UPDATE jobs SET progress=?, updated=? WHERE id=?", (round(float(frac), 3), time.time(), jid))

    def set_done(self, jid, result):
        with self._lock, self._conn() as c:
            c.execute("UPDATE jobs SET status='done', progress=1.0, result=?, updated=? WHERE id=?",
                      (json.dumps(result, ensure_ascii=False), time.time(), jid))

    def set_error(self, jid, err):
        with self._lock, self._conn() as c:
            c.execute("UPDATE jobs SET status='error', error=?, updated=? WHERE id=?", (err, time.time(), jid))

    def get(self, jid):
        with self._lock, self._conn() as c:
            row = c.execute("SELECT id, module, params, status, progress, result, error, created, updated FROM jobs WHERE id=?",
                            (jid,)).fetchone()
        if row is None:
            return None
        return self._row(row)

    def recent(self, limit=20):
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT id, module, params, status, progress, result, error, created, updated "
                             "FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r, with_result=False) for r in rows]

    def mark_interrupted(self):
        """服务器重启时调用：running → interrupted（结果没有，参数保留）。返回受影响任务数。"""
        with self._lock, self._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()[0]
            c.execute("UPDATE jobs SET status='interrupted', error=?, updated=? WHERE status='running'",
                      ("服务器重启，任务中断；参数已保留，可重新提交", time.time()))
        return n

    @staticmethod
    def _row(row, with_result=True):
        jid, module, params, status, progress, result, error, created, updated = row
        j = {"id": jid, "module": module, "status": status, "progress": progress,
             "params": json.loads(params or "{}"), "created": created, "updated": updated}
        if with_result and result is not None:
            j["result"] = json.loads(result)
        if error:
            j["error"] = error
        return j

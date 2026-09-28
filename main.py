import asyncio
import json
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

import duckdb
import gradio as gr
import httpx
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

# --- Config ---
BUCKET_URL = (
    "https://huggingface.co/buckets/rehuuuu/TELEGRAM-COUNTRY-bucket"
    "/resolve/main/simple_all/simple_all.parquet"
)
LOCAL_PATH = "/tmp/simple_all.parquet"

PARALLELISM = int(os.environ.get("TC_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("TC_THREADS_PER_CONN", "2"))

CREDIT_INFO = {"developer": "rehuu", "channel": "@RehuSzr"}

_conns = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")
_downloaded = False
_download_lock = threading.Lock()


def _ensure_downloaded():
    """Download parquet from HF Bucket to /tmp once."""
    global _downloaded
    if _downloaded and os.path.exists(LOCAL_PATH):
        return
    with _download_lock:
        if _downloaded and os.path.exists(LOCAL_PATH):
            return
        print("[DB] Downloading parquet from HF Bucket...", flush=True)
        token = os.getenv("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            with httpx.stream(
                "GET",
                BUCKET_URL,
                headers=headers,
                follow_redirects=True,
                timeout=httpx.Timeout(600.0, connect=30.0),
            ) as r:
                r.raise_for_status()
                with open(LOCAL_PATH, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
        except Exception as e:
            print(f"[DB] Download failed: {e}", flush=True)
            if os.path.exists(LOCAL_PATH):
                os.remove(LOCAL_PATH)
            raise
        size_mb = os.path.getsize(LOCAL_PATH) / (1024 * 1024)
        print(f"[DB] Downloaded: {size_mb:.1f} MB", flush=True)
        _downloaded = True


def _new_conn():
    print("[DB] Creating DuckDB connection...", flush=True)
    _ensure_downloaded()
    con = duckdb.connect()
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute(f"SET threads = {THREADS_PER_CONN}")
    cnt = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{LOCAL_PATH}')"
    ).fetchone()[0]
    print(f"[DB] Ready. Rows: {cnt}", flush=True)
    return con


def _thread_id():
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns)
            _thread_local.id = tid
    return tid


def _get_conn():
    ident = _thread_id()
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]


def _lookup_user(user_id: str):
    uid = str(user_id).strip().replace("'", "''")
    sql = f"""
        SELECT user_id, phone, username, country_info
        FROM read_parquet('{LOCAL_PATH}')
        WHERE user_id = '{uid}'
        LIMIT 1
    """
    con = _get_conn()
    row = con.execute(sql).fetchone()
    if row is None:
        return None
    cols = ["user_id", "phone", "username", "country_info"]
    return dict(zip(cols, row))


# --- FastAPI ---
fastapi_app = FastAPI(title="Telegram Country API")


class BatchRequest(BaseModel):
    user_ids: list[str]


@fastapi_app.get("/")
def root():
    return {
        "app": "Telegram Country API",
        "usage": "GET /user/{user_id}",
        "credit": CREDIT_INFO,
    }


@fastapi_app.get("/health")
def health():
    return {
        "status": "ok",
        "downloaded": _downloaded,
        "file_exists": os.path.exists(LOCAL_PATH),
        "credit": CREDIT_INFO,
    }


@fastapi_app.get("/debug")
def debug():
    try:
        con = _get_conn()
        cnt = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{LOCAL_PATH}')"
        ).fetchone()[0]
        return {"ok": True, "total_rows": cnt, "credit": CREDIT_INFO}
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "credit": CREDIT_INFO,
        }


@fastapi_app.get("/user/{user_id}")
async def get_user(user_id: str):
    loop = asyncio.get_running_loop()
    try:
        data = await loop.run_in_executor(pool, _lookup_user, user_id)
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")
    if data is None:
        raise HTTPException(404, f"User ID '{user_id}' not found")
    return Response(
        content=json.dumps(
            {"success": True, **data, "credit": CREDIT_INFO},
            indent=2,
            ensure_ascii=False,
        ),
        media_type="application/json",
    )


@fastapi_app.post("/users/batch")
async def get_users_batch(req: BatchRequest):
    if not req.user_ids:
        raise HTTPException(400, "user_ids empty")
    if len(req.user_ids) > 100:
        raise HTTPException(400, "max 100")
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(pool, _lookup_user, uid) for uid in req.user_ids]
    rows = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for uid, data in zip(req.user_ids, rows):
        if isinstance(data, Exception):
            results.append({"user_id": uid, "found": False, "error": str(data)})
        elif data is None:
            results.append({"user_id": uid, "found": False})
        else:
            results.append({"user_id": data["user_id"], "found": True, **data})
    return {
        "total": len(req.user_ids),
        "found": sum(1 for r in results if r["found"]),
        "results": results,
        "credit": CREDIT_INFO,
    }


async def pinger():
    port = os.getenv("PORT", "7860")
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try:
                r = await client.get(url)
                print(f"[Pinger] {r.status_code}", flush=True)
            except Exception as e:
                print(f"[Pinger] {e}", flush=True)


@fastapi_app.on_event("startup")
async def startup_event():
    asyncio.create_task(pinger())
    loop = asyncio.get_running_loop()
    loop.run_in_executor(pool, _ensure_downloaded)


# --- Gradio UI ---
def ui_lookup(user_id: str) -> str:
    if not user_id or not user_id.strip():
        return "User ID daalo."
    try:
        data = _lookup_user(user_id.strip())
    except Exception as e:
        return f"Error: {e}"
    if data is None:
        return f"User ID: {user_id}\n\nNot found."
    return (
        f"User ID: {data['user_id']}\n\n"
        f"- phone: {data['phone']}\n"
        f"- username: {data['username']}\n"
        f"- country_info: {data['country_info']}"
    )


def build_ui():
    with gr.Blocks(title="Telegram Country API", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Telegram Country API")
        with gr.Row():
            uid_input = gr.Textbox(label="User ID", placeholder="723625545", scale=3)
            btn = gr.Button("Lookup", variant="primary", scale=1)
        output = gr.Markdown()
        btn.click(fn=ui_lookup, inputs=uid_input, outputs=output)
        uid_input.submit(fn=ui_lookup, inputs=uid_input, outputs=output)
    return demo


demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import duckdb
import gradio as gr
import httpx
from fastapi import FastAPI, HTTPException, Response
from huggingface_hub import HfFileSystem
from pydantic import BaseModel

# --- Config ---
BUCKET_PARQUET = os.environ.get(
    "TC_BUCKET_PARQUET",
    "hf://buckets/rehuuuu/TELEGRAM-COUNTRY-bucket/simple_all/simple_all.parquet",
)

PARALLELISM = int(os.environ.get("TC_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("TC_THREADS_PER_CONN", "2"))

CREDIT_INFO = {
    "developer": "rehuu",
    "channel": "@RehuSzr",
}

# --- DuckDB Connection Pool ---
_conns: list[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")


def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL httpfs; LOAD httpfs;")

    # Register HF filesystem so DuckDB can read hf://buckets/
    hffs = HfFileSystem(token=os.getenv("HF_TOKEN"))
    duckdb.register_filesystem(hffs)

    con.execute(f"SET threads = {THREADS_PER_CONN}")
    return con


def _thread_id() -> int:
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns)
            _thread_local.id = tid
    return tid


def _get_conn() -> duckdb.DuckDBPyConnection:
    ident = _thread_id()
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]


# --- Core Lookup ---
def _lookup_user(user_id: str):
    """Fetch a single user by user_id."""
    uid = str(user_id).strip().replace("'", "''")
    sql = f"""
        SELECT user_id, phone, username, country_info
        FROM read_parquet('{BUCKET_PARQUET}')
        WHERE user_id = '{uid}'
        LIMIT 1
    """
    con = _get_conn()
    row = con.execute(sql).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in con.description]
    return dict(zip(cols, row))


# --- FastAPI ---
fastapi_app = FastAPI(title="Telegram Country API")


class BatchRequest(BaseModel):
    user_ids: list[str]


@fastapi_app.get("/")
def root():
    return {
        "app": "Telegram Country API",
        "source": "hf://buckets/rehuuuu/TELEGRAM-COUNTRY-bucket",
        "usage": "GET /user/{user_id}",
        "developer": "rehuu | channel @RehuSzr",
    }


@fastapi_app.get("/health")
def health():
    return {"status": "ok", "credit": CREDIT_INFO}


@fastapi_app.get("/user/{user_id}")
async def get_user(user_id: str):
    """Primary endpoint - user_id to phone, username, country_info."""
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, _lookup_user, user_id)

    if data is None:
        raise HTTPException(status_code=404, detail=f"User ID '{user_id}' not found")

    result = {
        "success": True,
        "user_id": data["user_id"],
        "phone": data["phone"],
        "username": data["username"],
        "country_info": data["country_info"],
        "credit": CREDIT_INFO,
    }
    return Response(
        content=json.dumps(result, indent=2, ensure_ascii=False),
        media_type="application/json",
    )


@fastapi_app.post("/users/batch")
async def get_users_batch(req: BatchRequest):
    """Batch lookup - up to 100 user_ids."""
    if not req.user_ids:
        raise HTTPException(400, "user_ids must not be empty")
    if len(req.user_ids) > 100:
        raise HTTPException(400, "max 100 user_ids per batch")

    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(pool, _lookup_user, uid) for uid in req.user_ids]
    rows = await asyncio.gather(*tasks)

    results = []
    for uid, data in zip(req.user_ids, rows):
        if data is None:
            results.append({"user_id": uid, "found": False})
        else:
            results.append({
                "user_id": data["user_id"],
                "found": True,
                "phone": data["phone"],
                "username": data["username"],
                "country_info": data["country_info"],
            })

    return {
        "total": len(req.user_ids),
        "found": sum(1 for r in results if r["found"]),
        "results": results,
        "credit": CREDIT_INFO,
    }


# --- Pinger (Render free tier keep-alive) ---
async def pinger():
    port = os.getenv("PORT", "7860")
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try:
                r = await client.get(url)
                print(f"[Pinger] {r.status_code}")
            except Exception as e:
                print(f"[Pinger] {e}")


@fastapi_app.on_event("startup")
async def startup_event():
    asyncio.create_task(pinger())


# --- Gradio UI ---
def ui_lookup(user_id: str) -> str:
    if not user_id or not user_id.strip():
        return "User ID daalo."

    try:
        data = _lookup_user(user_id.strip())
    except Exception as e:
        return f"Error: {e}"

    if data is None:
        return (
            f"User ID: {user_id}\n\n"
            f"Not found.\n\n---\n\n"
            f"Developer: rehuu | Channel: @RehuSzr"
        )

    return (
        f"User ID: {data['user_id']}\n\n"
        f"- phone: {data['phone']}\n"
        f"- username: {data['username']}\n"
        f"- country_info: {data['country_info']}\n\n"
        f"---\n\nDeveloper: rehuu | Channel: @RehuSzr"
    )


def build_ui():
    with gr.Blocks(
        title="Telegram Country API",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# Telegram Country API")
        gr.Markdown("User ID daalo - phone, username, country milega")

        with gr.Row():
            uid_input = gr.Textbox(
                label="User ID",
                placeholder="e.g. 723625545",
                lines=1,
                scale=3,
            )
            btn = gr.Button("Lookup", variant="primary", scale=1)

        output = gr.Markdown(label="Result")

        btn.click(fn=ui_lookup, inputs=uid_input, outputs=output)
        uid_input.submit(fn=ui_lookup, inputs=uid_input, outputs=output)

        gr.Markdown("---")
        with gr.Accordion("API Info", open=False):
            gr.Markdown(
                "**Endpoints:**\n"
                "- `GET /user/{user_id}` - Single lookup\n"
                "- `POST /users/batch` - Batch (max 100)\n"
                "- `GET /health` - Health check\n"
                "- `GET /docs` - Swagger UI\n\n"
                "**Example:**\n"
                "```bash\n"
                "curl https://your-app.onrender.com/user/723625545\n"
                "```\n\n"
                "Developer: rehuu | Channel: @RehuSzr"
            )

        gr.Markdown("---\nDeveloper: rehuu | Channel: @RehuSzr")

    return demo


# --- Mount Gradio on FastAPI ---
demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

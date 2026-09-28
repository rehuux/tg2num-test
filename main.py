from fastapi import FastAPI, HTTPException
import duckdb
import os

app = FastAPI(title="Telegram Country API")

# Hugging Face Bucket parquet path
PARQUET_PATH = "hf://buckets/rehuuuu/TELEGRAM-COUNTRY-bucket/simple_all/simple_all.parquet"

# DuckDB connection (singleton)
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

# Optional: Hugging Face token for higher rate limits
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    try:
        con.execute(f"CREATE SECRET hf_secret (TYPE HUGGINGFACE, TOKEN '{HF_TOKEN}')")
        print("Hugging Face secret configured.")
    except Exception as e:
        print(f"HF secret setup failed: {e}")


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Use /get_user/{user_id} to fetch data",
        "example": "/get_user/723625545",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/get_user/{user_id}")
def get_user(user_id: str):
    try:
        row = con.execute(
            """
            SELECT user_id, phone, username, country_info
            FROM read_parquet(?)
            WHERE user_id = ?
            LIMIT 1
            """,
            [PARQUET_PATH, user_id],
        ).fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {e}")

    if row is None:
        raise HTTPException(status_code=404, detail=f"User ID '{user_id}' not found")

    return {
        "user_id": row[0],
        "phone": row[1],
        "username": row[2],
        "country_info": row[3],
    }


@app.on_event("startup")
def warmup():
    """Pre-warm DuckDB by fetching parquet metadata once."""
    try:
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{PARQUET_PATH}')"
        ).fetchone()
        print("DuckDB warmup complete.")
    except Exception as e:
        print(f"Warmup failed (non-fatal): {e}")

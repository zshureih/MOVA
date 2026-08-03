import json
import os
import time

import redis

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
STUB_DELAY_SECONDS = int(os.environ.get("STUB_DELAY_SECONDS", 15))

QUEUE_KEY = "video_queue:pending"

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def video_key(request_id: str) -> str:
    return f"video:{request_id}"


def process(request_id: str) -> None:
    raw = r.get(video_key(request_id))
    if raw is None:
        return
    record = json.loads(raw)

    record["status"] = "processing"
    r.set(video_key(request_id), json.dumps(record))

    # STUB: real inference (src/MOVA/main_bind.py) goes here once the model
    # is wired up. For now this just simulates processing time.
    time.sleep(STUB_DELAY_SECONDS)

    record["status"] = "completed"
    record["result"] = None
    record["note"] = "stub worker - no inference has run yet"
    r.set(video_key(request_id), json.dumps(record))


def main() -> None:
    print("MOVA stub worker started, waiting for jobs...")
    while True:
        item = r.blpop(QUEUE_KEY, timeout=5)
        if item is None:
            continue
        _, request_id = item
        print(f"Processing {request_id} (stub)")
        try:
            process(request_id)
        except Exception as exc:  # noqa: BLE001
            raw = r.get(video_key(request_id))
            if raw:
                record = json.loads(raw)
                record["status"] = "failed"
                record["error"] = str(exc)
                r.set(video_key(request_id), json.dumps(record))


if __name__ == "__main__":
    main()

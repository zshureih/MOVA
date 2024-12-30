from fastapi import FastAPI, Request, BackgroundTasks
import redis
import json
from typing import Protocol
import uuid

from main_bind import main_bind

app = FastAPI()
r = redis.Redis()

class RequestProtocol(Protocol):
    artist: str
    song_title: str
    prompt: str
    audio_strength: float
    alpha: float
    noise_level: float
    ddim_sampling_steps: int
    ddim_eta: float
    interpolation_steps: int
    height: int
    width: int
    seed: int

    def json() -> dict:
        pass

# get root
@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

# user submits a request to generate a video
@app.post("/video")
async def request_video(
    request: RequestProtocol,
    background_tasks: BackgroundTasks
):
    # Retrieve the request data
    data = await request.json()
    request_id = str(uuid.uuid4())

    # Add the request to the queue
    r.hset("video_queue", request_id, json.dumps({'data': data, 'status': 'queued'}))
    background_tasks.add_task(worker, request_id)

async def worker(request_id):
    # Retrieve the request data
    data = r.hget("video_queue", request_id)
    data = json.loads(data)
    
    # update the status to processing
    data['status'] = 'processing'
    data['id'] = request_id
    r.hset("video_queue", request_id, json.dumps(data))

    # run the model
    video_path = main_bind(data['data'])

    # update the status to completed and add the video path
    data['status'] = 'completed'
    data['result'] = video_path
    r.hset("video_queue", request_id, json.dumps(data))

@app.get("/video/{request_id}")
async def get_video(request_id: str):
    data = r.hget("video_queue", request_id)
    return json.loads(data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
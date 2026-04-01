# Dashboard Container — Usage Guide

All commands run from the **repo root** (`genie_sim/`).

---

## Build

> Build context must be the repo root (`.`), not `./dashboard`.

```bash
docker build -f dashboard/Dockerfile -t dashboard-test .
```

---

## Services

### 1. SAM3 Tracker

Real-time text-guided object segmentation using a webcam.

**Requirements:** GPU (RTX 40 series, 16 GB+ VRAM), camera at `/dev/video0`, X display.

```bash
docker run --rm -it \
    --gpus all \
    --device /dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    dashboard-test \
    python3 -m sam3_tracker.app \
        --text "coffee mug" \
        --camera 0 \
        --device cuda
```

**Multiple prompts:**
```bash
python3 -m sam3_tracker.app --text "person" "red mug" "laptop" --camera 0
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--text` | (required) | One or more object descriptions |
| `--camera` | `0` | Camera device index |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--config` | `sam3_tracker/configs/default.yaml` | Config file path |

**Keys while running:**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset detection |

---

### 2. OpenAI VLM Service (chaser_brain)

Lightweight REST server — accepts an image + task, calls OpenAI vision API.

**Port:** `8765`

**Setup `.env`** (one-time, already done):
```
source/chaser_brain/.env  ←  contains OPENAI_API_KEY, MODEL=gpt-4o
```

**Launch:**
```bash
docker run --rm -d --name chaser-brain -p 8765:8765 \
    -v $(pwd)/source/chaser_brain/.env:/app/chaser_brain/.env \
    dashboard-test \
    uvicorn chaser_brain.server:app --host 0.0.0.0 --port 8765
```

**Health check:**
```bash
curl -s http://localhost:8765/health | python3 -m json.tool
# {"status": "ok", "default_model": "gpt-4o"}
```

**Infer (image → VLM response):**
```bash
IMAGE_B64=$(base64 -w 0 /path/to/image.jpg)
curl -s -X POST http://localhost:8765/infer \
    -H "Content-Type: application/json" \
    -d "{\"task\": \"图中有什么物体？\", \"image\": \"$IMAGE_B64\"}" \
    | python3 -m json.tool
```

**Swagger UI:** http://localhost:8765/docs

**Stop:**
```bash
docker stop chaser-brain
```

---

### 3. Dashboard

Web dashboard (default container CMD).

**Port:** `8200`

```bash
docker run --rm -d --name dashboard -p 8200:8200 dashboard-test
```

Open: http://localhost:8200

---

## Interactive Shell

```bash
docker run --rm -it dashboard-test bash
```

Inside the container, modules are available at:
- `/app/sam3_tracker/` — SAM3 tracker wrapper
- `/app/chaser_brain/` — OpenAI VLM server
- `/app/dashboard/` — dashboard app

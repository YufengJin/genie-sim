import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Vision Reasoning Server")

DEFAULT_MODEL = os.environ.get("MODEL", "gpt-4o")


def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


class InferRequest(BaseModel):
    task: str
    image: str  # base64 encoded image (JPEG/PNG); or data URI like "data:image/png;base64,..."
    model: Optional[str] = None


class InferResponse(BaseModel):
    response: str
    model: str
    usage: dict


@app.get("/health")
def health():
    return {"status": "ok", "default_model": DEFAULT_MODEL}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    client = get_client()
    model = req.model or DEFAULT_MODEL

    # Accept raw base64 or full data URI
    if req.image.startswith("data:"):
        image_url = req.image
    else:
        image_url = f"data:image/jpeg;base64,{req.image}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": req.task},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    try:
        completion = client.chat.completions.create(model=model, messages=messages)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return InferResponse(
        response=completion.choices[0].message.content,
        model=completion.model,
        usage=completion.usage.model_dump(),
    )

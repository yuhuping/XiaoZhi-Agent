# XiaoZhi v1.1

XiaoZhi is a small FastAPI prototype for a child-friendly multimodal learning companion.

## What v1.1 does

- accepts text input, base64 image input, or image URL input
- identifies a simple topic or object
- returns a short child-friendly explanation
- asks one easy follow-up question
- exposes one health endpoint and one main demo endpoint

## Quick start

### Install

```powershell
pip install -r requirements.txt
```

### Run backend

From the project root `D:\XiaoZ`:

```powershell
python -m uvicorn app.main:app --reload
```

or

```powershell
.\scripts\run_backend.ps1
```

If your current directory is `D:\XiaoZ\app`, use:

```powershell
python -m uvicorn main:app --reload
```

If Windows blocks the reloader process in your environment, use:

```powershell
python -m uvicorn app.main:app
```

### Run tests

```powershell
pytest
```

## Environment

Copy `.env.example` values into your shell or environment manager.

- `MOCK_MODEL=true` keeps the demo runnable without an OpenAI API key.
- Set `MOCK_MODEL=false` and provide `OPENAI_API_KEY` to call the OpenAI Responses API.

## API

### Health check

`GET /health`

### Main demo endpoint

`POST /api/v1/chat/explain-and-ask`

Example request:

```json
{
  "text": "What is this apple?",
  "age_hint": "4-6"
}
```

Image URL example:

```json
{
  "text": "What is in this picture?",
  "image_url": "https://example.com/cat.png",
  "age_hint": "3-8"
}
```

Example response:

```json
{
  "topic": "apple",
  "explanation": "An apple is a fruit. It is often round, crunchy, and sweet.",
  "follow_up_question": "What color do you think an apple can be?",
  "metadata": {
    "source_mode": "mock",
    "confidence": "high",
    "safety_notes": "",
    "used_image": false
  }
}
```

## Notes

The OpenAI implementation uses the official Responses API with:

- `input_text` and `input_image` content items
- `data:` URLs for base64 image input
- `text.format.type=json_schema` for structured output

If the OpenAI request fails, the service falls back to the local mock path so the demo still works.

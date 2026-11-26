# Voice Chat

A local voice assistant: speak (or type), get a short spoken reply. FastAPI on the
backend, plain HTML/CSS/JS on the frontend, [Ollama](https://ollama.com) for the model.
No build step, no cloud services.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) running locally
- A browser with the Web Speech API. **Chrome or Edge** for voice input (Firefox does not
  implement speech recognition); any modern browser for typed input and spoken replies.

## Quick start

```bash
# 1. Get a model (the default config uses a tiny one)
ollama pull smollm:135m
#    For noticeably better answers: ollama pull llama3.2:3b, then set OLLAMA_MODEL

# 2. Install and run
pip install -r requirements.txt
python server.py
```

Open <http://127.0.0.1:8080>. The page and the API are served from the same origin, so
there is no second dev server and no CORS configuration.

## Configuration

Set any of these as environment variables (see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Where Ollama listens |
| `OLLAMA_MODEL` | `smollm:135m` | Model name |
| `HOST` / `PORT` | `127.0.0.1` / `8080` | Server bind address |
| `OLLAMA_TIMEOUT` | `120` | Seconds to wait for a reply |
| `CONTEXT_MESSAGES` | `12` | Recent turns kept in the prompt |
| `CONTEXT_CHAR_BUDGET` | `6000` | Character cap for that history |
| `MAX_REPLY_TOKENS` | `220` | Reply length cap |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` |

## How it works

1. The browser captures one utterance with `SpeechRecognition`, then stops the mic so it
   never transcribes the assistant's own voice.
2. The message is POSTed to `/api/chat`. The backend streams the model's reply as NDJSON:
   `{"type":"delta","text":"…"}` per chunk, then one `{"type":"done"}`.
3. The page renders tokens as they arrive and hands each finished sentence to a speech
   queue, so audio starts on the first sentence instead of waiting for the full reply.
4. Replies are spoken with the browser's `SpeechSynthesis`. There is no server-side audio;
   voice choice, language, and speed live in **Voice settings**.

If Ollama is down or the model is missing, the header shows a status pill and the page
explains the exact command to fix it.

**Keyboard:** `Enter` sends, `Shift+Enter` adds a line, `Esc` stops speaking.

## Tests

```bash
pip install pytest
pytest
```

The suite covers request validation, the unreachable-Ollama paths, and that only the
`static/` folder is exposed (backend source and `.env` are not downloadable).

## Troubleshooting

- **"Ollama isn't running"** — start it with `ollama serve`.
- **"Model not pulled"** — run the `ollama pull` command shown in the banner.
- **Mic button disabled / no permission** — use Chrome or Edge and allow microphone
  access; the app warns instead of silently failing, and typing still works.
- **No voices in the dropdown** — some browsers populate voices asynchronously; the list
  refreshes on `voiceschanged`. On Linux you may need `espeak-ng` installed.

## Notes on changes

This replaces an earlier version whose server crashed on Windows (`fcntl`), whose
"text-to-speech" wrote a placeholder `b"WAVDATA"` file nothing could play, whose mic kept
recording while the bot spoke (so it heard itself), and whose voice dropdown was usually
empty. The simulated TTS, the canned "let's get back on topic" interruptions, and the
invented `confidence_score` fields are gone rather than faked.

Still open, if you want to take it further: server-side neural voice output (e.g.
`edge-tts`) for consistent quality across browsers, and a wake-word / always-listening
mode.

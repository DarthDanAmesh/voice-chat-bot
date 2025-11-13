from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple
from enum import Enum
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import asyncio
import hashlib
import logging
import httpx
import re
import time
import uuid
import os
import fcntl
from collections import OrderedDict
from asyncio import Semaphore

# --- CONFIGURATION AND LOGGING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat_tts_app")

# Environment variables for configuration
TTS_WORKERS = int(os.getenv("TTS_WORKERS", "4"))
TTS_TIMEOUT = float(os.getenv("TTS_TIMEOUT", "30.0"))
TTS_OUTPUT_DIR = Path(os.getenv("TTS_OUTPUT_DIR", "/tmp/tts"))
MAX_CONCURRENT_TTS = int(os.getenv("MAX_CONCURRENT_TTS", "10"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "smollm:135m")

# Global variables
tts_executor: ProcessPoolExecutor | None = None
tts_semaphore = Semaphore(MAX_CONCURRENT_TTS)
tts_jobs: "JobStorage" = None  # Will be initialized in lifespan

OLLAMA_API_URL = "http://localhost:11434/api/generate"

# --- ENUMS AND MODELS

class ResponseQuality(str, Enum):
    CONCISE = "concise"
    DETAILED = "detailed"
    FACTUAL = "factual"

class Message(BaseModel):
    role: str
    content: str
    metadata: Optional[Dict] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    messages: List[Message]
    role: str
    preferred_style: Optional[ResponseQuality] = ResponseQuality.CONCISE
    continue_conversation: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    tts_response: str
    confidence_score: float
    source_type: str
    needs_verification: bool
    voice_suitable: bool

class TTSRequest(BaseModel):
    text: str
    immediate: bool = False

# --- JOB STORAGE WITH TTL AND SIZE LIMITS

class JobStorage:
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self._jobs = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
    
    def add(self, job_id: str, job_data: dict):
        # Cleanup before adding
        self._cleanup()
        if len(self._jobs) >= self.max_size:
            # Remove oldest
            self._jobs.popitem(last=False)
        self._jobs[job_id] = {"created": time.time(), **job_data}
    
    def get(self, job_id: str) -> Optional[dict]:
        job = self._jobs.get(job_id)
        if job and time.time() - job["created"] > self.ttl:
            del self._jobs[job_id]
            return None
        return job
    
    def _cleanup(self):
        now = time.time()
        expired = [jid for jid, job in self._jobs.items() 
                  if now - job["created"] > self.ttl]
        for jid in expired:
            del self._jobs[jid]

# --- TEXT & TTS HELPERS

def clean_response(text: str) -> str:
    text = re.sub(r'<break time=".*?"/>', '', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(!+)', '!', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s*([.,!?])', r'\1', text)
    text = re.sub(r'\b(um|uh|ah|like)\b', '', text, flags=re.IGNORECASE)
    if not text.endswith(('.', '!', '?')):
        text += '.'
    return text

def optimize_for_tts(text: str) -> str:
    num_map = {'0':'zero','1':'one','2':'two','3':'three','4':'four',
               '5':'five','6':'six','7':'seven','8':'eight','9':'nine'}
    for n, w in num_map.items():
        text = re.sub(r'\b'+n+r'\b', w, text)
    text = re.sub(r'([.!?])\s+', r'\1 <break time="0.5s"/> ', text)
    text = re.sub(r',\s+', ', <break time="0.3s"/> ', text)
    return text

# --- LIFESPAN / EXECUTOR MANAGEMENT

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_executor, tts_jobs
    
    # Initialize job storage
    tts_jobs = JobStorage()
    
    # Create TTS output directory
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, TTS_OUTPUT_DIR.mkdir, parents=True, exist_ok=True)
    
    # Start TTS executor
    tts_executor = ProcessPoolExecutor(max_workers=TTS_WORKERS)
    logger.info(f"Started {TTS_WORKERS} TTS worker processes")
    
    yield
    
    logger.info("Shutting down TTS workers...")
    tts_executor.shutdown(wait=False, cancel_futures=True)
    logger.info("TTS workers shut down successfully")

# --- TTS SYNTHESIS FUNCTIONS

def synthesize_tts(text: str, output_dir: Path = TTS_OUTPUT_DIR) -> str:
    """Simulated heavy TTS synthesis — replace with real model later."""
    output_dir.mkdir(parents=True, exist_ok=True)
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    file_path = output_dir / f"tts_{text_hash}.wav"
    temp_path = file_path.with_suffix(".tmp")

    if file_path.exists():
        return str(file_path)

    try:
        # Use file locking for cross-process coordination
        with open(temp_path, "wb") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            
            # Check again after acquiring lock
            if file_path.exists():
                return str(file_path)
                
            # Simulate heavy TTS inference
            time.sleep(1.2)
            f.write(b"WAVDATA")
        
        temp_path.rename(file_path)
        return str(file_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

async def synthesize_tts_async(text: str, timeout: float = TTS_TIMEOUT) -> str:
    if tts_executor is None:
        raise RuntimeError("TTS executor not initialized")
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(tts_executor, synthesize_tts, text),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"TTS synthesis timeout after {timeout}s for: {text[:40]}...")
        raise HTTPException(status_code=504, detail="TTS synthesis timeout")

async def safe_background_tts(text: str):
    """Safely run TTS in the background with proper error handling."""
    try:
        async with tts_semaphore:
            await synthesize_tts_async(text)
            logger.info(f"Background TTS completed for: {text[:40]}...")
    except Exception as e:
        logger.error(f"Background TTS failed for '{text[:40]}...': {e}")

# --- CHAT UTILITIES

def is_greeting(text: str) -> bool:
    greetings = {'hello','hi','hey','greetings','good morning','good afternoon','good evening'}
    return text.lower().strip('.,!?') in greetings

def is_simple_question(text: str) -> bool:
    patterns = [r"^what is",r"^where is",r"^who is",r"^when was",r"^how many",r"^which"]
    return any(re.match(p, text.lower()) for p in patterns)

def get_response_template(msg: str, is_greet: bool, is_simple: bool) -> dict:
    if is_greet:
        return {"prefix":"","suffix":"","temperature":0.1,"max_tokens":20}
    elif is_simple:
        return {"prefix":"","suffix":".","temperature":0.2,"max_tokens":40}
    else:
        return {"prefix":"","suffix":" Would you like to know more?",
                "temperature":0.4,"max_tokens":100}

def extract_topics(text: str, min_word_length: int = 4) -> set:
    stop_words = {'this','that','what','when','where','which','who','with','from',
                  'have','been','were','there','their','will','would','could','should',
                  'about','just','like'}
    words = {w.lower() for w in re.findall(r'\b[a-z]+\b', text.lower())
             if len(w)>=min_word_length}
    return words - stop_words

def check_topic_drift(prev_msgs: List[Message], curr_msg: str, threshold: float = 0.15):
    if len(prev_msgs)<2: return False,1.0
    
    # Cache topic extraction for efficiency
    prev_topics=set()
    for m in prev_msgs[-5:]: 
        prev_topics.update(extract_topics(m.content))
    
    curr_topics=extract_topics(curr_msg)
    if not curr_topics or not prev_topics: return False,0.5
    
    # Jaccard similarity: intersection / union
    intersection = len(curr_topics & prev_topics)
    union = len(curr_topics | prev_topics)
    similarity = intersection / union if union > 0 else 0.0
    
    return similarity < threshold, similarity

def build_context(messages: List[Message]) -> str:
    ctx=[]
    for m in messages[-4:-1]:
        if m.role.lower()!="system":
            role="User" if m.role.lower()=="user" else "Assistant"
            ctx.append(f"{role}: {m.content}")
    return "\n".join(ctx[-3:])

# --- FASTAPI APP

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- /api/chat ENDPOINT

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        latest = request.messages[-1].content
        is_greet = is_greeting(latest)
        is_simple = is_simple_question(latest)
        prev = request.messages[:-1] if len(request.messages)>1 else []
        drift, sim = check_topic_drift(prev, latest)

        if drift and not is_greet and len(prev)>=2:
            text="Let's focus back on the previous topic."
            cleaned=clean_response(text)
            return ChatResponse(response=cleaned,
                                tts_response=optimize_for_tts(cleaned),
                                confidence_score=0.8,
                                source_type="topic_correction",
                                needs_verification=False,
                                voice_suitable=True)

        if is_greet:
            text="Hi there!"
            cleaned=clean_response(text)
            return ChatResponse(response=cleaned,
                                tts_response=optimize_for_tts(cleaned),
                                confidence_score=0.9,
                                source_type="greeting",
                                needs_verification=False,
                                voice_suitable=True)

        template=get_response_template(latest,is_greet,is_simple)
        context=build_context(request.messages)
        prompt=f"{context}\nUser: {latest}" if context else f"User: {latest}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp=await client.post(OLLAMA_API_URL,json={
                "model":OLLAMA_MODEL,"prompt":prompt,"stream":False,
                "options":{"num_predict":template["max_tokens"],
                           "temperature":template["temperature"],
                           "top_p":0.1 if is_simple else 0.3}})
            if resp.status_code!=200:
                raise HTTPException(status_code=500,detail="LLM request failed")
            resp_text=resp.json().get("response","").strip()

        cleaned=clean_response(resp_text)
        final_txt=f"{template['prefix']}{cleaned}{template['suffix']}"
        final_tts=f"{template['prefix']}{optimize_for_tts(cleaned)}{template['suffix']}"
        
        # Fire off async TTS synthesis (background) with proper error handling
        asyncio.create_task(safe_background_tts(cleaned))
        
        return ChatResponse(response=final_txt,
                            tts_response=final_tts,
                            confidence_score=0.9 if is_simple else 0.7,
                            source_type="factual" if is_simple else "generated",
                            needs_verification=not is_simple,
                            voice_suitable=True)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}",exc_info=True)
        err="I apologize, but I encountered an error. Please try again."
        return ChatResponse(response=err,tts_response=err,
                            confidence_score=0.0,source_type="error",
                            needs_verification=True,voice_suitable=True)

# --- TTS JOB ENDPOINTS

@app.post("/api/tts")
async def generate_tts(request: TTSRequest):
    job_id=str(uuid.uuid4())
    tts_jobs.add(job_id, {"status":"pending","text":request.text})

    async def _run_tts():
        try:
            path=await synthesize_tts_async(request.text)
            tts_jobs.add(job_id, {"status":"completed","path":path})
        except Exception as e:
            tts_jobs.add(job_id, {"status":"failed","error":str(e)})
            logger.error(f"TTS job {job_id} failed: {e}",exc_info=True)

    if request.immediate:
        await _run_tts()
        return tts_jobs.get(job_id)
    else:
        asyncio.create_task(_run_tts())
        return {"job_id":job_id,"status":"queued"}

@app.get("/api/tts/{job_id}")
async def get_tts_status(job_id: str):
    job=tts_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404,detail="Job not found")
    return job

# --- HEALTH CHECK

@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient(timeout=1.0) as c:
            r=await c.get("http://localhost:11434/api/tags")
            if r.status_code==200:
                return {"status":"healthy","ollama":"connected","models":r.json()}
    except Exception:
        pass
    return {"status":"unhealthy","ollama":"disconnected"}

# --- ENTRY POINT

if __name__=="__main__":
    import uvicorn
    logger.info("Starting Chat+TTS FastAPI server")
    uvicorn.run(app,host="127.0.0.1",port=8080)
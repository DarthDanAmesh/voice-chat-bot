from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import httpx
import logging
import re
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseQuality(str, Enum):
    CONCISE = "concise"      # For simple, direct answers
    DETAILED = "detailed"    # For complex explanations
    FACTUAL = "factual"      # For verified facts only

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
    response: str  # Clean text response (no SSML)
    tts_response: str  # TTS-optimized response (with SSML)
    confidence_score: float
    source_type: str
    needs_verification: bool
    voice_suitable: bool

    
def clean_response(text: str) -> str:
    """Clean and optimize response for TTS and natural readability."""
    # Remove SSML tags (e.g., <break time="0.5s"/>)
    text = re.sub(r'<break time=".*?"/>', '', text)
    
    # Remove markdown and code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove italics
    
    # Remove numbered lists and extra spaces
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Improve sentence structure for TTS
    text = re.sub(r'(!+)', '!', text)  # Remove multiple exclamation marks
    text = re.sub(r'\.{2,}', '.', text)  # Remove ellipsis
    text = re.sub(r'\s*([.,!?])', r'\1', text)  # Fix spacing around punctuation
    
    # Remove unnecessary pauses or filler words
    text = re.sub(r'\b(um|uh|ah|like)\b', '', text, flags=re.IGNORECASE)
    
    # Ensure proper sentence ending
    if not text.endswith(('.', '!', '?')):
        text += '.'
        
    return text

def optimize_for_tts(text: str) -> str:
    """Additional optimizations for TTS output."""
    # Convert numbers to words for better speech
    number_words = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    
    for num, word in number_words.items():
        text = re.sub(r'\b' + num + r'\b', word, text)
    
    # Add slight pauses for better speech rhythm (SSML tags for TTS only)
    text = re.sub(r'([.!?])\s+', r'\1 <break time="0.5s"/> ', text)
    text = re.sub(r',\s+', ', <break time="0.3s"/> ', text)
    
    return text

def is_greeting(text: str) -> bool:
    """Check if the message is a greeting."""
    greetings = {'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'}
    return text.lower().strip('.,!?') in greetings

def is_simple_question(text: str) -> bool:
    """Check if the message is a simple question."""
    patterns = [
        r"^what is",
        r"^where is",
        r"^who is",
        r"^when was",
        r"^how many",
        r"^which"
    ]
    return any(re.match(pattern, text.lower()) for pattern in patterns)

def get_response_template(message: str) -> dict:
    """Get appropriate response template based on message type."""
    if is_greeting(message):
        return {
            "prefix": "",
            "suffix": "",
            "temperature": 0.1,
            "max_tokens": 20
        }
    elif is_simple_question(message):
        return {
            "prefix": "",
            "suffix": ".",
            "temperature": 0.2,
            "max_tokens": 40
        }
    else:
        return {
            "prefix": "",
            "suffix": " Would you like to know more?",
            "temperature": 0.4,
            "max_tokens": 100
        }

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_API_URL = "http://localhost:11434/api/generate"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        latest_message = request.messages[-1].content
        template = get_response_template(latest_message)
        
        ollama_request = {
            "model": "smollm:135m",
            "prompt": latest_message,
            "stream": False,
            "options": {
                "num_predict": template["max_tokens"],
                "temperature": template["temperature"],
                "top_p": 0.1 if is_simple_question(latest_message) else 0.3
            }
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(OLLAMA_API_URL, json=ollama_request)
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to get response from language model")
            
            response_text = response.json().get('response', '').strip()
            
            # For greetings, provide simple, friendly responses
            if is_greeting(latest_message):
                greetings = ["Hi there!", "Hello!", "Hi! How can I help you?", "Hello! How can I assist you?"]
                response_text = greetings[0]  # Use first greeting consistently
            
            # Clean the response for text output
            cleaned_response = clean_response(response_text)
            
            # Optimize the response for TTS (with SSML tags)
            tts_optimized = optimize_for_tts(cleaned_response)
            
            # Prepare final responses
            final_text_response = f"{template['prefix']}{cleaned_response}{template['suffix']}"
            final_tts_response = f"{template['prefix']}{tts_optimized}{template['suffix']}"
            
            return ChatResponse(
                response=final_text_response,  # Clean text response (no SSML)
                tts_response=final_tts_response,  # TTS-optimized response (with SSML)
                confidence_score=0.9 if is_greeting(latest_message) or is_simple_question(latest_message) else 0.7,
                source_type="greeting" if is_greeting(latest_message) else "factual" if is_simple_question(latest_message) else "generated",
                needs_verification=not (is_greeting(latest_message) or is_simple_question(latest_message)),
                voice_suitable=True
            )
            
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        error_response = "I apologize, but I encountered an error. Please try again."
        return ChatResponse(
            response=error_response,
            tts_response=error_response,  # No SSML needed for error responses
            confidence_score=0.0,
            source_type="error",
            needs_verification=True,
            voice_suitable=True
        )

@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                return {"status": "healthy", "ollama": "connected", "models": response.json()}
    except Exception:
        return {"status": "unhealthy", "ollama": "disconnected", "error": "Could not connect to Ollama service."}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="127.0.0.1", port=8080)
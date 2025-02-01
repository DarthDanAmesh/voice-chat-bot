from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx
import json
import logging
from fastapi.responses import JSONResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    role: str

class ChatResponse(BaseModel):
    response: str

OLLAMA_API_URL = "http://localhost:11434/api/generate"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request with {len(request.messages)} messages")

        # Format the conversation history
        formatted_prompt = ""
        for msg in request.messages[-3:]:  # Only use last 3 messages for context
            if msg.role == "user":
                formatted_prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                formatted_prompt += f"Assistant: {msg.content}\n"
        
        formatted_prompt += "Assistant:"

        logger.info(f"Formatted prompt: {formatted_prompt}")

        # Prepare Ollama request with shorter context
        ollama_request = {
            "model": "smollm:135m",
            "prompt": formatted_prompt,
            "stream": False,
            "context": [],  # Empty context to reduce processing time
            "options": {
                "num_predict": 100,  # Limit response length
                "temperature": 0.7
            }
        }

        # Make request to Ollama with shorter timeout
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                # First check if Ollama is responsive
                health_check = await client.get("http://localhost:11434/api/tags", timeout=2.0)
                if health_check.status_code != 200:
                    raise HTTPException(status_code=503, detail="Ollama service is not responding properly")

                logger.info("Sending request to Ollama")
                response = await client.post(
                    OLLAMA_API_URL,
                    json=ollama_request,
                )
                
                logger.info(f"Ollama response status: {response.status_code}")
                
                if response.status_code != 200:
                    error_detail = f"Ollama API error: {response.text}"
                    logger.error(error_detail)
                    raise HTTPException(status_code=500, detail=error_detail)
                
                response_data = response.json()
                assistant_response = response_data.get('response', '').strip()
                
                if not assistant_response:
                    raise HTTPException(status_code=500, detail="Empty response from Ollama")
                
                logger.info("Successfully got response from Ollama")
                return ChatResponse(response=assistant_response)

            except httpx.TimeoutException as e:
                logger.error(f"Timeout while connecting to Ollama: {str(e)}")
                raise HTTPException(
                    status_code=504,
                    detail="Request to Ollama timed out. The model might be busy or not responding."
                )
            
            except httpx.ConnectError as e:
                logger.error(f"Failed to connect to Ollama: {str(e)}")
                raise HTTPException(
                    status_code=503,
                    detail="Could not connect to Ollama. Please check if Ollama is running on port 11434."
                )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json()
                return {
                    "status": "healthy",
                    "ollama": "connected",
                    "models": models
                }
    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama": "disconnected",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="127.0.0.1", port=8080)
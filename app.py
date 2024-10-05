import os
from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_api_key = os.getenv("API_KEY")
model = os.getenv("LLM_MODEL")
token = os.getenv("TOKEN")

groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

system_prompt = 'You are a friendly conversational chatbot'
conversational_memory_length = 5
memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

class ChatRequest(BaseModel):
    question: str

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Parameter required."
        }
    )

@app.exception_handler(HTTPException)
async def bad_request_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 400:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Parameter cannot be blank."
            }
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail
        }
    )

async def verify_token(authorization: str = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Token required.")
    
    verify_token = authorization.split(" ")[1] if " " in authorization else authorization
    
    if verify_token != token:
        raise HTTPException(status_code=403, detail="Invalid token.")

@app.post("/chat")
async def chat(request: ChatRequest, authorization: str = Header(None)):
    await verify_token(authorization)
    user_question = request.question

    if user_question:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=False,
            memory=memory,
        )

        response = conversation.predict(human_input=user_question)
        return {
            "success": True,
            "message": response
        }

    raise HTTPException(status_code=400, detail="Parameter cannot be blank.")


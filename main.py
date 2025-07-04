# run -> uvicorn main:app --reload  -> in vsc and go to the local host shown on vsc terminal

from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import agent_executor

app = FastAPI()

class Query(BaseModel):
    message: str

@app.post("/chat")
async def chat(query: Query):
    try:
        result = await agent_executor.ainvoke({"query": query.message})
        return {"response": result["output"]}
    except Exception as e:
        return {"response": "Sorry, something went wrong.", "error": str(e)}


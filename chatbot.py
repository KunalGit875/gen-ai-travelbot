# main.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from tools.budget_tool import estimate_budget_tool, TravelModel, format_travel_summary
from tools.weather_tool import weather_forecast_tool
from tools.rag_tool import get_rag_tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain.memory import ConversationBufferMemory

load_dotenv()

# === Prompt and Agent Setup ===
def get_integrated_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.2)

    parser = PydanticOutputParser(pydantic_object=TravelModel)

    prompt = ChatPromptTemplate.from_messages([
        ("system", '''
You are a smart tourism assistant. Based on user's query, perform one or more of the following tasks:
- If the user asks tourism or privacy policy-related questions, use the RAG tool.
- If the user needs a travel plan or cost estimate, call the budget tool.
- If the user asks for weather/climate, call the weather tool.
Always be concise and helpful.
'''),
        ("system", "..."),
        ("placeholder", "{chat_history}"),
        ("user", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    tools = [
        estimate_budget_tool,
        weather_forecast_tool,
        get_rag_tool()
    ]

    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
    executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
    return executor

agent_executor = get_integrated_agent()

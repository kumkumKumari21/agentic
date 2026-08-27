# agentic
1...............................................................
!pip install groq -q
!pip install -U agno groq yfinance python-dotenv
import os

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools

os.environ["GROQ_API_KEY"] =""

agent = Agent(
    model=Groq(
        id="openai/gpt-oss-20b"
    ),
    
    tools=[
        YFinanceTools(
            enable_stock_price=True,
            enable_company_info=True,
            enable_analyst_recommendations=True,
            enable_company_news=True
        )
    ],


    instructions=[
        "Create a concise professional financial report.",
        "Use tables wherever useful.",
        "Include the current stock price.",
        "Include basic company information.",
        "Include important financial metrics.",
        "Include analyst recommendations.",
        "Include recent company news.",
        "Clearly separate factual data from analysis.",
        "Do not invent financial information.",
        "Keep the report concise and easy to understand."
    ],

    markdown=True
)

response = agent.run(
    "Generate a financial report on NVIDIA Corporation (NVDA). "
    "Include the current stock price, company information, "
    "important financial metrics, analyst recommendations, "
    "and recent company news."
)

print(response.content)

2.......................................................................
#b)	Planning design pattern
from groq import Groq

client = Groq(
    api_key=""
)

response = client.chat.completions.create(
    #model="llama-3.3-70b-versatile",
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role":"user",
            "content":"""
Goal:
Organize a college technical fest.

Create a detailed step-by-step plan.
"""
        }
    ]
)

print(response.choices[0].message.content)
#c)	REACT design pattern
from groq import Groq

client = Groq(
    api_key=""
)

response = client.chat.completions.create(
    #model="llama-3.3-70b-versatile",
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role":"user",
            "content":"""
Question:
What is 25 × 18?

Show:

Thought
Action
Observation
Final Answer
"""
        }
    ]
)

print(response.choices[0].message.content)

3.......................................................................................................................
!pip install -q langchain langchain-core langchain-groq langserve fastapi uvicorn sse_starlette
!pip install -q langserve
!pip install -q langchain-groq
!pip install -q nest_asyncio
!pip install -q langchain langchain-core langchain-groq langserve fastapi uvicorn sse_starlette nest_asyncio
!pip install -q langchain langchain-core langchain-google-genai langserve fastapi uvicorn sse_starlette nest_asyncio
#3.	How to convert the chain to web API using LANGSERVE framework
!pip install -q langchain-groq langserve fastapi uvicorn nest_asyncio

import os, threading, nest_asyncio, uvicorn
from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

os.environ["GROQ_API_KEY"] = ""

app = FastAPI()

llm = ChatGroq(model="openai/gpt-oss-20b")

prompt = ChatPromptTemplate.from_template(
    "You are an expert teacher. Explain clearly: {question}"
)

chain = prompt | llm | StrOutputParser()

add_routes(app, chain, path="/chat")

nest_asyncio.apply()

threading.Thread(
    target=lambda: uvicorn.run(
        app,
        host="127.0.0.1",
        port=8002
    ),
    daemon=True
).start()

print("Server: http://127.0.0.1:8002")
print("Playground: http://127.0.0.1:8002/chat/playground/")

4..................................................................................................
#4.	Build a Self-correcting Coding Assistant with LangChain

!pip install -q langchain-groq

import os, subprocess, tempfile
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = ""
llm = ChatGroq(model="openai/gpt-oss-20b")

task = "Calculate factorial of 5 using recursion."

code = """def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)

print(fact(5))"""

for i in range(2):
    print(f"\nAttempt {i+1}:\n{code}")
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name

    r = subprocess.run(["python", path], capture_output=True, text=True)

    if r.returncode == 0:
        print("\nOutput:", r.stdout)
        break

    print("\nError:", r.stderr.strip())

    code = llm.invoke(
        f"Fix this code. Return only code.\nCode:\n{code}\nError:\n{r.stderr}"
    ).content.replace("```python","").replace("```","").strip()
5........................................................................................................
#5.	Building a Finance Bot with LangGraph.
!pip install -q langgraph langchain langchain-core langchain-groq
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from IPython.display import Image, display
os.environ["GROQ_API_KEY"] = ""
llm = ChatGroq(model="openai/gpt-oss-20b")

class FinanceState(TypedDict):
    question: str
    stock: str
    price: float
    answer: str

def extract_stock(state: FinanceState):
    question = state["question"].upper()
    if "APPLE" in question or "AAPL" in question:
        stock = "AAPL"
    elif "TESLA" in question or "TSLA" in question:
        stock = "TSLA"
    elif "MICROSOFT" in question or "MSFT" in question:
        stock = "MSFT"
    else:
        stock = "UNKNOWN"
    return {"stock": stock}

def get_stock_price(state: FinanceState):
    fake_prices = {
        "AAPL": 195.50,
        "TSLA": 180.25,
        "MSFT": 420.10,
        "UNKNOWN": 0.0
    }
    price = fake_prices[state["stock"]]
    return {"price": price}

def generate_answer(state: FinanceState):
    prompt = f"""
    You are a finance assistant.
    User question: {state['question']}
    Stock symbol: {state['stock']}
    Stock price: {state['price']}
    Give a simple financial response.
    Do not give investment advice.
    """
    response = llm.invoke(prompt)
    return {"answer": response.content}

graph = StateGraph(FinanceState)
graph.add_node("extract_stock", extract_stock)
graph.add_node("get_stock_price", get_stock_price)
graph.add_node("generate_answer", generate_answer)
graph.add_edge(START, "extract_stock")
graph.add_edge("extract_stock", "get_stock_price")
graph.add_edge("get_stock_price", "generate_answer")
graph.add_edge("generate_answer", END)
finance_bot = graph.compile()
display(
    Image(finance_bot.get_graph().draw_mermaid_png())
)

result = finance_bot.invoke({
    "question": "What is the current price of MICROSOFT stock?"
})
print(result["answer"])

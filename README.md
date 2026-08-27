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
!pip install -q langgraph langchain-groq

import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from IPython.display import Image, display

os.environ["GROQ_API_KEY"] = "YOUR_API_KEY"
llm = ChatGroq(model="openai/gpt-oss-20b")

class State(TypedDict):
    question: str
    stock: str
    price: float
    answer: str

def stock(s):
    q = s["question"].upper()
    return {"stock": "AAPL" if "APPLE" in q or "AAPL" in q
            else "TSLA" if "TESLA" in q or "TSLA" in q
            else "MSFT" if "MICROSOFT" in q or "MSFT" in q else "UNKNOWN"}

def price(s):
    return {"price": {"AAPL":195.50,"TSLA":180.25,"MSFT":420.10,"UNKNOWN":0}[s["stock"]]}

def answer(s):
    return {"answer": llm.invoke(
        f"Finance assistant. Question: {s['question']}. "
        f"Stock: {s['stock']}, Price: ${s['price']}. "
        "Give a simple response. No investment advice."
    ).content}

g = StateGraph(State)
g.add_node("Stock", stock)
g.add_node("Price", price)
g.add_node("Answer", answer)
g.add_edge(START,"Stock")
g.add_edge("Stock","Price")
g.add_edge("Price","Answer")
g.add_edge("Answer",END)

bot = g.compile()

display(Image(bot.get_graph().draw_mermaid_png()))

result = bot.invoke({"question":"What is the current price of MICROSOFT stock?"})
print(result["answer"])

6......................................................................................

#6.	Create an AI-Powered Sales Report Analyzer with LlamaIndex
import os,pandas as pd
from llama_index.core import *
from llama_index.core.embeddings import MockEmbedding
from llama_index.llms.openai_like import OpenAILike

os.environ["GROQ_API_KEY"]=""
Settings.llm=OpenAILike(model="openai/gpt-oss-20b",
api_base="https://api.groq.com/openai/v1",
api_key=os.environ["GROQ_API_KEY"],is_chat_model=True)
Settings.embed_model=MockEmbedding(embed_dim=384)

df=pd.DataFrame({
"Month":["Jan","Feb","Mar","Apr","May"],
"Product":["Laptop","Mobile","Tablet","Laptop","Mobile"],
"Region":["South","North","East","West","South"],
"Revenue":[50000,40000,25000,60000,55000],
"Profit":[10000,8000,4000,12000,11000]})

display(df)

idx=VectorStoreIndex.from_documents([Document(text=df.to_string(index=False))])
print(idx.as_query_engine().query(

"""Which product generated the highest total revenue?
Show calculations for Laptop, Mobile and Tablet.
Use plain numbers only. Do not use $, USD, or any currency symbol.
Then state which product has the highest revenue."""
))
7..............................................................................
#7)Create a Market Research Agent with RAG & Cohere

!pip install -q cohere

import cohere, os

os.environ["COHERE_API_KEY"] = ""
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

docs = [ "EVs are growing in India due to low running costs and government incentives.", 
         "Main challenges are charging infrastructure, battery cost and range anxiety.", 
         "Major companies include Tata Motors, Mahindra, Ola Electric and Ather.", 
         "Opportunities include battery swapping, fast charging and renewable energy." ]

def research(q): 
    e = co.embed(model="embed-v4.0", texts=docs, input_type="search_document", embedding_types=["float"]) 
    qe = co.embed(model="embed-v4.0", texts=[q], input_type="search_query", embedding_types=["float"])

# Find most relevant documents
    scores = [
        sum(a*b for a,b in zip(qe.embeddings.float[0], d))
        for d in e.embeddings.float
    ]
    context = "\n".join(docs[i] for i in sorted(
         range(len(docs)), key=lambda i: scores[i], reverse=True
    )[:3])

    r = co.chat(
         model="command-a-03-2025",
         messages=[{"role":"user",
                    "content":f"Answer using this information:\n{context}\n\nQuestion: {q}"}]
    )
    return r.message.content[0].text
print(research("Give a short market report on Indian EVs"))
8..........................................................................................
#8.	Design a Data Analysis Agent with Phidata
import os
import pandas as pd

from phi.agent import Agent
from phi.model.groq import Groq

# Use your NEW Groq API key
os.environ["GROQ_API_KEY"] =""

# Sample sales data
df = pd.DataFrame({
    "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
    "Product": ["Laptop", "Mobile", "Tablet", "Laptop", "Mobile"],
    "Region": ["South", "North", "East", "West", "South"],
    "Revenue": [50000, 40000, 25000, 60000, 55000],
    "Profit": [10000, 8000, 4000, 12000, 11000]
})

display(df)

# Convert dataframe to text
data_text = df.to_string(index=False)

# Create Phidata agent
data_agent = Agent(
    name="Data Analysis Agent",
    model=Groq(id="openai/gpt-oss-20b"),
    instructions=[
        "You are a data analysis assistant.",
        "Analyze the given sales data carefully.",
        "Show calculations clearly.",
        "Give short business insights."
    ],
    markdown=True
)

# Question
question = "Which product has the highest total revenue? Show calculation."

# Run agent
response = data_agent.run(
    f"""
Here is the sales data:

{data_text}

Question:
{question}
"""
)

print(response.content)
9........................................................................
#9.	Simple Customer Support Chatbot using LangGraph Multi-Agent Workflow
!pip install -q langgraph langchain-groq

import os 
from typing import TypedDict 
from langgraph.graph import StateGraph, END 
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = "" 
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

class S(TypedDict): 
    query: str
    category: str 
    response: str

def router(s):
        r = llm.invoke( f"Classify as faq, order, or complaint. Return one word.\n{s['query']}" ).content.lower() 
        return {"category": "order" if "order" in r else "complaint" if "complaint" in r else "faq"}

def agent(s): 
        prompts = { "faq": "Answer politely.", "order": "Help with the order. Ask for order ID if missing.", "complaint": "Apologize and suggest the next step." } 
        return {"response": llm.invoke( f"{prompts[s['category']]}\n{s['query']}" ).content}

g = StateGraph(S)
g.add_node("router", router) 
g.add_node("agent", agent) 
g.set_entry_point("router") 
g.add_edge("router", "agent") 
g.add_edge("agent", END)

app = g.compile()

r = app.invoke({ "query": "I received a damaged product", "category": "", "response": "" })

print("Category:", r["category"]) 
print("Response:", r["response"])

10................................................................................

!pip install crewai litellm yfinance nest_asyncio -q

import os
import nest_asyncio
import yfinance as yf

nest_asyncio.apply()

os.environ.pop("OPENAI_API_KEY", None)
os.environ["GROQ_API_KEY"] = "YOUR_NEW_GROQ_API_KEY"

import litellm.main as _lm

if not getattr(_lm, "_patched", False):
    _orig = _lm.completion

    def _patched(*args, **kwargs):
        kwargs["messages"] = [
            {k: v for k, v in m.items() if k != "cache_breakpoint"}
            if isinstance(m, dict) else m
            for m in kwargs.get("messages", [])
        ]
        return _orig(*args, **kwargs)

    import litellm
    litellm.completion = _patched
    _lm.completion = _patched
    _lm._patched = True

from crewai import Agent, Task, Crew, Process, LLM

llm = LLM(
    model="groq/qwen/qwen3-32b",
    api_key=os.environ["GROQ_API_KEY"]
)

def get_stock_info(ticker: str) -> str:
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="1mo")

    avg_vol = int(hist["Volume"].mean()) if not hist.empty else "N/A"

    return f"""
Company: {info.get('longName')}
Sector: {info.get('sector')}
Industry: {info.get('industry')}
Current Price: {info.get('currentPrice')}
Market Cap: {info.get('marketCap')}
P/E Ratio: {info.get('trailingPE')}
EPS: {info.get('trailingEps')}
52-Week High: {info.get('fiftyTwoWeekHigh')}
52-Week Low: {info.get('fiftyTwoWeekLow')}
Dividend Yld: {info.get('dividendYield')}
Avg Volume: {avg_vol}
Beta: {info.get('beta')}
"""

ticker = "AAPL"
stock_data = get_stock_info(ticker)

data_collector = Agent(
    role="Stock Data Collector",
    goal="Fetch and organize raw stock data for analysis",
    backstory="You are a meticulous data specialist who gathers accurate stock market data and presents it cleanly.",
    llm=llm,
    verbose=True
)

analyst = Agent(
    role="Senior Stock Analyst",
    goal="Perform deep analysis of stock data to identify strengths, risks, and trends",
    backstory="You are a seasoned Wall Street analyst with 20 years of experience evaluating stocks and financial markets.",
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Financial Report Writer",
    goal="Write a clear, structured, beginner-friendly stock analysis report",
    backstory="You are a financial journalist who translates complex market data into easy-to-understand reports for retail investors.",
    llm=llm,
    verbose=True
)

collection_task = Task(
    description=f"""
Review and organize the following stock data for {ticker}:

{stock_data}

Summarize the key data points clearly so the analyst can work with them.
""",
    expected_output="A clean, organized summary of all key stock metrics.",
    agent=data_collector
)

analysis_task = Task(
    description=f"""
Using the collected data for {ticker}, perform a comprehensive analysis:

1. Evaluate the valuation (P/E, EPS, Market Cap)
2. Assess price momentum (52-week range, current price position)
3. Identify 3 key strengths
4. Identify 3 key risks
5. Give overall sentiment: Bullish / Neutral / Bearish with reasoning
""",
    expected_output="Detailed analysis with valuation, momentum, strengths, risks, and sentiment.",
    agent=analyst
)

report_task = Task(
    description=f"""
Using the analysis, write a professional stock report for {ticker}:

1. Executive Summary
2. Company Overview
3. Financial Highlights
4. Strengths & Opportunities
5. Risks & Challenges
6. Analyst Verdict
""",
    expected_output="A structured 6-section stock analysis report.",
    agent=writer
)

crew = Crew(
    agents=[data_collector, analyst, writer],
    tasks=[collection_task, analysis_task, report_task],
    process=Process.sequential,
    verbose=True
)

result = await crew.kickoff_async()

print("\n" + "=" * 60)
print(" FINAL STOCK ANALYSIS REPORT")
print("=" * 60)
print(result)


11...............................................................................


!pip install -q requests autogen

import os
import requests

os.environ["GROQ_API_KEY"] = "YOUR_NEW_GROQ_API_KEY"

def search_arxiv(topic: str, max_results: int = 3) -> str:
    import urllib.parse
    import xml.etree.ElementTree as ET

    query = urllib.parse.quote(topic)

    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query=all:{query}&start=0"
        f"&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return "Failed to fetch papers."

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)

    if not entries:
        return f"No papers found for: {topic}"

    results = []

    for i, entry in enumerate(entries, 1):
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip()[:300]
        link = entry.find("atom:id", ns).text.strip()

        authors = [
            a.find("atom:name", ns).text
            for a in entry.findall("atom:author", ns)
        ][:3]

        results.append(
            f"{i}. {title}\n"
            f"   Authors: {', '.join(authors)}\n"
            f"   Link: {link}\n"
            f"   Summary: {summary}...\n"
        )

    return "\n".join(results)


def get_wikipedia_summary(topic: str) -> str:
    url = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + topic.replace(" ", "_")
    )

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return f"{data.get('title')}\n{data.get('extract', 'No summary.')}"

    return f"Could not find Wikipedia article for: {topic}"


def call_agent(system_msg: str, user_msg: str) -> str:
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
            "Content-Type": "application/json"
        },
        json={
            "model": "qwen/qwen3-32b",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
    )

    return response.json()["choices"][0]["message"]["content"]


research_topic = "Large Language Models"

print(f"Researching: {research_topic}\n")

wiki_info = get_wikipedia_summary(research_topic)
papers_info = search_arxiv(research_topic, max_results=3)

print("Data gathered!\n")

print("Agent 1 - Summarizer working...\n")

summary = call_agent(
    system_msg="You are a research summarizer. Summarize the given data clearly and concisely.",
    user_msg=f"""
Summarize this research data on: {research_topic}

WIKIPEDIA:
{wiki_info}

RECENT PAPERS:
{papers_info}
"""
)

print("Agent 1 Output:\n", summary)

print("\nAgent 2 - Research Analyst working...\n")

analysis = call_agent(
    system_msg="You are an expert research analyst. Analyze the summary and identify key themes, challenges, and opportunities.",
    user_msg=f"""
Analyze this summary about {research_topic}:

{summary}

Provide:

1. Key Themes
2. Main Challenges
3. Opportunities & Applications
"""
)

print("Agent 2 Output:\n", analysis)

print("\nAgent 3 - Report Writer working...\n")

report = call_agent(
    system_msg="You are a technical report writer. Write structured, professional research reports.",
    user_msg=f"""
Write a comprehensive research report on: {research_topic}

Use this analysis:

{analysis[:1500]}

Structure the report with these sections:

1. Overview
2. Recent Research Highlights
3. Key Challenges
4. Real-World Applications
5. Future Directions
6. Conclusion

Keep each section concise and professional.
"""
)

print("\n" + "=" * 60)
print(" FINAL RESEARCH REPORT")
print("=" * 60)
print(report)
12...............................................................................
#12.	AI Observability with LangSmith is to trace one simple LLM call and then view that run in the LangSmith dashboard.
# !pip install -U langsmith langchain-groq

import os
from langchain_groq import ChatGroq

# -----------------------------
# 1. API Keys
# -----------------------------
os.environ["GROQ_API_KEY"] = ""
os.environ["LANGSMITH_API_KEY"] = ""

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"

# Optional: project name shown in LangSmith
os.environ["LANGSMITH_PROJECT"] = "AI-Observability-Demo"

# -----------------------------
# 2. Create LLM
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b"
)

# -----------------------------
# 3. Call the model
# -----------------------------
response = llm.invoke(
    "Explain Artificial Intelligence in two sentences."
)

print(response.content)


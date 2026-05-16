from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict
from csv_tools import get_csv_summary
import argparse
import json

# ── CONFIG ──────────────────────────────────────────────────────
# Local Ollama — change to your Desktop Tailscale IP when ready:
#llm_base = ChatOllama(model="llama3.1:8b", base_url="http://desktop-bbjn0a3:11434")
#llm local
llm_base = ChatOllama(model="llama3.2:3b", base_url="http://localhost:11434")
filename="test.csv"


# --- CLASSES ----------------------------------------------------

class Transaction:
    def __init__(self):
        self.transactions =[]
    
    #def describe(self) -> str:
        #return f"{self.amount} {self.currency}"
    
    def deposit(self, amount: float, currency: str):
        self.transactions.append({"amount" : amount, "currency" : currency})
    
    def to_prompt(self) -> str:
        return json.dumps(self.transactions, indent=2)
        

# ── TOOLS ────────────────────────────────────────────────────────
@tool
def read_file(filename: str) -> str:
    """Read the contents of a local text file."""
    try:
        with open(filename) as f:
            return f.read()
    except FileNotFoundError:
        return f"File {filename} not found"

@tool
def word_count(text: str) -> str:
    """Count the number of words in a text string."""
    count = len(text.split())
    return f"{count} words"





tools = [read_file, word_count, get_csv_summary]

# ── AGENTS ───────────────────────────────────────────────────────

# Agent 1: Researcher — has access to tools
llm_with_tools = llm_base.bind_tools(tools)

def researcher(state: MessagesState):
    """Researches the topic using available tools and LLM."""
    system = SystemMessage(content=(
        "You are a research assistant."
        "Use tools when helpful."
        "If the user mentions a CSV file, use the get_csv_summary tool to analyze it."
        "Do not try to parse CSV files manually."
        #"If the user wants to analize a transaction, use the Transaction class and analyze its contents."
        "Gather key facts about the topic provided."
    ))
    messages = [system] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Agent 2: Writer — no tools, just writes
def writer(state: MessagesState):
    """Takes research output and writes a clear summary paragraph."""
    system = SystemMessage(content=(
        "You are a professional writer. "
        "Read the conversation so far and write a clear, concise paragraph "
        "summarising the key findings. Do not use bullet points."
    ))
    messages = [system] + state["messages"]
    response = llm_base.invoke(messages)
    return {"messages": [response]}



# ── GRAPH ────────────────────────────────────────────────────────
graph = StateGraph(MessagesState)

graph.add_node("researcher", researcher)
graph.add_node("tools", ToolNode(tools))
graph.add_node("writer", writer)

graph.add_edge(START, "researcher")
graph.add_conditional_edges("researcher", tools_condition)
graph.add_edge("tools", "researcher")
graph.add_edge("writer", END)

app = graph.compile()

# ── RUN ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=False)
parser.add_argument("--amount", required=False)
parser.add_argument("--currency", required=False)
parser.add_argument("--message", required=False)
args = parser.parse_args()

if __name__ == "__main__":

    #Transaction:
    ledger = Transaction()
    ledger.deposit(args.amount, args.currency)
    
    #topic = "financial crime detection with AI"
    
    
    
    result = app.invoke({
        #"messages": [HumanMessage(content=f"Research this topic: {topic}")]
        #"messages": [HumanMessage(content=f"Analyze this transaction:\n {ledger.to_prompt()}. ")]
        #"messages": [HumanMessage(content=f"Read the contents of {args.file} and give me some statistics")]
        "messages": [HumanMessage(content=f"{args.message}")]
    })


    print("=== FINAL OUTPUT ===")
    print(result["messages"][-1].content)
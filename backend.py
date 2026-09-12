import os
import certifi
from dotenv import load_dotenv
from langgraph.graph import StateGraph,END,START
import uuid
import operator
from typing import TypedDict,Annotated
import psycopg
from psycopg.rows import dict_row #return PostgreSQL rows as dictionaries instead of tuples.
from langgraph.checkpoint.postgres import PostgresSaver #we cant use the in memory checkpoint here therefore we are using psotgrssaver this will persist the state even after the appliction is restarted 
#postgressave measn Save LangGraph checkpoints in PostgreSQL.

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

load_dotenv()
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url

GROQ_API_KEY=os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("api key is missing please add it in your .env file")

#llm 
llm=ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY
)

#defining the state
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int

#flight agent 
def flight_agent(state:TravelState):
    query=state["user_query"]
    flight_data=search_flights(query)
    return{
        "flight_results":flight_data,
        "messages":[AIMessage(content="flight result fetched ")],
        "llm_calls":state.get("llm_calls",0)+1
    }


#hotel agent 
def hotel_agent(state:TravelState):
    query=f"best hotels for {state['user_query']}"
    hotel_results=tavily_search(query)
    return{
        "hotel_results":hotel_results,
        "messages":[AIMessage(content="Hotel information  fetched ")],
        "llm_calls":state.get("llm_calls",0)+1
    }

#itenary agent here we will require the llm for making the plan 
def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Make the itinerary practical, budget-aware, and easy to follow.
"""

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

#final response agent whih will generte the entire plan 
def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Keep the response useful for real travel planning.
"""

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

#build the graph using langgraph
graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


#adding the checkipoiter and estblishing the database conncetion 
DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)

#final function which will shownoutput in the frontend 
def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }




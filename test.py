# from tools.tavily_tool import tavily_search
# from tools.flight_tool import search_flights
# from backend import run_travel_agent
# # res=tavily_search("best hotels in india")
# # print(res)

# # res=search_flights("plan a 7 days trip from india to uae")
# # print(res)

# user_input = input("Enter travel request: ")

# response = run_travel_agent(
#     user_input=user_input,
#     thread_id="test_user"
# )

# print("\nFINAL RESPONSE:\n")
# print(response["answer"])


import asyncio
from mcp_client_test import get_all_tools,tavily_mcp_search

if __name__=="__main__":
    query="latest news about ai"
    asyncio.run(tavily_mcp_search(query))

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
# res=tavily_search("best hotels in india")
# print(res)

res=search_flights("plan a 7 days trip from india to uae")
print(res)
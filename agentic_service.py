import os
import asyncio
import requests
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch

WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4,
    max_tokens=500,
)

@tool
def get_weather_update(city_name: str) -> str:
    if not WEATHERSTACK_API_KEY:
        return "Missing WEATHERSTACK_API_KEY"

    url = f"http://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={city_name}"
    r = requests.get(url, timeout=10)
    data = r.json()

    if "error" in data:
        return data["error"].get("info", "Weather API error")

    loc = data["location"]["name"]
    temp = data["current"]["temperature"]
    desc = data["current"]["weather_descriptions"][0]

    return f"Weather in {loc}: {desc}, {temp}°C"


tools = [
    TavilySearch(max_results=3),
    get_weather_update,
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful weather assistant.",
)


async def process_request(input_data: dict) -> str:
    """
    THIS is what Masumi calls after payment.
    """

    text = input_data.get("text", "")

    payload = {
        "messages": [
            {"role": "user", "content": text}
        ]
    }

    result = await asyncio.to_thread(agent.invoke, payload)

    msgs = result.get("messages", [])
    return msgs[-1].content if msgs else str(result)

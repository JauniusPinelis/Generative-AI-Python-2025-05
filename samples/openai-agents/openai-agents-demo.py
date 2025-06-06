

import os
from typing import TypedDict
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from openai import AsyncOpenAI

class Location(TypedDict):
    lat: float
    long: float

@function_tool  
async def fetch_weather(location: Location) -> str:
    
    """Fetch the weather for a given location.

    Args:
        location: The location to fetch the weather for.
    """
    # In real life, we'd fetch the weather from a weather API
    return "super hot and sunny"

@function_tool  
async def save_to_file(weather: str):

    """Save the weather information for a given location to a file.

    Args:
        location: The location to fetch the weather for.
    """
    # In real life, we'd fetch the weather from a weather API
    print (f"Saving weather information to file: {weather}")

# Settings
token = os.getenv('SECRET')
endpoint = 'https://models.github.ai/inference'
model = 'openai/gpt-4.1-nano'
 
client = AsyncOpenAI(
    base_url = endpoint,
    api_key = token
)
set_tracing_disabled(True)
 
model_instance = OpenAIChatCompletionsModel(
    model = model,
    openai_client = client
)
 
 
agent = Agent(
    name = "Assistant",
    instructions = "You are a helpful assistant that can fetch weather information.",
    model = model_instance,
    model_settings = ModelSettings(
            temperature = 0.1
        ),
    tools=[fetch_weather, save_to_file]
)

async def main():
    result = await Runner.run(agent, "What's the weather like in Vilnius? Please save the weather information to a file.")
    print(result.final_output)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


TITLE: Setting OpenAI API Key (Bash)
DESCRIPTION: This command sets your OpenAI API key as an environment variable. The Agents SDK uses this key to authenticate requests to the OpenAI API, enabling your agents to interact with models.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/quickstart.md#_snippet_3

LANGUAGE: bash
CODE:
```
export OPENAI_API_KEY=sk-...
```

----------------------------------------

TITLE: Creating a Single Agent (Python)
DESCRIPTION: This snippet demonstrates how to instantiate a basic `Agent` object from the `agents` library. An agent is defined by its `name` and `instructions`, which guide its behavior and responses.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/quickstart.md#_snippet_4

LANGUAGE: python
CODE:
```
from agents import Agent

agent = Agent(
    name="Math Tutor",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)
```

----------------------------------------

TITLE: Installing OpenAI Agents SDK (Bash)
DESCRIPTION: This snippet demonstrates how to install the OpenAI Agents SDK using the pip package manager. This is the essential first step to set up the SDK in your development environment.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/index.md#_snippet_0

LANGUAGE: bash
CODE:
```
pip install openai-agents
```

----------------------------------------

TITLE: Installing OpenAI Agents SDK (Bash)
DESCRIPTION: This command installs the OpenAI Agents SDK into your active virtual environment using pip. This makes the `agents` library available for use in your Python project.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/quickstart.md#_snippet_2

LANGUAGE: bash
CODE:
```
pip install openai-agents # or `uv add openai-agents`, etc
```

----------------------------------------

TITLE: Running a Basic Agent with OpenAI Agents SDK (Python)
DESCRIPTION: This example initializes a simple AI agent with specific instructions and uses the `Runner` class to execute a task synchronously. It illustrates the fundamental process of creating an `Agent` instance and retrieving its final output, showcasing a minimal agentic workflow.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/index.md#_snippet_1

LANGUAGE: python
CODE:
```
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
```

----------------------------------------

TITLE: Running a Complete Agent Workflow with Guardrails and Handoffs (Python)
DESCRIPTION: This comprehensive example integrates agent definitions, handoffs, and an input guardrail into a full asynchronous workflow. It demonstrates how to run the `triage_agent` with different user inputs, showcasing the routing and guardrail functionality in action.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/quickstart.md#_snippet_9

LANGUAGE: python
CODE:
```
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from pydantic import BaseModel
import asyncio

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)


async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)

async def main():
    result = await Runner.run(triage_agent, "who was the first president of the united states?")
    print(result.final_output)

    result = await Runner.run(triage_agent, "what is life")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

----------------------------------------

TITLE: Basic Agent Creation and Execution (Hello World)
DESCRIPTION: This example shows how to create a simple `Agent` with instructions and use `Runner.run_sync` to execute a task. It demonstrates a basic synchronous interaction with the agent to generate a haiku.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/README.md#_snippet_2

LANGUAGE: python
CODE:
```
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```

----------------------------------------

TITLE: Running an Agent Asynchronously with Runner.run() in Python
DESCRIPTION: This snippet demonstrates how to initialize an `Agent` and execute it asynchronously using `Runner.run()`. It shows how to pass an initial prompt to the agent and print its final output. The `Runner.run()` method returns a `RunResult` object containing the execution outcome.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/running_agents.md#_snippet_0

LANGUAGE: Python
CODE:
```
from agents import Agent, Runner

async def main():
    agent = Agent(name="Assistant", instructions="You are a helpful assistant")

    result = await Runner.run(agent, "Write a haiku about recursion in programming.")
    print(result.final_output)
    # Code within the code,
    # Functions calling themselves,
    # Infinite loop's dance.
```

----------------------------------------

TITLE: Activating Virtual Environment (Bash)
DESCRIPTION: This command activates the previously created Python virtual environment. It must be run in every new terminal session to ensure that subsequent Python commands and package installations are isolated within this environment.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/quickstart.md#_snippet_1

LANGUAGE: bash
CODE:
```
source .venv/bin/activate
```

----------------------------------------

TITLE: Defining Function Tools with Decorators in Python
DESCRIPTION: This snippet demonstrates how to define Python functions as tools using the `@function_tool` decorator. It shows both a simple async function (`fetch_weather`) and a function with an overridden name and optional arguments (`read_file`), illustrating how docstrings are used for descriptions and how function arguments are automatically converted to JSON schemas. It also shows how to instantiate an `Agent` with these tools and inspect their properties.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/tools.md#_snippet_1

LANGUAGE: Python
CODE:
```
import json

from typing_extensions import TypedDict, Any

from agents import Agent, FunctionTool, RunContextWrapper, function_tool


class Location(TypedDict):
    lat: float
    long: float

@function_tool  # (1)!
async def fetch_weather(location: Location) -> str:
    # (2)!
    """Fetch the weather for a given location.

    Args:
        location: The location to fetch the weather for.
    """
    # In real life, we'd fetch the weather from a weather API
    return "sunny"


@function_tool(name_override="fetch_data")  # (3)!
def read_file(ctx: RunContextWrapper[Any], path: str, directory: str | None = None) -> str:
    """Read the contents of a file.

    Args:
        path: The path to the file to read.
        directory: The directory to read the file from.
    """
    # In real life, we'd read the file from the file system
    return "<file contents>"


agent = Agent(
    name="Assistant",
    tools=[fetch_weather, read_file],  # (4)!
)

for tool in agent.tools:
    if isinstance(tool, FunctionTool):
        print(tool.name)
        print(tool.description)
        print(json.dumps(tool.params_json_schema, indent=2))
        print()
```

----------------------------------------

TITLE: Running Agent Orchestration (Python)
DESCRIPTION: This asynchronous function demonstrates how to execute an agent workflow using the `Runner.run()` method. It initiates the process with a `triage_agent` and a specific query, then prints the final output generated by the orchestrated agents.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/quickstart.md#_snippet_7

LANGUAGE: python
CODE:
```
from agents import Runner

async def main():
    result = await Runner.run(triage_agent, "What is the capital of France?")
    print(result.final_output)
```

----------------------------------------

TITLE: Disabling LLM Data Logging (Bash)
DESCRIPTION: This snippet shows how to disable the logging of sensitive LLM (Large Language Model) inputs and outputs by setting an environment variable. This is crucial for privacy and security when dealing with user data.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/config.md#_snippet_7

LANGUAGE: Bash
CODE:
```
export OPENAI_AGENTS_DONT_LOG_MODEL_DATA=1
```

----------------------------------------

TITLE: Managing Multi-Turn Conversations with Agents in Python
DESCRIPTION: This example illustrates how to maintain a conversational thread across multiple turns using the `Runner.run()` method. It demonstrates using `result.to_input_list()` to capture the previous turn's context and append new user input for subsequent agent runs, simulating a continuous chat experience. Tracing is also enabled for the conversation.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/running_agents.md#_snippet_1

LANGUAGE: Python
CODE:
```
async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")

    with trace(workflow_name="Conversation", group_id=thread_id):
        # First turn
        result = await Runner.run(agent, "What city is the Golden Gate Bridge in?")
        print(result.final_output)
        # San Francisco

        # Second turn
        new_input = result.to_input_list() + [{"role": "user", "content": "What state is it in?"}]
        result = await Runner.run(agent, new_input)
        print(result.final_output)
        # California
```

----------------------------------------

TITLE: Integrating Custom Functions as Tools
DESCRIPTION: This snippet demonstrates how to define and integrate custom Python functions as tools for an agent using the `@function_tool` decorator. The agent can then call these tools based on its instructions and user input, such as fetching weather information.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/README.md#_snippet_4

LANGUAGE: python
CODE:
```
import asyncio

from agents import Agent, Runner, function_tool


@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."


agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_weather]
)


async def main():
    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    print(result.final_output)
    # The weather in Tokyo is sunny.


if __name__ == "__main__":
    asyncio.run(main())
```

----------------------------------------

TITLE: Setting OpenAI API Key Environment Variable (Bash)
DESCRIPTION: This snippet shows how to set the `OPENAI_API_KEY` environment variable in a bash shell. This key is crucial for the OpenAI Agents SDK to authenticate and interact with the OpenAI API services.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/index.md#_snippet_2

LANGUAGE: bash
CODE:
```
export OPENAI_API_KEY=sk-...
```

----------------------------------------

TITLE: Initializing Agent with Web and File Search Tools in Python
DESCRIPTION: This snippet demonstrates how to initialize an `Agent` instance with `WebSearchTool` and `FileSearchTool`. The `FileSearchTool` is configured to return a maximum of 3 results and requires a `vector_store_ids`. It then shows how to run an asynchronous query using `Runner.run` with the configured agent and print the final output.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/tools.md#_snippet_0

LANGUAGE: python
CODE:
```
from agents import Agent, FileSearchTool, Runner, WebSearchTool

agent = Agent(
    name="Assistant",
    tools=[
        WebSearchTool(),
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=["VECTOR_STORE_ID"],
        ),
    ],
)

async def main():
    result = await Runner.run(agent, "Which coffee shop should I go to, taking into account my preferences and the weather today in SF?")
    print(result.final_output)
```

----------------------------------------

TITLE: Handling Streamed Audio Pipeline Results in Python
DESCRIPTION: This Python code demonstrates how to process the asynchronous stream of events returned by a `VoicePipeline` run. It shows how to await the `run()` method and then iterate through `VoiceStreamEvent` types, such as `voice_stream_event_audio` for audio playback, `voice_stream_event_lifecycle` for turn management, and `voice_stream_event_error` for error handling. This pattern is crucial for real-time voice applications.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/voice/pipeline.md#_snippet_1

LANGUAGE: Python
CODE:
```
result = await pipeline.run(input)

async for event in result.stream():
    if event.type == "voice_stream_event_audio":
        # play audio
    elif event.type == "voice_stream_event_lifecycle":
        # lifecycle
    elif event.type == "voice_stream_event_error"
        # error
    ...
```

----------------------------------------

TITLE: Setting Default OpenAI API Key (Python)
DESCRIPTION: This snippet demonstrates how to programmatically set the default OpenAI API key using set_default_openai_key(). This is useful when the OPENAI_API_KEY environment variable cannot be set before the application starts. The key is used for LLM requests and tracing.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/config.md#_snippet_0

LANGUAGE: Python
CODE:
```
from agents import set_default_openai_key

set_default_openai_key("sk-...")
```

----------------------------------------

TITLE: Implementing Output Guardrail with Nested Agent in Python
DESCRIPTION: This snippet illustrates how to implement an output guardrail that checks if the agent's final response includes any math. It utilizes a 'guardrail_agent' to analyze the output and triggers a tripwire if math content is found, preventing the potentially undesirable output from being delivered. The `output_guardrail` decorator marks the function as an output guardrail.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md#_snippet_1

LANGUAGE: Python
CODE:
```
from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    output_guardrail,
)
class MessageOutput(BaseModel): # (1)!
    response: str

class MathOutput(BaseModel): # (2)!
    reasoning: str
    is_math: bool

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the output includes any math.",
    output_type=MathOutput,
)

@output_guardrail
async def math_guardrail(  # (3)!
    ctx: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, output.response, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math,
    )

agent = Agent( # (4)!
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    output_guardrails=[math_guardrail],
    output_type=MessageOutput,
)

async def main():
    # This should trip the guardrail
    try:
        await Runner.run(agent, "Hello, can you help me solve for x: 2x + 3 = 11?")
        print("Guardrail didn't trip - this is unexpected")

    except OutputGuardrailTripwireTriggered:
        print("Math output guardrail tripped")
```

----------------------------------------

TITLE: Installing OpenAI Agents SDK
DESCRIPTION: This command installs the OpenAI Agents SDK using pip. An optional 'voice' group can be installed for voice support by appending '[voice]' to the package name.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/README.md#_snippet_1

LANGUAGE: bash
CODE:
```
pip install openai-agents
```

----------------------------------------

TITLE: Customizing Agent Tool Execution with Runner.run in OpenAI Agents Python
DESCRIPTION: This code illustrates how to create a custom function tool that runs an agent with advanced configurations not directly supported by 'agent.as_tool'. By decorating an asynchronous function with '@function_tool' and directly calling 'Runner.run' inside it, developers can specify parameters like 'max_turns' or 'run_config' for fine-grained control over agent execution within a tool.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/tools.md#_snippet_5

LANGUAGE: python
CODE:
```
@function_tool
async def run_my_agent() -> str:
    """A tool that runs the agent with custom configs"""

    agent = Agent(name="My agent", instructions="...")

    result = await Runner.run(
        agent,
        input="...",
        max_turns=5,
        run_config=...
    )

    return str(result.final_output)
```

----------------------------------------

TITLE: Defining Multiple Agents with Handoff Descriptions (Python)
DESCRIPTION: This code shows how to create multiple specialized `Agent` instances. Each agent includes a `handoff_description` which provides additional context, assisting a routing agent in determining the most appropriate agent for a given task.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/quickstart.md#_snippet_5

LANGUAGE: python
CODE:
```
from agents import Agent

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)
```

----------------------------------------

TITLE: Defining Agent Handoffs (Python)
DESCRIPTION: This snippet illustrates how to configure an agent, such as a 'Triage Agent', with a list of `handoffs`. This allows the agent to dynamically route tasks to other specified agents based on its instructions and the incoming query.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/quickstart.md#_snippet_6

LANGUAGE: python
CODE:
```
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent]
)
```

----------------------------------------

TITLE: Complete Voice Agent Application Example (Python)
DESCRIPTION: This comprehensive Python script combines agent definition, tool usage, handoff logic, voice pipeline setup, and audio processing into a single runnable example. It demonstrates how to create a voice-enabled assistant that can use tools (like get_weather) and handoff to specialized agents (like spanish_agent) based on user input, processing audio from input to output.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/voice/quickstart.md#_snippet_5

LANGUAGE: python
CODE:
```
import asyncio
import random

import numpy as np
import sounddevice as sd

from agents import (
    Agent,
    function_tool,
    set_tracing_disabled,
)
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"[debug] get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."


spanish_agent = Agent(
    name="Spanish",
    handoff_description="A spanish speaking agent.",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. Speak in Spanish.",
    ),
    model="gpt-4o-mini",
)

agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. If the user speaks in Spanish, handoff to the spanish agent.",
    ),
    model="gpt-4o-mini",
    handoffs=[spanish_agent],
    tools=[get_weather],
)


async def main():
    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))
    buffer = np.zeros(24000 * 3, dtype=np.int16)
    audio_input = AudioInput(buffer=buffer)

    result = await pipeline.run(audio_input)

    # Create an audio player using `sounddevice`
    player = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
    player.start()

    # Play the audio stream as it comes in
    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            player.write(event.data)


if __name__ == "__main__":
    asyncio.run(main())
```

----------------------------------------

TITLE: Configuring Agent for Structured Pydantic Output (Python)
DESCRIPTION: This snippet shows how to configure an `Agent` to produce structured outputs using Pydantic models. By setting `output_type` to `CalendarEvent` (a Pydantic `BaseModel`), the agent is instructed to generate responses that conform to the defined schema, facilitating structured data extraction.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/agents.md#_snippet_2

LANGUAGE: python
CODE:
```
from pydantic import BaseModel
from agents import Agent


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

agent = Agent(
    name="Calendar extractor",
    instructions="Extract calendar events from text",
    output_type=CalendarEvent,
)
```

----------------------------------------

TITLE: Example of Agent with LiteLLM Model Integration in Python
DESCRIPTION: This Python script demonstrates how to create an `Agent` using `LitellmModel` to interact with various AI providers. It includes a `function_tool` for weather information and prompts the user for a model name and API key, showcasing a complete agent setup and execution flow.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/models/litellm.md#_snippet_1

LANGUAGE: python
CODE:
```
from __future__ import annotations

import asyncio

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel

@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


async def main(model: str, api_key: str):
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=LitellmModel(model=model, api_key=api_key),
        tools=[get_weather],
    )

    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)


if __name__ == "__main__":
    # First try to get model/api key from args
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--api-key", type=str, required=False)
    args = parser.parse_args()

    model = args.model
    if not model:
        model = input("Enter a model name for Litellm: ")

    api_key = args.api_key
    if not api_key:
        api_key = input("Enter an API key for Litellm: ")

    asyncio.run(main(model, api_key))
```

----------------------------------------

TITLE: Using LiteLLM-Prefixed Models with OpenAI Agents
DESCRIPTION: This code illustrates how to initialize Agent instances using models integrated via LiteLLM. It shows examples of configuring agents with Anthropic's Claude and Google's Gemini models by prefixing their names with 'litellm/', enabling access to a wide range of supported LLMs.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/models/index.md#_snippet_1

LANGUAGE: python
CODE:
```
claude_agent = Agent(model="litellm/anthropic/claude-3-5-sonnet-20240620", ...)
gemini_agent = Agent(model="litellm/gemini/gemini-2.5-flash-preview-04-17", ...)
```

----------------------------------------

TITLE: Defining Agent Context with a Dataclass (Python)
DESCRIPTION: This code illustrates how to define a custom `context` type for an agent using a Python dataclass. The `UserContext` class serves as a container for dependencies and state, which can be passed to the agent during its run, enabling context-aware operations.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/agents.md#_snippet_1

LANGUAGE: python
CODE:
```
@dataclass
class UserContext:
    uid: str
    is_pro_user: bool

    async def fetch_purchases() -> list[Purchase]:
        return ...

agent = Agent[UserContext](
    ...,
)
```

----------------------------------------

TITLE: Creating a Basic Agent Handoff in Python
DESCRIPTION: This snippet demonstrates how to create a simple agent handoff. It initializes `billing_agent` and `refund_agent`, then configures `triage_agent` to hand off tasks to them. Handoffs can be specified directly as `Agent` objects or using the `handoff()` function for more control.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/handoffs.md#_snippet_0

LANGUAGE: Python
CODE:
```
from agents import Agent, handoff

billing_agent = Agent(name="Billing agent")
refund_agent = Agent(name="Refund agent")

# (1)!
triage_agent = Agent(name="Triage agent", handoffs=[billing_agent, handoff(refund_agent)])
```

----------------------------------------

TITLE: Generating Agent Graph with draw_graph - Python
DESCRIPTION: This example demonstrates how to define multiple `Agent` instances, including one with tools and handoffs, and then use the `draw_graph` function from `agents.extensions.visualization` to generate a visual representation of the agent structure. The `triage_agent` acts as an entry point, routing requests to `spanish_agent` or `english_agent` and utilizing the `get_weather` tool.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/visualization.md#_snippet_1

LANGUAGE: python
CODE:
```
from agents import Agent, function_tool
from agents.extensions.visualization import draw_graph

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
    tools=[get_weather],
)

draw_graph(triage_agent)
```

----------------------------------------

TITLE: Customizing Agent Handoffs with `handoff()` in Python
DESCRIPTION: This example illustrates how to customize an agent handoff using the `handoff()` function. It defines an `on_handoff` callback for custom logic, and overrides the default tool name and description. This allows for fine-grained control over the handoff behavior and its representation to the LLM.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/handoffs.md#_snippet_1

LANGUAGE: Python
CODE:
```
from agents import Agent, handoff, RunContextWrapper

def on_handoff(ctx: RunContextWrapper[None]):
    print("Handoff called")

agent = Agent(name="My agent")

handoff_obj = handoff(
    agent=agent,
    on_handoff=on_handoff,
    tool_name_override="custom_handoff_tool",
    tool_description_override="Custom description",
)
```

----------------------------------------

TITLE: Configuring a Basic Agent with Tools (Python)
DESCRIPTION: This snippet demonstrates the basic configuration of an `Agent` in Python, including setting its name, instructions, the LLM model to use, and integrating a custom `function_tool`. The `get_weather` function is decorated as a tool, allowing the agent to use it for specific tasks.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/agents.md#_snippet_0

LANGUAGE: python
CODE:
```
from agents import Agent, ModelSettings, function_tool

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"

agent = Agent(
    name="Haiku agent",
    instructions="Always respond in haiku form",
    model="o3-mini",
    tools=[get_weather],
)
```

----------------------------------------

TITLE: Defining Agents with Tools and Handoffs (Python)
DESCRIPTION: This Python snippet defines two Agent instances: spanish_agent and agent. The agent includes a function_tool for get_weather and is configured to handoff to spanish_agent if the user speaks Spanish, demonstrating agent collaboration and tool usage within the SDK.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/voice/quickstart.md#_snippet_2

LANGUAGE: python
CODE:
```
import asyncio
import random

from agents import (
    Agent,
    function_tool,
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions



@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"[debug] get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."


spanish_agent = Agent(
    name="Spanish",
    handoff_description="A spanish speaking agent.",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. Speak in Spanish.",
    ),
    model="gpt-4o-mini",
)

agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. If the user speaks in Spanish, handoff to the spanish agent.",
    ),
    model="gpt-4o-mini",
    handoffs=[spanish_agent],
    tools=[get_weather],
)
```

----------------------------------------

TITLE: Integrating Recommended Handoff Prompts in Python
DESCRIPTION: This snippet demonstrates how to incorporate recommended handoff instructions into an agent's prompt using `RECOMMENDED_PROMPT_PREFIX`. This prefix helps LLMs better understand and utilize handoff capabilities, improving the overall agent delegation process. It's crucial for ensuring proper LLM behavior with handoffs.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/handoffs.md#_snippet_4

LANGUAGE: Python
CODE:
```
from agents import Agent
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

billing_agent = Agent(
    name="Billing agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    <Fill in the rest of your prompt here>.""",
)
```

----------------------------------------

TITLE: Defining Handoff Inputs with Pydantic in Python
DESCRIPTION: This snippet demonstrates how to configure a handoff to expect structured input from the LLM using a Pydantic `BaseModel`. The `EscalationData` model defines the expected `reason` field. The `on_handoff` callback receives this structured input, enabling the agent to process specific data provided during the handoff.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/handoffs.md#_snippet_2

LANGUAGE: Python
CODE:
```
from pydantic import BaseModel

from agents import Agent, handoff, RunContextWrapper

class EscalationData(BaseModel):
    reason: str

async def on_handoff(ctx: RunContextWrapper[None], input_data: EscalationData):
    print(f"Escalation agent called with reason: {input_data.reason}")

agent = Agent(name="Escalation agent")

handoff_obj = handoff(
    agent=agent,
    on_handoff=on_handoff,
    input_type=EscalationData,
)
```

----------------------------------------

TITLE: Executing the Voice Pipeline with Audio Input (Python)
DESCRIPTION: This Python snippet demonstrates how to run the configured VoicePipeline with simulated audio input. It creates a silent audio buffer, passes it to the pipeline, and then streams the resulting audio output using sounddevice, illustrating the end-to-end audio processing.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/voice/quickstart.md#_snippet_4

LANGUAGE: python
CODE:
```
import numpy as np
import sounddevice as sd
from agents.voice import AudioInput

# For simplicity, we'll just create 3 seconds of silence
# In reality, you'd get microphone data
buffer = np.zeros(24000 * 3, dtype=np.int16)
audio_input = AudioInput(buffer=buffer)

result = await pipeline.run(audio_input)

# Create an audio player using `sounddevice`
player = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
player.start()

# Play the audio stream as it comes in
async for event in result.stream():
    if event.type == "voice_stream_event_audio":
        player.write(event.data)
```

----------------------------------------

TITLE: Orchestrating Multiple Translation Agents with OpenAI Agents Python
DESCRIPTION: This snippet demonstrates how to create and orchestrate multiple specialized agents (e.g., for Spanish and French translation) using a central 'orchestrator_agent'. It shows how to convert agents into callable tools using the 'as_tool' method and execute them asynchronously via 'Runner.run' to handle user requests requiring specific translations.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/tools.md#_snippet_4

LANGUAGE: python
CODE:
```
from agents import Agent, Runner
import asyncio

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You translate the user's message to Spanish",
)

french_agent = Agent(
    name="French agent",
    instructions="You translate the user's message to French",
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate."
        "If asked for multiple translations, you call the relevant tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
    ],
)

async def main():
    result = await Runner.run(orchestrator_agent, input="Say 'Hello, how are you?' in Spanish.")
    print(result.final_output)
```

----------------------------------------

TITLE: Setting Up a Voice Pipeline with SingleAgentVoiceWorkflow (Python)
DESCRIPTION: This Python snippet initializes a VoicePipeline using SingleAgentVoiceWorkflow. It demonstrates how to integrate a pre-defined agent (from the previous step) into a voice workflow, preparing it to handle audio input and generate audio output through the pipeline.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/voice/quickstart.md#_snippet_3

LANGUAGE: python
CODE:
```
from agents.voice import SingleAgentVoiceWorkflow, VoicePipeline
pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))
```

----------------------------------------

TITLE: Orchestrating Agents with Handoffs (Python)
DESCRIPTION: This example demonstrates the use of `handoffs` to enable an agent to delegate tasks to specialized sub-agents. The `triage_agent` is configured with instructions to route user queries to either a `booking_agent` or a `refund_agent` based on the query's relevance, promoting modularity and specialization.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/agents.md#_snippet_3

LANGUAGE: python
CODE:
```
from agents import Agent

booking_agent = Agent(...)
refund_agent = Agent(...)

triage_agent = Agent(
    name="Triage agent",
    instructions=(
        "Help the user with their questions."
        "If they ask about booking, handoff to the booking agent."
        "If they ask about refunds, handoff to the refund agent."
    ),
    handoffs=[booking_agent, refund_agent],
)
```

----------------------------------------

TITLE: Configuring Agent with Multiple MCP Servers - Python
DESCRIPTION: This snippet shows how to instantiate an Agent and associate it with one or more MCP servers. The 'mcp_servers' parameter accepts a list of initialized MCP server instances, enabling the Agent to discover and utilize the tools provided by these servers during its execution.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/mcp.md#_snippet_1

LANGUAGE: python
CODE:
```
agent=Agent(
    name="Assistant",
    instructions="Use the tools to achieve the task",
    mcp_servers=[mcp_server_1, mcp_server_2]
)
```

----------------------------------------

TITLE: Managing Local Context with RunContextWrapper in Python
DESCRIPTION: This Python snippet demonstrates how to define and utilize local context within an OpenAI Agent run using `RunContextWrapper`. It shows how to pass a custom dataclass (`UserInfo`) as context to the `Runner.run` method, allowing tool functions to access and use this contextual data for their operations. This local context is strictly for internal code logic and is not exposed to the LLM.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/context.md#_snippet_0

LANGUAGE: python
CODE:
```
import asyncio
from dataclasses import dataclass

from agents import Agent, RunContextWrapper, Runner, function_tool

@dataclass
class UserInfo:  # (1)!
    name: str
    uid: int

@function_tool
async def fetch_user_age(wrapper: RunContextWrapper[UserInfo]) -> str:  # (2)!
    return f"User {wrapper.context.name} is 47 years old"

async def main():
    user_info = UserInfo(name="John", uid=123)

    agent = Agent[UserInfo](  # (3)!
        name="Assistant",
        tools=[fetch_user_age],
    )

    result = await Runner.run(  # (4)!
        starting_agent=agent,
        input="What is the age of the user?",
        context=user_info,
    )

    print(result.final_output)  # (5)!
    # The user John is 47 years old.

if __name__ == "__main__":
    asyncio.run(main())
```

----------------------------------------

TITLE: Applying Model Settings to an Agent in OpenAI Agents
DESCRIPTION: This snippet illustrates how to apply additional configuration parameters, such as 'temperature', to an Agent's model using 'ModelSettings'. This allows for fine-tuning the behavior of the underlying LLM beyond just selecting the model name, enabling more precise control over responses.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/models/index.md#_snippet_3

LANGUAGE: python
CODE:
```
from agents import Agent, ModelSettings

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    model="gpt-4o",
    model_settings=ModelSettings(temperature=0.1),
)
```

----------------------------------------

TITLE: Implementing Agent Handoffs
DESCRIPTION: This example illustrates how to configure multiple agents and use a 'triage' agent to hand off control to the appropriate agent based on the input language. It showcases asynchronous execution with `asyncio` and the `handoffs` parameter for agent routing.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/README.md#_snippet_3

LANGUAGE: python
CODE:
```
from agents import Agent, Runner
import asyncio

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish."
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English"
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent]
)


async def main():
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print(result.final_output)
    # ¡Hola! Estoy bien, gracias por preguntar. ¿Y tú, cómo estás?


if __name__ == "__main__":
    asyncio.run(main())
```

----------------------------------------

TITLE: Creating Higher-Level Traces with OpenAI Agents SDK (Python)
DESCRIPTION: This snippet demonstrates how to group multiple `Runner.run` calls into a single, higher-level trace using the `trace` context manager. By wrapping the agent runs within a `with trace(...)` block, all operations within that block become part of the same logical workflow trace, rather than generating separate traces for each `run` call. This is useful for monitoring multi-step agent interactions as a single end-to-end operation.
SOURCE: https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md#_snippet_0

LANGUAGE: Python
CODE:
```
from agents import Agent, Runner, trace

async def main():
    agent = Agent(name="Joke generator", instructions="Tell funny jokes.")

    with trace("Joke workflow"): # (1)!
        first_result = await Runner.run(agent, "Tell me a joke")
        second_result = await Runner.run(agent, f"Rate this joke: {first_result.final_output}")
        print(f"Joke: {first_result.final_output}")
        print(f"Rating: {second_result.final_output}")
```
TITLE: Installing OpenAI Python Library
DESCRIPTION: This command installs the OpenAI Python library from PyPI, making it available for use in Python 3.8+ applications. It is the first step required to set up the library in your development environment.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_0

LANGUAGE: sh
CODE:
```
pip install openai
```

----------------------------------------

TITLE: Auto-parsing Function Tool Calls with Pydantic and `pydantic_function_tool` (Python)
DESCRIPTION: This example illustrates how `client.beta.chat.completions.parse()` can automatically parse function tool calls. It defines a `Query` Pydantic model for database queries, uses `openai.pydantic_function_tool()` to register it as a tool, and then demonstrates how the model's response, including tool call arguments, is automatically parsed into the `Query` object.
SOURCE: https://github.com/openai/openai-python/blob/main/helpers.md#_snippet_1

LANGUAGE: Python
CODE:
```
from enum import Enum
from typing import List, Union
from pydantic import BaseModel
import openai

class Table(str, Enum):
    orders = "orders"
    customers = "customers"
    products = "products"

class Column(str, Enum):
    id = "id"
    status = "status"
    expected_delivery_date = "expected_delivery_date"
    delivered_at = "delivered_at"
    shipped_at = "shipped_at"
    ordered_at = "ordered_at"
    canceled_at = "canceled_at"

class Operator(str, Enum):
    eq = "="
    gt = ">"
    lt = "<"
    le = "<="
    ge = ">="
    ne = "!="

class OrderBy(str, Enum):
    asc = "asc"
    desc = "desc"

class DynamicValue(BaseModel):
    column_name: str

class Condition(BaseModel):
    column: str
    operator: Operator
    value: Union[str, int, DynamicValue]

class Query(BaseModel):
    table_name: Table
    columns: List[Column]
    conditions: List[Condition]
    order_by: OrderBy

client = openai.OpenAI()
completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. The current date is August 6, 2024. You help users query for the data they are looking for by calling the query function.",
        },
        {
            "role": "user",
            "content": "look up all my orders in may of last year that were fulfilled but not delivered on time",
        },
    ],
    tools=[
        openai.pydantic_function_tool(Query),
    ],
)

tool_call = (completion.choices[0].message.tool_calls or [])[0]
print(tool_call.function)
assert isinstance(tool_call.function.parsed_arguments, Query)
print(tool_call.function.parsed_arguments.table_name)
```

----------------------------------------

TITLE: Handling Assistant API Stream Events with EventHandler (Python)
DESCRIPTION: This comprehensive example demonstrates how to create a custom `AssistantEventHandler` class to subscribe to and process various events from an OpenAI Assistant API stream, such as text creation, text deltas, and tool call details. It shows how to define overridden methods for specific event types and then use this handler with `client.beta.threads.runs.stream`.
SOURCE: https://github.com/openai/openai-python/blob/main/helpers.md#_snippet_5

LANGUAGE: Python
CODE:
```
from typing_extensions import override
from openai import AssistantEventHandler, OpenAI
from openai.types.beta.threads import Text, TextDelta
from openai.types.beta.threads.runs import ToolCall, ToolCallDelta

client = openai.OpenAI()

# First, we create a EventHandler class to define
# how we want to handle the events in the response stream.

class EventHandler(AssistantEventHandler):
  @override
  def on_text_created(self, text: Text) -> None:
    print(f"\nassistant > ", end="", flush=True)

  @override
  def on_text_delta(self, delta: TextDelta, snapshot: Text):
    print(delta.value, end="", flush=True)

  @override
  def on_tool_call_created(self, tool_call: ToolCall):
    print(f"\nassistant > {tool_call.type}\n", flush=True)

  @override
  def on_tool_call_delta(self, delta: ToolCallDelta, snapshot: ToolCall):
    if delta.type == "code_interpreter" and delta.code_interpreter:
      if delta.code_interpreter.input:
        print(delta.code_interpreter.input, end="", flush=True)
      if delta.code_interpreter.outputs:
        print(f"\n\noutput >", flush=True)
        for output in delta.code_interpreter.outputs:
          if output.type == "logs":
            print(f"\n{output.logs}", flush=True)

# Then, we use the `stream` SDK helper
# with the `EventHandler` class to create the Run
# and stream the response.

with client.beta.threads.runs.stream(
  thread_id="thread_id",
  assistant_id="assistant_id",
  event_handler=EventHandler(),
) as stream:
  stream.until_done()
```

----------------------------------------

TITLE: Handling OpenAI API Errors with Try-Except (Python)
DESCRIPTION: This snippet demonstrates robust error handling for OpenAI API calls. It catches specific exceptions like `APIConnectionError` for network issues, `RateLimitError` for 429 responses, and `APIStatusError` for other non-success HTTP status codes, providing different handling logic for each.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_16

LANGUAGE: Python
CODE:
```
import openai
from openai import OpenAI

client = OpenAI()

try:
    client.fine_tuning.jobs.create(
        model="gpt-4o",
        training_file="file-abc123",
    )
except openai.APIConnectionError as e:
    print("The server could not be reached")
    print(e.__cause__)  # an underlying Exception, likely raised within httpx.
except openai.RateLimitError as e:
    print("A 429 status code was received; we should back off a bit.")
except openai.APIStatusError as e:
    print("Another non-200-range status code was received")
    print(e.status_code)
    print(e.response)
```

----------------------------------------

TITLE: Generating Text with OpenAI Responses API (Python)
DESCRIPTION: This Python snippet demonstrates how to generate text using the new OpenAI Responses API. It initializes an OpenAI client, sets a model and instructions, and then sends a text input to the model to receive a generated response. The API key is retrieved from environment variables for security.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_1

LANGUAGE: python
CODE:
```
import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.responses.create(
    model="gpt-4o",
    instructions="You are a coding assistant that talks like a pirate.",
    input="How do I check if a Python object is an instance of a class?",
)

print(response.output_text)
```

----------------------------------------

TITLE: Using OpenAI Responses API Asynchronously (Python)
DESCRIPTION: This Python snippet demonstrates asynchronous interaction with the OpenAI API using AsyncOpenAI. It initializes an asynchronous client and uses await for API calls within an async function, enabling non-blocking operations suitable for concurrent applications. The asyncio.run() function executes the main asynchronous task.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_5

LANGUAGE: python
CODE:
```
import os
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


async def main() -> None:
    response = await client.responses.create(
        model="gpt-4o", input="Explain disestablishmentarianism to a smart five year old."
    )
    print(response.output_text)


asyncio.run(main())
```

----------------------------------------

TITLE: Connecting to Realtime API for Text Conversations (Python)
DESCRIPTION: This example demonstrates how to establish an asynchronous connection to the OpenAI Realtime API using `AsyncOpenAI` for text-based conversations. It shows how to update session modalities, send a user message, initiate a response, and stream text deltas until the conversation is complete. It requires the `openai` and `asyncio` libraries.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_8

LANGUAGE: python
CODE:
```
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI()

    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as connection:
        await connection.session.update(session={'modalities': ['text']})

        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hello!"}],
            }
        )
        await connection.response.create()

        async for event in connection:
            if event.type == 'response.text.delta':
                print(event.delta, flush=True, end="")

            elif event.type == 'response.text.done':
                print()

            elif event.type == "response.done":
                break

asyncio.run(main())
```

----------------------------------------

TITLE: Streaming Chat Completions with AsyncOpenAI Python SDK
DESCRIPTION: This asynchronous Python snippet demonstrates how to use the `AsyncOpenAI` client's `beta.chat.completions.stream()` method within a context manager to receive streamed chat completion events. It iterates through the stream, specifically printing `content.delta` events as they arrive, providing real-time output. This method ensures proper resource closure and offers a granular event API for processing different types of stream updates.
SOURCE: https://github.com/openai/openai-python/blob/main/helpers.md#_snippet_2

LANGUAGE: Python
CODE:
```
from openai import AsyncOpenAI

client = AsyncOpenAI()

async with client.beta.chat.completions.stream(
    model='gpt-4o-2024-08-06',
    messages=[...],
) as stream:
    async for event in stream:
        if event.type == 'content.delta':
            print(event.content, flush=True, end='')
```

----------------------------------------

TITLE: Streaming OpenAI Responses Synchronously (Python)
DESCRIPTION: This Python example shows how to stream responses from the OpenAI API using the synchronous client. By setting stream=True, the API returns a stream object that can be iterated over to receive events as they become available, which is useful for real-time output or long-running generations.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_6

LANGUAGE: python
CODE:
```
from openai import OpenAI

client = OpenAI()

stream = client.responses.create(
    model="gpt-4o",
    input="Write a one-sentence bedtime story about a unicorn.",
    stream=True,
)

for event in stream:
    print(event)
```

----------------------------------------

TITLE: Streaming OpenAI Responses Asynchronously (Python)
DESCRIPTION: This Python snippet demonstrates how to stream responses from the OpenAI API using the asynchronous client. Similar to the synchronous version, setting stream=True returns an asynchronous stream that can be iterated over with async for, allowing for non-blocking processing of streamed events.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_7

LANGUAGE: python
CODE:
```
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()


async def main():
    stream = await client.responses.create(
        model="gpt-4o",
        input="Write a one-sentence bedtime story about a unicorn.",
        stream=True,
    )

    async for event in stream:
        print(event)


asyncio.run(main())
```

----------------------------------------

TITLE: Integrating with Azure OpenAI using AzureOpenAI Class in Python
DESCRIPTION: This snippet demonstrates how to initialize and use the `AzureOpenAI` client for interacting with Azure OpenAI services. It shows how to configure `api_version` and `azure_endpoint`, and then make a chat completion request, highlighting the use of `model` as a deployment name specific to Azure.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_30

LANGUAGE: python
CODE:
```
from openai import AzureOpenAI

# gets the API Key from environment variable AZURE_OPENAI_API_KEY
client = AzureOpenAI(
    # https://learn.microsoft.com/azure/ai-services/openai/reference#rest-api-versioning
    api_version="2023-07-01-preview",
    # https://learn.microsoft.com/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
    azure_endpoint="https://example-endpoint.openai.azure.com",
)

completion = client.chat.completions.create(
    model="deployment-name",  # e.g. gpt-35-instant
    messages=[
        {
            "role": "user",
            "content": "How do I output all files in a directory using Python?"
        }
    ]
)
print(completion.to_json())
```

----------------------------------------

TITLE: Streaming API Response Body in OpenAI Python Client
DESCRIPTION: This Python snippet demonstrates how to stream the API response body using `.with_streaming_response` within a context manager. It allows processing the response incrementally, for example, by iterating over lines, without eagerly reading the full body.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_25

LANGUAGE: python
CODE:
```
with client.chat.completions.with_streaming_response.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-4o",
) as response:
    print(response.headers.get("X-My-Header"))

    for line in response.iter_lines():
        print(line)
```

----------------------------------------

TITLE: Performing Vision Tasks with Image URL (Python)
DESCRIPTION: This Python snippet demonstrates how to use the OpenAI Responses API for vision tasks by providing an image URL. It constructs an input array containing both text and an image URL, then sends it to a vision-capable model like gpt-4o-mini to get a description of the image content.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_3

LANGUAGE: python
CODE:
```
prompt = "What is in this image?"
img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/2023_06_08_Raccoon1.jpg/1599px-2023_06_08_Raccoon1.jpg"

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"{img_url}"},
            ],
        }
    ],
)
```

----------------------------------------

TITLE: Making Custom/Undocumented HTTP Requests with OpenAI Python Client
DESCRIPTION: This Python example shows how to make requests to undocumented API endpoints using generic HTTP methods like `client.post`. It demonstrates sending a custom body and casting the response to an `httpx.Response` object, respecting client options like retries.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_26

LANGUAGE: python
CODE:
```
import httpx

response = client.post(
    "/foo",
    cast_to=httpx.Response,
    body={"my_param": True},
)

print(response.headers.get("x-foo"))
```

----------------------------------------

TITLE: Creating Content Moderation in OpenAI Python
DESCRIPTION: This method checks content for policy violations using the OpenAI Moderation API. It accepts parameters such as the input text or image, returning a `ModerationCreateResponse` with flags for categories like hate speech or violence. This is essential for ensuring content safety and compliance.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_29

LANGUAGE: Python
CODE:
```
client.moderations.create(**params)
```

----------------------------------------

TITLE: Managing OpenAI HTTP Resources with Context Manager in Python
DESCRIPTION: This snippet demonstrates how to manage HTTP resources by using the `OpenAI` client as a context manager. This ensures that underlying HTTP connections are automatically closed when exiting the `with` block, preventing resource leaks and simplifying connection management.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_29

LANGUAGE: python
CODE:
```
from openai import OpenAI

with OpenAI() as client:
  # make requests here
  ...

# HTTP client is now closed
```

----------------------------------------

TITLE: Accessing Paginated Data Directly (Python)
DESCRIPTION: This snippet shows how to access paginated response data directly from the `first_page` object. It demonstrates retrieving the `after` cursor for manual pagination and iterating through the `data` attribute to process items on the current page. This approach is useful when only the current page's items are needed or for building custom pagination logic. This snippet is designed for asynchronous usage but notes its applicability to synchronous contexts by removing `await`.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_13

LANGUAGE: python
CODE:
```
first_page = await client.fine_tuning.jobs.list(
    limit=20,
)

print(f"next page cursor: {first_page.after}")  # => "next page cursor: ..."
for job in first_page.data:
    print(job.id)

# Remove `await` for non-async usage.
```

----------------------------------------

TITLE: Managing Chat Completions with OpenAI Python Client
DESCRIPTION: Provides a suite of methods for interacting with chat completions, including creation, retrieval, updating, listing, and deletion. These methods allow full lifecycle management of chat interactions via the OpenAI API.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_5

LANGUAGE: python
CODE:
```
client.chat.completions.create(**params) -> ChatCompletion
```

LANGUAGE: python
CODE:
```
client.chat.completions.retrieve(completion_id) -> ChatCompletion
```

LANGUAGE: python
CODE:
```
client.chat.completions.update(completion_id, **params) -> ChatCompletion
```

LANGUAGE: python
CODE:
```
client.chat.completions.list(**params) -> SyncCursorPage[ChatCompletion]
```

LANGUAGE: python
CODE:
```
client.chat.completions.delete(completion_id) -> ChatCompletionDeleted
```

----------------------------------------

TITLE: Creating a Run for a Thread (OpenAI Python)
DESCRIPTION: This method initiates a new execution run for a specified `thread_id`. It accepts `params` to configure the run, such as the Assistant to use, and returns a `Run` object representing the newly started execution.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_99

LANGUAGE: Python
CODE:
```
client.beta.threads.runs.create(thread_id, **params)
```

----------------------------------------

TITLE: Configuring Retry Behavior for OpenAI API Requests (Python)
DESCRIPTION: This example shows how to configure the `max_retries` setting for OpenAI API requests. It demonstrates setting a global default for the client and overriding it for specific requests using `with_options`, controlling how many times certain errors are automatically retried.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_19

LANGUAGE: Python
CODE:
```
from openai import OpenAI

# Configure the default for all requests:
client = OpenAI(
    # default is 2
    max_retries=0,
)

# Or, configure per-request:
client.with_options(max_retries=5).chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "How can I get the name of the current day in JavaScript?",
        }
    ],
    model="gpt-4o",
)
```

----------------------------------------

TITLE: Manual Pagination Control with `has_next_page` (Python)
DESCRIPTION: This example demonstrates how to manually control pagination using methods like `has_next_page()`, `next_page_info()`, and `get_next_page()`. It allows developers to explicitly check for the existence of subsequent pages and retrieve them, providing more granular control over the pagination process compared to automatic iteration. This snippet is designed for asynchronous usage but notes its applicability to synchronous contexts by removing `await`.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_12

LANGUAGE: python
CODE:
```
first_page = await client.fine_tuning.jobs.list(
    limit=20,
)
if first_page.has_next_page():
    print(f"will fetch next page using these details: {first_page.next_page_info()}")
    next_page = await first_page.get_next_page()
    print(f"number of items we just fetched: {len(next_page.data)}")

# Remove `await` for non-async usage.
```

----------------------------------------

TITLE: Performing Vision Tasks with Base64 Image (Python)
DESCRIPTION: This Python example shows how to perform vision tasks by encoding an image as a base64 string. It reads an image file, converts it to base64, and then includes it in the input_image field of the Responses API call, allowing the model to process local image files.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_4

LANGUAGE: python
CODE:
```
import base64
from openai import OpenAI

client = OpenAI()

prompt = "What is in this image?"
with open("path/to/image.png", "rb") as image_file:
    b64_image = base64.b64encode(image_file.read()).decode("utf-8")

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"},
            ],
        }
    ],
)
```

----------------------------------------

TITLE: Submitting Tool Outputs and Streaming (Python)
DESCRIPTION: This method is used to submit tool outputs to an Assistant API run that is currently waiting for them, and then continues to stream the run's response. It's essential for handling tool calls in a streaming context.
SOURCE: https://github.com/openai/openai-python/blob/main/helpers.md#_snippet_10

LANGUAGE: Python
CODE:
```
client.beta.threads.runs.submit_tool_outputs_stream()
```

----------------------------------------

TITLE: Retrieving Request IDs from OpenAI APIStatusError (Python)
DESCRIPTION: This snippet demonstrates how to access the `request_id` property from an `openai.APIStatusError` exception. This is essential for debugging failed API requests, allowing developers to log and report specific request identifiers.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_18

LANGUAGE: Python
CODE:
```
import openai

try:
    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Say this is a test"}], model="gpt-4"
    )
except openai.APIStatusError as exc:
    print(exc.request_id)  # req_123
    raise exc
```

----------------------------------------

TITLE: Creating and Streaming a Thread Run (OpenAI Python)
DESCRIPTION: This method creates a new run for a thread and streams real-time updates as the run progresses. It returns an `AssistantStreamManager` which allows for event-driven processing of the run's lifecycle.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_106

LANGUAGE: Python
CODE:
```
client.beta.threads.runs.create_and_stream(*args)
```

----------------------------------------

TITLE: Auto-parsing Response Content with Pydantic Models (Python)
DESCRIPTION: This snippet demonstrates how to use `client.beta.chat.completions.parse()` with Pydantic models to automatically define and parse structured JSON responses from the OpenAI API. It shows how to define a `MathResponse` model with nested `Step` objects, send a math problem to the model, and then access the parsed steps and final answer from the completion message.
SOURCE: https://github.com/openai/openai-python/blob/main/helpers.md#_snippet_0

LANGUAGE: Python
CODE:
```
from typing import List
from pydantic import BaseModel
from openai import OpenAI

class Step(BaseModel):
    explanation: str
    output: str

class MathResponse(BaseModel):
    steps: List[Step]
    final_answer: str

client = OpenAI()
completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "solve 8x + 31 = 2"},
    ],
    response_format=MathResponse,
)

message = completion.choices[0].message
if message.parsed:
    print(message.parsed.steps)
    print("answer: ", message.parsed.final_answer)
else:
    print(message.refusal)
```

----------------------------------------

TITLE: Iterating Over All Assistant API Stream Events (Python)
DESCRIPTION: This snippet demonstrates how to iterate directly over all events received from an OpenAI Assistant API stream. It provides a generic way to access different event types and their data, specifically showing how to extract text content from `thread.message.delta` events.
SOURCE: https://github.com/openai/openai-python/blob/main/helpers.md#_snippet_6

LANGUAGE: Python
CODE:
```
with client.beta.threads.runs.stream(
  thread_id=thread.id,
  assistant_id=assistant.id
) as stream:
    for event in stream:
        # Print the text from text delta events
        if event.event == "thread.message.delta" and event.data.delta.content:
            print(event.data.delta.content[0].text)
```

----------------------------------------

TITLE: Creating an Assistant in Python
DESCRIPTION: This method creates a new assistant. It accepts configuration parameters (`params`) and returns an `Assistant` object, which represents the newly created AI assistant.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_85

LANGUAGE: python
CODE:
```
client.beta.assistants.create(**params) -> Assistant
```

----------------------------------------

TITLE: Setting Timeouts for OpenAI API Requests (Python)
DESCRIPTION: This snippet demonstrates how to configure request timeouts for the OpenAI client. It shows setting a simple float for a global timeout and using an `httpx.Timeout` object (requires `httpx` library) for more granular control over connect, read, and write timeouts.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_20

LANGUAGE: Python
CODE:
```
from openai import OpenAI

# Configure the default for all requests:
client = OpenAI(
    # 20 seconds (default is 10 minutes)
    timeout=20.0,
)

# More granular control:
client = OpenAI(
    timeout=httpx.Timeout(60.0, read=5.0, write=10.0, connect=2.0),
)
```

----------------------------------------

TITLE: Creating Text Completions with OpenAI Python Client
DESCRIPTION: Initiates a text completion request using the OpenAI API. This method sends parameters to the `/completions` endpoint and returns a `Completion` object, representing the generated text and associated details.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_2

LANGUAGE: python
CODE:
```
client.completions.create(**params) -> Completion
```

----------------------------------------

TITLE: Creating a File in OpenAI Python
DESCRIPTION: This method uploads a file to the OpenAI API. It accepts parameters such as the file content and purpose, returning a `FileObject` upon successful creation. This is used for preparing files for use with other OpenAI services like fine-tuning or assistants.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_10

LANGUAGE: Python
CODE:
```
client.files.create(**params)
```

----------------------------------------

TITLE: Polling Asynchronous API Operations (Python)
DESCRIPTION: These methods provide helper functions to poll the status of asynchronous API actions until they reach a terminal state, then return the resulting object. They simplify handling operations like creating runs or managing vector store files, which typically take time to complete. Polling frequency can be adjusted via poll_interval_ms.
SOURCE: https://github.com/openai/openai-python/blob/main/helpers.md#_snippet_22

LANGUAGE: python
CODE:
```
client.beta.threads.create_and_run_poll(...)
client.beta.threads.runs.create_and_poll(...)
client.beta.threads.runs.submit_tool_outputs_and_poll(...)
client.beta.vector_stores.files.upload_and_poll(...)
client.beta.vector_stores.files.create_and_poll(...)
client.beta.vector_stores.file_batches.create_and_poll(...)
client.beta.vector_stores.file_batches.upload_and_poll(...)
```

----------------------------------------

TITLE: Creating a Realtime Session in Python
DESCRIPTION: This method allows for the creation of a new real-time session. It accepts parameters (`params`) to configure the session and returns a `SessionCreateResponse` object upon successful creation. This is the entry point for initiating real-time interactions.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_81

LANGUAGE: python
CODE:
```
client.beta.realtime.sessions.create(**params) -> SessionCreateResponse
```

----------------------------------------

TITLE: Accessing Raw HTTP Response Data in OpenAI Python Client
DESCRIPTION: This Python example illustrates how to access the raw HTTP response object, including headers, by prefixing API calls with `.with_raw_response`. It then demonstrates parsing the response to get the standard completion object.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_24

LANGUAGE: python
CODE:
```
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.with_raw_response.create(
    messages=[{
        "role": "user",
        "content": "Say this is a test",
    }],
    model="gpt-4o",
)
print(response.headers.get('X-My-Header'))

completion = response.parse()  # get the object that `chat.completions.create()` would have returned
print(completion)
```

----------------------------------------

TITLE: Subscribing to ToolCall Lifecycle Events (Python)
DESCRIPTION: These methods enable subscription to events related to the creation, incremental updates (delta), and completion of tool calls made by the Assistant. They are essential for monitoring and interacting with the assistant's use of external tools.
SOURCE: https://github.com/openai/openai-python/blob/main/helpers.md#_snippet_16

LANGUAGE: python
CODE:
```
def on_tool_call_created(self, tool_call: ToolCall)
def on_tool_call_delta(self, delta: ToolCallDelta, snapshot: ToolCall)
def on_tool_call_done(self, tool_call: ToolCall)
```

----------------------------------------

TITLE: Creating Embeddings with OpenAI Python Client
DESCRIPTION: Generates embeddings for input text or data using the OpenAI API. This method sends parameters to the `/embeddings` endpoint and returns a `CreateEmbeddingResponse` object containing the generated embedding vectors.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_8

LANGUAGE: python
CODE:
```
client.embeddings.create(**params) -> CreateEmbeddingResponse
```

----------------------------------------

TITLE: Creating a Thread (OpenAI Python)
DESCRIPTION: This method creates a new conversational thread in the OpenAI API. It accepts `params` to configure the thread, such as initial messages or metadata, and returns a `Thread` object representing the newly created thread.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_91

LANGUAGE: Python
CODE:
```
client.beta.threads.create(**params)
```

----------------------------------------

TITLE: Using Nested Parameters with OpenAI Chat Completions (Python)
DESCRIPTION: This snippet demonstrates how to pass nested dictionary parameters, specifically for the `input` field in a chat completion request. It shows how to define user and content roles within the input list, and how to specify a `response_format` for JSON output.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_14

LANGUAGE: Python
CODE:
```
from openai import OpenAI

client = OpenAI()

response = client.chat.responses.create(
    input=[
        {
            "role": "user",
            "content": "How much ?",
        }
    ],
    model="gpt-4o",
    response_format={"type": "json_object"},
)
```

----------------------------------------

TITLE: Generating Images in OpenAI Python
DESCRIPTION: This method generates new images from a text prompt using the OpenAI API. It accepts parameters like the prompt, desired size, and number of images, returning an `ImagesResponse`. This is the primary method for creating images from scratch based on textual descriptions.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_20

LANGUAGE: Python
CODE:
```
client.images.generate(**params)
```

----------------------------------------

TITLE: Creating a Message in an OpenAI Thread (Python)
DESCRIPTION: This method adds a new message to a specified thread. It requires the `thread_id` and accepts various parameters to define the message content and properties, returning a `Message` object upon successful creation.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_114

LANGUAGE: python
CODE:
```
client.beta.threads.messages.create(thread_id, **params) -> Message
```

----------------------------------------

TITLE: Searching a Vector Store in OpenAI Python
DESCRIPTION: This method performs a search operation within a specified vector store using the OpenAI client. It requires `vector_store_id` and `params` for the search query, returning a `SyncPage` of `VectorStoreSearchResponse` objects.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_59

LANGUAGE: python
CODE:
```
client.vector_stores.search(vector_store_id, **params) -> SyncPage[VectorStoreSearchResponse]
```

----------------------------------------

TITLE: Creating and Polling a Vector Store File in Python
DESCRIPTION: Creates a new file and then polls its status until it's processed. Accepts arguments via `*args`. Returns the `VectorStoreFile` object once processing is complete.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_67

LANGUAGE: python
CODE:
```
client.vector_stores.files.create_and_poll(*args) -> VectorStoreFile
```

----------------------------------------

TITLE: Retrieving Final Chat Completion (Python)
DESCRIPTION: This snippet demonstrates how to retrieve the accumulated `ParsedChatCompletion` object from a chat completions stream after it has finished processing. It requires an active `client.beta.chat.completions.stream` instance.
SOURCE: https://github.com/openai/openai-python/blob/main/helpers.md#_snippet_3

LANGUAGE: Python
CODE:
```
async with client.beta.chat.completions.stream(...) as stream:
    ...

completion = await stream.get_final_completion()
print(completion.choices[0].message)
```

----------------------------------------

TITLE: Creating Audio Transcriptions in OpenAI Python
DESCRIPTION: This method transcribes audio into text using the OpenAI API. It accepts parameters such as the audio file and model, returning a `TranscriptionCreateResponse`. This is used for converting spoken language from audio files into written text.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_23

LANGUAGE: Python
CODE:
```
client.audio.transcriptions.create(**params)
```

----------------------------------------

TITLE: Streaming an Existing Run (OpenAI Python)
DESCRIPTION: This method streams real-time updates for an existing run identified by `run_id` within a `thread_id`. It provides an `AssistantStreamManager` for event-driven processing of the run's status and output.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_108

LANGUAGE: Python
CODE:
```
client.beta.threads.runs.stream(*args)
```

----------------------------------------

TITLE: Setting Per-Request Timeout in OpenAI Python Client
DESCRIPTION: This snippet shows how to override the default timeout for a specific API request using the `with_options` method. It sets a 5-second timeout for a chat completion request, throwing an `APITimeoutError` if exceeded.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_21

LANGUAGE: python
CODE:
```
client.with_options(timeout=5.0).chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "How can I list all files in a directory using Python?",
        }
    ],
    model="gpt-4o",
)
```

----------------------------------------

TITLE: Asynchronous Auto-Pagination for List Methods (Python)
DESCRIPTION: This snippet shows asynchronous auto-pagination for OpenAI API list methods using `AsyncOpenAI`. Similar to its synchronous counterpart, iterating with `async for` over the list response automatically handles fetching all pages, making it suitable for non-blocking applications. It requires `asyncio` for execution.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_11

LANGUAGE: python
CODE:
```
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()


async def main() -> None:
    all_jobs = []
    # Iterate through items across all pages, issuing requests as needed.
    async for job in client.fine_tuning.jobs.list(
        limit=20,
    ):
        all_jobs.append(job)
    print(all_jobs)


asyncio.run(main())
```

----------------------------------------

TITLE: Generating Text with OpenAI Chat Completions API (Python)
DESCRIPTION: This Python example illustrates text generation using the legacy Chat Completions API, which is still supported. It initializes an OpenAI client and sends a list of messages with different roles to the specified model to get a conversational response. The generated content is then printed from the completion object.
SOURCE: https://github.com/openai/openai-python/blob/main/README.md#_snippet_2

LANGUAGE: python
CODE:
```
from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "developer", "content": "Talk like a pirate."},
        {
            "role": "user",
            "content": "How do I check if a Python object is an instance of a class?",
        },
    ],
)

print(completion.choices[0].message.content)
```

----------------------------------------

TITLE: Creating and Running a Stream with New Message (Python)
DESCRIPTION: This method combines adding a message to a thread, starting a new Assistant API run, and then streaming the response. It's a convenience method for initiating a new interaction with streaming.
SOURCE: https://github.com/openai/openai-python/blob/main/helpers.md#_snippet_9

LANGUAGE: Python
CODE:
```
client.beta.threads.create_and_run_stream()
```

----------------------------------------

TITLE: Creating and Polling a Thread Run (OpenAI Python)
DESCRIPTION: This method creates a new run for a thread and then synchronously polls for its completion. It's a convenience for blocking operations, returning the final `Run` object once execution finishes.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_105

LANGUAGE: Python
CODE:
```
client.beta.threads.runs.create_and_poll(*args)
```

----------------------------------------

TITLE: Creating a Vector Store File in Python
DESCRIPTION: Creates a new file within a specified vector store. Requires `vector_store_id` and accepts additional parameters via `**params`. Returns a `VectorStoreFile` object upon successful creation.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_61

LANGUAGE: python
CODE:
```
client.vector_stores.files.create(vector_store_id, **params) -> VectorStoreFile
```

----------------------------------------

TITLE: Submitting Tool Outputs and Polling Run (OpenAI Python)
DESCRIPTION: This method submits tool outputs to a run and then synchronously polls for its completion. It's a convenience for blocking operations after providing tool results, returning the final `Run` object.
SOURCE: https://github.com/openai/openai-python/blob/main/api.md#_snippet_109



----------------------------------------

TITLE: Retrieving Final Accumulated Objects from Assistant Stream (Python)
DESCRIPTION: These methods are convenience functions to collect and return the final accumulated objects (Run, RunSteps, Messages) after a stream has completed. Calling them will consume the entire stream until its completion, then provide the full, final state of the requested objects.
SOURCE: https://github.com/openai/openai-python/blob/main/helpers.md#_snippet_21

LANGUAGE: python
CODE:
```
def get_final_run(self) -> Run
def get_final_run_steps(self) -> List[RunStep]
def get_final_messages(self) -> List[Message]
```
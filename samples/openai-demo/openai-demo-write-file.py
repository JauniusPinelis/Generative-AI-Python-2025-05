import os
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()  # take environment variables

# from .env file
# Load environment variables from .env file

token = os.getenv("SECRET")  # Replace with your actual token
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

def setup_client():
    # initialize the OpenAI client
    # Note: The OpenAI client is initialized with the base URL and API key.
    return OpenAI(
        base_url=endpoint,
        api_key=token,
    )

def generate_joke(client):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Only answer in Lithuanian."
                "Answer in super polite way and use the word 'prašome' in your answer."
                "Do not answer any others questions, just about Vilnius.",
            },
            {
                "role": "user",
                "content": "Tell me a better joke about Vilnius.",
            }
        ],
        temperature=1.0,
        top_p=1.0,
        model=model
    )

    return response.choices[0].message.content

def write_to_file(joke):
    # Write the joke to a file
    with open("joke.txt", "w") as file:
        file.write(joke)  # type: ignore

client = setup_client()

joke = generate_joke(client)

write_to_file(joke)
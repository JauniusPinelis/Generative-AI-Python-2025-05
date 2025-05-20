import os
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()  # take environment variables

def print_hello():
    hello_world = "Hello, world!"
    print(hello_world)

def print_hello_parameters(hello_world):
    print(hello_world)

def print_hello_return():
    return "Hello, world jaunius!"


print_hello_parameters("Hello, world 2!")
print_hello()

print_hello_parameters("Hello, world Karoli!")

response = print_hello_return()
print(response)



# # from .env file
# # Load environment variables from .env file

# token = os.getenv("SECRET")  # Replace with your actual token
# endpoint = "https://models.github.ai/inference"
# model = "openai/gpt-4.1-nano"

# # initialize the OpenAI client
# # Note: The OpenAI client is initialized with the base URL and API key.
# client = OpenAI(
#     base_url=endpoint,
#     api_key=token,
# )

# response = client.chat.completions.create(
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a helpful assistant.",
#         },
#         {
#             "role": "user",
#             "content": "Tell me a better joke about Vilnius.",
#         }
#     ],
#     temperature=1.0,
#     top_p=1.0,
#     model=model
# )

# joke = response.choices[0].message.content

# # Write the joke to a file
# with open("joke.txt", "w") as file:
#     file.write(joke) # type: ignore
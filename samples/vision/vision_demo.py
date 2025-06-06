# This software will identifies if the image is a valid id.

import os
from google import genai
from pydantic import BaseModel
from rich import print

from dotenv import load_dotenv

load_dotenv()

class IsIdValidResponse(BaseModel):
    valid_id: bool = False
    reasons: list[str] = []

api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

my_file = client.files.upload(file="C:\\Users\\Jauni\\projects\\Generative-AI-Python-2025-05\\samples\\vision\\id3.jpg")

response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20",
    contents=[my_file, """
                You are a fake id scanner. You job is to process Lithuanian Ids.
                Only pass the id if it is a valid Lithuanian ID.
                The valid lithuanian id should not be expired, should have a valid photo, and should not be damaged.
                Ensure the id is not a sample id, but a real id.
                If you are not sure, return a JSON object with the key "valid_id" set to false.
                If id is blurry or not readable, return a JSON object with the key "valid_id" set to false.
                The id should contain a visible photo of a person with a clear face.
                Personal code, name, surname, expiration date and photo should be visible and readable.
                """
              ],
    config={
        "response_mime_type": "application/json",
        "response_schema": IsIdValidResponse,
    },
)

is_valid_response: IsIdValidResponse = response.parsed # type: ignore

if is_valid_response.valid_id:
    print("The image is a valid Lithuanian ID.")
else:
    print("The image is not a valid Lithuanian ID.")

print(response.text)
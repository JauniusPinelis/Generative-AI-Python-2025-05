# a fastapi system for personal_prompt managament

import os
from fastapi import FastAPI
import openai
from pydantic import BaseModel

from dotenv import load_dotenv
from rich import print

load_dotenv()  # Load environment variables from .env file

token = os.getenv("SECRET")  # Replace with your actual token
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

app = FastAPI()

client = openai.OpenAI(
    api_key=token,
    base_url=endpoint)

class PersonalPrompt(BaseModel):
    id: int
    prompt: str

class EnterprisePrompt(BaseModel):
    id: int
    prompt: str

class PersonalPromptReview(BaseModel):
    is_approved: bool
    review_comments: str

# type hint for a list of personal_prompts
personal_prompts:list[PersonalPrompt] = []

enterprise_prompts:list[EnterprisePrompt] = []

def review_personal_prompt(personal_prompt: PersonalPrompt):
    # Placeholder for review logic

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": """
             You are a helpful assistant that will review personal prompts.
             Personal prompt is a prompt that is used by a user to interact with the model.
             You will be used to ensure no harmful content or bad-mouthed content is present.
             Please also dont approve if the prompt has spelling mistakes or grammatical errors.
             Please provide review_comments."""},
            {"role": "user", "content": personal_prompt.prompt}
        ],
        temperature=0.7,
        response_format=PersonalPromptReview
    )

    parsed_response = response.choices[0].message.parsed

    return parsed_response



@app.post("/personal_prompts/")
def create_personal_prompt(personal_prompt: PersonalPrompt):
    personal_prompts.append(personal_prompt)
    return {"message": "personal_prompt created successfully"}

@app.get("/personal_prompts/")
def get_personal_prompts():
    return personal_prompts

@app.get("/personal_prompts/{personal_prompt_id}")
def get_personal_prompt(personal_prompt_id: int):
    for personal_prompt in personal_prompts:
        if personal_prompt.id == personal_prompt_id:
            return personal_prompt
    return {"message": "personal_prompt not found"}

@app.delete("/personal_prompts/{personal_prompt_id}")
def delete_personal_prompt(personal_prompt_id: int):
    personal_prompts[:] = [personal_prompt for personal_prompt in personal_prompts if personal_prompt.id != personal_prompt_id]
    return {"message": "personal_prompt deleted successfully"}

@app.post("/personal_prompts/{personal_prompt_id}/promote_to_enterprise")
def promote_to_enterprise(personal_prompt_id: int):

    personal_prompt = next((prompt for prompt in personal_prompts if prompt.id == personal_prompt_id), None)

    review_result = review_personal_prompt(personal_prompt)

    if review_result.is_approved:
        enterprise_prompt = EnterprisePrompt(id=personal_prompt.id, prompt=personal_prompt.prompt)
        enterprise_prompts.append(enterprise_prompt)
        return {"message": "Personal prompt promoted to enterprise successfully", "review_comments": review_result.review_comments}
    else:
        return {"message": "Personal prompt not approved for enterprise", "review_comments": review_result.review_comments}
    
@app.get("/enterprise_prompts/")
def get_enterprise_prompts():
    return enterprise_prompts
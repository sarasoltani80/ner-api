from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv


load_dotenv()


class Entity(BaseModel):
    """Represents a single named entity."""
    text: str = Field(description="The extracted text span of the entity")
    type: str = Field(description="The type of entity (PERSON, ORGANIZATION, LOCATION, DATE, etc.)")


class NEROutput(BaseModel):
    """The structured output containing all extracted entities."""
    entities: List[Entity] = Field(description="List of extracted named entities")



parser = PydanticOutputParser(pydantic_object=NEROutput)


prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert Named Entity Recognition (NER) system. 
Your task is to extract named entities from the provided text and categorize them.

Extract the following entity types:
- PERSON: Names of people (e.g., "John Doe", "Mary Smith")
- ORGANIZATION: Companies, institutions, agencies (e.g., "Acme Inc.", "United Nations")
- LOCATION: Countries, cities, states, addresses (e.g., "London", "New York", "123 Main St")
- DATE: Absolute or relative dates (e.g., "next Tuesday", "January 1, 2024", "tomorrow")
- PRODUCT: Named products or services (e.g., "iPhone", "Windows 10")
- TITLE: Job titles or positions (e.g., "CEO", "President", "Engineer")
- MONEY: Monetary values (e.g., "$100", "50 euros")
- TIME: Times of day (e.g., "3:00 PM", "noon")
- EVENT: Named events (e.g., "World Cup", "Christmas")

Important guidelines:
1. Extract only the specific entity text, not surrounding words
2. Each entity should appear only once in the output
3. Be precise with entity boundaries
4. If no entities are found, return an empty entities list

{format_instructions}"""),
    ("user", "{text}")
])


prompt = prompt_template.partial(format_instructions=parser.get_format_instructions())


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")



llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    openai_api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_API_BASE,
    temperature=0
)


ner_chain = prompt | llm | parser


app = FastAPI(
    title="Named Entity Recognition API",
    description="Extract named entities from text using LLM-powered NER",
    version="1.0.0"
)


add_routes(
    app,
    ner_chain,
    path="/ner",
    input_type=dict,
    output_type=NEROutput
)


if __name__ == "__main__":
    print("Starting NER API server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Playground: http://localhost:8000/ner/playground")
    print("\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
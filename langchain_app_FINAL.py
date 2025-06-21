from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# Load .env file
load_dotenv()

# Load LLM
model = ChatGroq(model="llama3-8b-8192")
# Alternative:
# model = ChatOpenAI(model="gpt-4o")

# Read OCR-extracted text from file
with open("ocr_output_demo2.txt", "r", encoding="utf-8") as f:
    input_text = f.read()

# --- PROMPTS ---
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a document summarizer."),
    ("human", "Summarize the following text:\n\n{document_text}")
])

entity_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at extracting information from text."),
    ("human", "Extract all key entities (names, dates, places, terms, etc.) from this text:\n\n{document_text}")
])

# --- DEFINE BRANCH FUNCTIONS ---
def prepare_summary_input(document_text):
    return summary_prompt.format_prompt(document_text=document_text)

def prepare_entity_input(document_text):
    return entity_prompt.format_prompt(document_text=document_text)

# --- CHAINS FOR PARALLEL PROCESSING ---
summary_chain = (
    RunnableLambda(lambda x: prepare_summary_input(x)) |
    model |
    StrOutputParser()
)

entity_chain = (
    RunnableLambda(lambda x: prepare_entity_input(x)) |
    model |
    StrOutputParser()
)

# --- FINAL COMBINER ---
def combine_summary_and_entities(summary, entities):
    return f"Summary:\n{summary}\n\n Key Entities:\n{entities}"

# --- FULL PARALLEL CHAIN ---
pipeline = (
    RunnableLambda(lambda x: x["ocr_text"]) |
    RunnableParallel({
        "summary": summary_chain,
        "entities": entity_chain
    }) |
    RunnableLambda(lambda x: combine_summary_and_entities(x["summary"], x["entities"]))
)

# --- RUN ---
if __name__ == "__main__":
    result = pipeline.invoke({"ocr_text": input_text})
    print(result)

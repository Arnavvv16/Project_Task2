from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# Load .env file
load_dotenv()

# Load LLM (choose one)
model = ChatGroq(model="llama3-8b-8192")
# model = ChatOpenAI(model="gpt-4o")

# Read OCR-extracted text
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

# --- LCEL CHAINS ---

# Summary chain
summary_chain = (
    {"document_text": lambda x: x["ocr_text"]}
    | summary_prompt
    | model
    | StrOutputParser()
)

# Entity extraction chain
entity_chain = (
    {"document_text": lambda x: x["ocr_text"]}
    | entity_prompt
    | model
    | StrOutputParser()
)

# Parallel execution
pipeline = (
    RunnableParallel({
        "summary": summary_chain,
        "entities": entity_chain
    })
    | (lambda x: f"📄 Summary:\n{x['summary']}\n\n🔍 Key Entities:\n{x['entities']}")
)

# --- RUN ---
if __name__ == "__main__":
    result = pipeline.invoke({"ocr_text": input_text})
    print(result)

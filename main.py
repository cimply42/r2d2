import os
from dotenv import load_dotenv
from flask import Flask, request
from pinecone import Pinecone, ServerlessSpec
from langchain_openai.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

parser = StrOutputParser()

template = """
Answer the question based on the context below.

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

app = Flask(__name__)

def get_assistant_chain():
  pinecone_index_name = "peronsal-assistant-index"
  if pinecone_index_name not in pc.list_indexes().names():
    pc.create_index(
      name=pinecone_index_name,
      dimension=1536,
      metric='cosine',
      spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
      )
    )

  loader = TextLoader("input.txt")
  text_documents = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
  text_chunks = text_splitter.split_documents(text_documents)

  embeddings = OpenAIEmbeddings()
  pinecone = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=pinecone_index_name)

  chain = (
    {"context": pinecone.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | model
    | parser
  )

  return chain

chain = get_assistant_chain()

@app.post("/api/prompt")
def prompt_assistant():
  if 'prompt' not in request.json:
    return 'Invalid request. Please provide prompt parameter', 400

  prompt = request.json['prompt']
  assistant_response = chain.invoke(prompt)
  response = {"response": assistant_response}
  return response, 200

if __name__ == '__main__':
  app.run(debug=True, port=5000)
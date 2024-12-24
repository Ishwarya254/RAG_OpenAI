
### Retrieval Augmented Generation (RAG) for a QA bot for a Business, leveraging the OpenAI API and a vector database (Pinecone DB).

# Install all the required Libraries


# Commented out IPython magic to ensure Python compatibility.
# %pip install -qU pypdf
# %pip install pinecone-client
# %pip install langchain-community
# %pip install langchain-huggingface
# %pip install langchain-pinecone
# %pip install protoc-gen-openapiv2
# %pip install langchain-openai

import os
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI

# Access API keys from your .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Load your pdf

loader = PyPDFLoader('file path for the pdf')
pages = loader.load_and_split()

#Initialize the language model with the GPT-4o-mini model for generating responses

llm = ChatOpenAI(model="gpt-4o-mini")

pages[0]

#Split the document into smaller chunks of 1000 characters with a 200-character overlap to prepare for embedding

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

docs = text_splitter.split_documents(pages)
len(docs)

#Generate embeddings for the document chunks using the HuggingFace model

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#Create a new Pinecone index with required dimensions, using cosine similarity as the metric

# To create  a new index
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

pc = Pinecone(api_key="Enter your pinecone API")

pc.create_index(
  name="your index name",
  dimension=384,
  metric="cosine",
  spec=ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  ),
  deletion_protection="disabled"
)

index = pc.Index("your index name")

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

index = pc.Index(index_name)

vector_store = PineconeVectorStore(embedding=embeddings, index=index)

# Creating a Retrieval-based Question Answering (QA) chain with sources using the initialized language model (LLM) and the vector store retriever

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = vector_store.as_retriever())

chain


query = "Type your question here!"

langchain.debug = True

chain({"question": query}, return_only_outputs=True)


"""The delay in receiving the expected output is due to a rate limit error encountered while accessing the OpenAI API (or other relevant API).

As I have exceeded the request limit set by the API, this error appears.


To avoid this error, either we can use ChatGroq or Ollama else can upgarde to OpenAI paid service
"""
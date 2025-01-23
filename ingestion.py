import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":

    loader = TextLoader("mediumblog1.txt", encoding='utf-8')
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents=document)

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    PineconeVectorStore.from_documents(texts, embedding_model, index_name=os.environ['INDEX_NAME'])

    
    print("Data indexed...")
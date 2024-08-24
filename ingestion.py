import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters.character import CharacterTextSplitter
from openai import embeddings

load_dotenv()


def load_text(file_path):
    print("Loading file...")
    loader = TextLoader(file_path)
    document = loader.load()
    return document


def split_text(document, chunk_size=1000, chunk_overlap=0, separator="\n"):
    print("Splitting...")
    text_splitter = CharacterTextSplitter(
        separator=separator, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")
    return texts


def generate_embeddings():
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return embeddings


def pinecone_ingest(texts, embeddings):
    """Ingresa los documentos a Pinecone."""
    print("Pinecone ingesting...")
    PineconeVectorStore.from_documents(
        documents=texts,
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings,
    )
    print("Finish!")


def main():
    document = load_text("./mediumblog1.txt")
    texts = split_text(document)
    embeddings = generate_embeddings()
    pinecone_ingest(texts, embeddings)


if __name__ == "__main__":
    main()

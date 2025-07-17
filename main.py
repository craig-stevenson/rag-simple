from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import init_chat_model
import bs4
from dotenv import load_dotenv


def load_data()->list[Document]:
    paths = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)
    bs = dict(parse_only=bs4.SoupStrainer( class_=("post-content", "post-title", "post-header")))
    loader = WebBaseLoader(web_paths=paths, bs_kwargs=bs)
    docs = loader.load()
    return docs

def split(docs: list[Document])->list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks

def store_chunks(chunks):
    vector_store = Chroma(
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"), 
        persist_directory="./chroma"
    )
    vector_store.add_documents(documents=chunks)
    return vector_store

def load_vector_store():
    vector_store = Chroma(
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"), 
        persist_directory="./chroma"
    )
    return vector_store

def create_database():
    print("Loading data")
    docs = load_data()

    print("splittin data")
    chunks = split(docs)

    print("creating vector store")
    vector_store = store_chunks(chunks)
    return vector_store
    

if __name__ == "__main__":
    load_dotenv(override=True)

    #the database only needs to be created once after that you can load it from disk
    vector_store = create_database()

    #vector_store = load_vector_store()

    print("retrieve data")
    question = "Who is the author of this blog post?"
    related_docs = vector_store.similarity_search(question)
    formatted_context = "\n\n".join(doc.page_content for doc in related_docs)

    print("augment prompt")
    augmented_prompt = f"""
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.
    Question: {question} 
    Context: {formatted_context}
    """
    
    print("asking LLM")
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    response = llm.invoke(augmented_prompt)

    print(response.content)

    
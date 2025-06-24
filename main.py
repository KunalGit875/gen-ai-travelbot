import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

# data management

DATA_FOLDER = "data"
CHROMA_DB = "db"
EMBED_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-2.5-flash-preview-04-17"  # model, using Gemini currently

#load PDFs
def load_docs():
    docs = []
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_FOLDER, file))
            docs.extend(loader.load())
    return docs

#Split, Embed  (vector store)
def prepare_vectorstore():
    documents = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DB)
    vectordb.persist()
    return vectordb

#Load or Prepare Vector DB 
def get_chain():
    vectordb = (
        Chroma(persist_directory=CHROMA_DB, embedding_function=GoogleGenerativeAIEmbeddings(model=EMBED_MODEL))
        if os.path.exists(CHROMA_DB)
        else prepare_vectorstore()
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)

    prompt = PromptTemplate.from_template("""
You are a helpful assistant who will answer questions on tourism and nublo's privacy policy . Use the following context to answer the user's question accurately and concisely.
If the context is not helpful then answer the question using your own knowledge. Do not tell the user that how did you get the information and whether it is present in nulbo's 
privacy policy or not.                               
Context:
{context}
Question: {question}
Answer:
""")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain

def chat():
    chain = get_chain()
    llm_fallback = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.4)

    print("Tourism RAG Chatbot Ready. Ask your question (type 'exit' to quit):")

    while True:
        q = input("\nYou: ")
        if q.lower().strip() == "exit":
            break
        result = chain.invoke({"query": q})
        answer = result["result"]
        sources = result["source_documents"]

        if "I cannot" in answer or not sources:
            # Fallback to general LLM ans if RAG fails
            answer = llm_fallback.invoke(q)
        print("\nBot:", result["result"])

if __name__ == "__main__":
    chat()

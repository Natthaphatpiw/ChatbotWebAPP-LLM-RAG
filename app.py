from flask import Flask, render_template, jsonify, request
from src.helper import download_embedding_model
# ✅ แบบใหม่ (ใช้ได้กับ langchain>=0.1.0 และ langchain-community)
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

from langchain_openai import AzureChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.environ.get('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT

# embedding = download_embedding_model()

index_name = "vetbot"

# docsearch = PineconeVectorStore.from_existing_index(
#  index_name= index_name,
#  embedding= embedding
# )

# retriever = docsearch.as_retriever(search_type = "similarity" , search_kwargs = {"k":5})
# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
#     huggingfacehub_api_token=HUGGINGFACE_API_KEY,
#     temperature=0.4,
#     max_new_tokens=500
# )

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}")
#     ]
# )

# question_answer = create_stuff_documents_chain(llm, prompt)
# rag = create_retrieval_chain(retriever, question_answer)

def get_rag_chain():
    embedding = download_embedding_model()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name="vetbot",
        embedding=embedding
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.4,
        max_tokens=500
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    question_answer = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer)
    return rag_chain


@app.route("/")
def index():
    return render_template('index.html')

# @app.route("/get", methods=["GET","POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     response = rag.invoke({"input": msg})
#     print("Response: ", response["answer"])
#     return str(response["answer"])

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    try:
        rag = get_rag_chain()
        response = rag.invoke({"input": msg})
        return str(response["answer"])
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return "Internal error"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


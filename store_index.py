from src.helper import load_pdf_file,text_split,download_embedding_model
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY

extrac_data = load_pdf_file(data= 'Data/')
text_chunk = text_split(extrac_data)
embedding = download_embedding_model()

pc = Pinecone(api_key= PINECONE_API_KEY)

index_name = "vetbot"

pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

docsearch = PineconeVectorStore.from_documents(
 documents= text_chunk,
 index_name= index_name,
 embedding= embedding
)

docsearch = PineconeVectorStore.from_existing_index(
 index_name= index_name,
 embedding= embedding
)

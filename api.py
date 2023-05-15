from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
import pinecone
import os.path
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
emdedding_model_name = os.getenv("emdedding_model_name")
pinecone_api_key = os.getenv("pinecone_api_key")
pinecone_environment = os.getenv("pinecone_environment")

app = FastAPI()

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

class Params(BaseModel):
    prompt: str = '你好'
    history: list[list[str]] = []
    max_length: int = 2048
    top_p: float = 0.7
    temperature: float = 0.95
    index_name: str = 'test'


async def create_chat(params: Params):
    global model , tokenizer
    response,history = model.chat(tokenizer,
                                  params.prompt,
                                  history=params.history,
                                  max_length=params.max_length,
                                  top_p=params.top_p,
                                  temperature=params.temperature)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + params.prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer

async def create_chat_stream(params: Params):
    global model , tokenizer
    for response,history in model.stream_chat(tokenizer,
                                  params.prompt,
                                  history=params.history,
                                  max_length=params.max_length,
                                  top_p=params.top_p,
                                  temperature=params.temperature):
        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
        }
        yield json.dumps(answer, ensure_ascii=False)
    log = "[" + time + "] " + '", prompt:"' + params.prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()


@app.post("/chat")
async def post_chat(params: Params):
    answer = await create_chat(params)
    return answer

@app.post("/chat_stream")
async def post_chat_stream(params: Params):
    return EventSourceResponse(create_chat_stream(params))

@app.post("/searchvectorsbase")
async def get_similarity_answer (params: Params):
    global embeddings
    index_name = params.index_name
    query = params.prompt
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    results = docsearch.similarity_search(query, k=2)
    docs = [results[i].page_content for i in range(len(results))]
    content = {
        "results": docs
    }
    torch_gc()
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    log = "[" + time + "] " + "query: "+ " " +str(query) +" searchvectorsbase: " + " " + str(content)
    print(log)
    return JSONResponse(status_code=200, content=content) 

@app.post("/embeddings")
async def embeddings_text (file: UploadFile = File(...)):
  global embeddings, pinecone
  content = await file.read()
  target_directory = os.path.dirname(__file__)  # the path of this program
  target_directory = os.path.join(target_directory, "knowledgeBase")  # build the path of the vector knowledge base
  target_file_path = os.path.join(target_directory, file.filename)  # build the path of the file
  with open(target_file_path, "wb") as f:
        f.write(content)
  index_name = os.path.splitext(file.filename)[0]
  indexes = pinecone.list_indexes()  #  delete existed indexes
  if len(indexes) != 0:
            for index in indexes:
                pinecone.delete_index(index)
  pinecone.create_index(index_name, dimension=1024, metric="euclidean")       
  loader = TextLoader(target_file_path,encoding="utf-8")
  documents = loader.load()
  text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
  docs = text_splitter.split_documents(documents)
  docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
  message = {
      "base_name": index_name
  }
  torch_gc()
  now = datetime.datetime.now()
  time = now.strftime("%Y-%m-%d %H:%M:%S")
  log = "[" + time + "] " + str(file.filename) + " hac been uploaded to " + str(index_name) + " index" + "\n"
  print(log)
  return JSONResponse(status_code=200,content=message)
    
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    model.eval()
    embeddings = HuggingFaceEmbeddings(model_name=emdedding_model_name, model_kwargs={'device': CUDA_DEVICE})
    pinecone.init(
        api_key=pinecone_api_key,  # find at app.pinecone.io
        environment=pinecone_environment  # next to api key in console
    )
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
    
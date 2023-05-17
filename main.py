from typing import Annotated
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI

from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
response = "Esperando resposta..."


def obter_resposta(pergunta):
    global response
    documents = SimpleDirectoryReader("static").load_data()
    # define prompt helper
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_output = 2048
    # set maximum chunk overlap
    max_chunk_overlap = 20

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=num_output))
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    response = "Esperando resposta..."
    qe = index.as_query_engine()
    response = qe.query(pergunta)
    # response = resp["response"]


@app.get("/")
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.get("/resultados")
async def obter_resultados():
    global response
    return (str(response)).replace("\n", "<br>")


@app.post("/gerar")
async def process_form(
    request: Request,
    arquivo: Annotated[UploadFile, File()],
    texto: Annotated[str, Form()],
    background_task: BackgroundTasks,
):
    global response
    # Realize aqui o processamento dos dados enviados pelo formulário
    # Salvar o arquivo no diretório "static"
    file_path = os.path.join("static", arquivo.filename)
    with open(file_path, "wb") as file:
        file.write(await arquivo.read())

    background_task.add_task(obter_resposta, texto)

    pergunta = f"Texto: {texto}<br>Arquivo: {arquivo.filename}"
    return templates.TemplateResponse("form.html", {"request": request, "pergunta": pergunta, "resposta": response})

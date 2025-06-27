import os
import tempfile
import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM

embedding_id = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=embedding_id)
# Variável para criação do bando de dados vetorizado que será alimentado de acordo com os arquivos enviados.
persist_directory = "db"


# Função para receber os arquivos no streamlit sem ter que informar o caminho das pastas, desta formas os arquivos estão vindo diretamente da memória
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    os.remove(temp_file_path)
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    chunks = text_spliter.split_documents(documents=docs)
    return chunks


# Função criada para validar se existe banco de dados criado, caso não existe ele irá criar um banco automáticamente.
def load_existing_vector_store():
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory="db",
            embedding_function=embedding,
        )
        return vector_store
    return None


# Função para criar o vector store caso ele seja nulo conforme teste anterior como abaixo o "vector_store=None"
def add_to_vector_store(chunks, vector_store=None):
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks, embedding=embedding, persist_directory=persist_directory
        )
    return vector_store


def ask_question(model, query, vector_store):
    # DEFINIÇÃO DO MODELO
    llm = OllamaLLM(model="llama3.2:3b")
    retriever = vector_store.as_retriever()
    system_prompt = """
    Você é uma assistente virtual.
    Você sempre irá respondar no idioma português e de forma formal e descontraída.
    Contexto: {context}"""
    # Adicionando as mensagem do usuário ou da IA com a estrutura abaixo
    messages = [("system", system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get("role"), message.get("content")))
    messages.append(("user", "{input}"))
    # Criando o prompt para o modelo
    prompt = ChatPromptTemplate.from_messages(messages)
    # Chain de perguntas e resposta utilizando a create_stuff_documents_chain, para facilitar as respostas da IA
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )
    # Chain com a create_retrieval_chain que é a partir dos dados importados também facilitando a resposta da IA
    chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=question_answer_chain
    )
    response = chain.invoke({"input": query})
    return response.get("answer")


vector_store = load_existing_vector_store()

st.set_page_config(
    page_title="Assistente Virtual",
    page_icon="📄",
)
st.header(
    "Assistente Virtual de Documentos 👩🏻‍💻",
    divider="gray",
    help="Ao anexar os arquivos, os dados serão gravados no Banco de Dados e o usuário poderá fazer perguntas referente aos arquivos.",
)
with st.sidebar:
    st.header("Upload de Arquivos 📄")
    uploaded_files = st.file_uploader(
        label="Faça Upload de Arquivos PDF", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processando documentos..."):
            all_chunks = []
            for uploaded_file in uploaded_files:
                chunks = process_pdf(file=uploaded_file)
                all_chunks.extend(chunks)
            vector_store = add_to_vector_store(
                chunks=all_chunks, vector_store=vector_store
            )

    model_option = [
        "llama32:3b",
    ]
    selected_model = st.sidebar.selectbox(
        label="Selecione o Modelo de IA", options=model_option
    )
# Função para criar os histórico de mensagens vazia
if "messages" not in st.session_state:
    st.session_state["messages"] = []

question = st.chat_input("Como posso ajudar ?")
# Função que verifica o "role" que é a mensagem foi do usuário ou da IA e o "content" que é a resposta que é enviada durante a conversa
if vector_store and question:
    for message in st.session_state.messages:
        st.chat_message(message.get("role")).write(message.get("content"))
    # Retornando abaixo a última mensagem do usuário
    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Buscando resposta..."):
        response = ask_question(
            model=selected_model, query=question, vector_store=vector_store
        )
        # st.chat_message é a mensagem que é recebida no chatbot
        # st.session_state é a sessão que foi criada ao abrir o navegador do chatbot
        st.chat_message("ai").write(response)
        st.session_state.messages.append({"role": "ai", "content": response})

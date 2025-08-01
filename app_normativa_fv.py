# --- PARCHE PARA ASYNCIO ---
import nest_asyncio
nest_asyncio.apply()
# ---------------------------

import os
import streamlit as st
import shutil
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader # <-- CAMBIO 1: Importamos el nuevo lector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- CONSTANTES Y RUTAS ---
DIRECTORIO_PERSISTENTE = "faiss_index"
DIRECTORIO_DOCUMENTOS = "documentos_normativos"

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente de Normativa FV", page_icon="💡", layout="wide")
st.title("💡 Asistente de Consulta para Normativas Fotovoltaicas")
st.write("Esta aplicación te permite hacer consultas en lenguaje natural sobre tus documentos de normativa. Sube tus PDFs, haz una pregunta y obtén una respuesta basada en ellos.")

# --- CONFIGURACIÓN DE LA API KEY Y GESTIÓN DE BD ---
with st.sidebar:
    st.header("Configuración")
    google_api_key = st.text_input("Ingresa tu API Key de Google AI", type="password")
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    else:
        st.warning("Por favor, ingresa tu API Key de Google AI para continuar.")
        st.stop()
    
    st.divider()

    st.subheader("Gestión de la Base de Datos")
    if st.button("Reiniciar y borrar base de datos"):
        if os.path.exists(DIRECTORIO_PERSISTENTE):
            with st.spinner("Borrando base de datos..."):
                shutil.rmtree(DIRECTORIO_PERSISTENTE)
            st.success("Base de datos borrada. La aplicación se recargará.")
            st.rerun()
        else:
            st.info("No hay ninguna base de datos para borrar.")

# Crear directorio para documentos si no existe
os.makedirs(DIRECTORIO_DOCUMENTOS, exist_ok=True)

# --- FUNCIONES CLAVE ---

@st.cache_resource
def cargar_y_procesar_documentos(ruta_documentos):
    """Carga y procesa los PDFs para crear la base de datos vectorial con FAISS."""
    st.info(f"Buscando documentos en '{ruta_documentos}'...")
    archivos_pdf = [f for f in os.listdir(ruta_documentos) if f.endswith('.pdf')]
    
    if not archivos_pdf:
        st.warning("No se encontraron archivos PDF para procesar.")
        return None

    st.success(f"Archivos PDF encontrados para procesar: {', '.join(archivos_pdf)}")

    documentos_cargados = []
    for archivo in archivos_pdf:
        ruta_completa = os.path.join(ruta_documentos, archivo)
        loader = PyMuPDFLoader(ruta_completa) # <-- CAMBIO 2: Usamos el nuevo lector
        documentos_cargados.extend(loader.load())

    st.info("Dividiendo documentos en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    fragmentos = text_splitter.split_documents(documentos_cargados)

    st.info("Creando embeddings y la base de datos vectorial (puede tardar un momento)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vectordb = FAISS.from_documents(fragmentos, embeddings)
    vectordb.save_local(DIRECTORIO_PERSISTENTE)
    
    return vectordb

@st.cache_resource
def cargar_cadena_qa():
    """Carga la cadena de consulta y recuperación (RetrievalQA) con un prompt de experto equilibrado."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vectordb = FAISS.load_local(DIRECTORIO_PERSISTENTE, embeddings, allow_dangerous_deserialization=True)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, convert_system_message_to_human=True)
    
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={"k": 7}),
        llm=llm
    )
    
    template = """
    Actúa como un Asistente experto en Normativa Eléctrica, especializado en plantas fotovoltaicas. Tu objetivo es proporcionar respuestas precisas y útiles basadas en la información extraída de los documentos normativos.

    Utiliza el siguiente contexto para responder la pregunta del usuario en español.
    
    Instrucciones:
    1. Basa tu respuesta directamente en la información del contexto proporcionado.
    2. Sintetiza la información de los diferentes artículos para dar una respuesta completa y coherente. Si el texto muestra un caso general, explica cómo se aplica al caso específico de la pregunta del usuario.
    3. Si la respuesta no se puede encontrar o inferir razonablemente del texto, indica que la información específica no fue encontrada en los documentos.
    4. Estructura tu respuesta de forma clara, usando listas o párrafos según sea necesario para facilitar la lectura.

    Contexto:
    {context}

    Pregunta: {question}

    Respuesta Asistente:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# --- LÓGICA PRINCIPAL DE LA APLICACIÓN ---

if not os.path.exists(DIRECTORIO_PERSISTENTE):
    st.warning("Base de datos vectorial no encontrada. Debes cargar documentos para crear una.")
    with st.sidebar:
        st.subheader("Cargar Documentos")
        uploaded_files = st.file_uploader(
            "Sube tus archivos PDF de normativas aquí",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("Procesar y Crear Base de Datos"):
            if uploaded_files:
                try:
                    for uploaded_file in uploaded_files:
                        with open(os.path.join(DIRECTORIO_DOCUMENTOS, uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    with st.spinner("Procesando documentos... Esta operación puede tardar varios minutos."):
                        cargar_y_procesar_documentos(DIRECTORIO_DOCUMENTOS)

                    st.success("¡Base de datos creada con éxito!")
                    st.info("La aplicación se recargará para usar la nueva base de datos.")
                    st.rerun()

                except Exception as e:
                    st.error(f"Ocurrió un error al procesar los documentos: {e}")
            else:
                st.error("No has subido ningún archivo.")
else:
    qa_chain = cargar_cadena_qa()
    
    st.header("Haz tu Consulta 💬")
    pregunta_usuario = st.text_area("Escribe aquí tu pregunta sobre la normativa:")

    if st.button("Obtener Respuesta"):
        if pregunta_usuario:
            with st.spinner("Buscando en la normativa y generando respuesta..."):
                try:
                    respuesta = qa_chain.invoke(pregunta_usuario)
                    
                    st.subheader("Respuesta:")
                    st.write(respuesta["result"])
                    
                    with st.expander("Ver fuentes utilizadas en la normativa"):
                        for doc in respuesta["source_documents"]:
                            nombre_archivo = os.path.basename(doc.metadata.get('source', 'N/A'))
                            st.info(f"**Fuente:** {nombre_archivo} | **Página:** {doc.metadata.get('page', 0) + 1}")
                            st.caption(doc.page_content)

                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")
        else:
            st.warning("Por favor, escribe una pregunta.")
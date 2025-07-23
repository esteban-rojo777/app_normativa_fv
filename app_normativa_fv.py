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
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# --- CONSTANTES Y RUTAS ---
DIRECTORIO_PERSISTENTE = "faiss_index"
DIRECTORIO_DOCUMENTOS = "documentos_normativos"

# --- CONFIGURACI��N DE LA P��GINA ---
st.set_page_config(page_title="Asistente de Normativa FV", page_icon="??", layout="wide")
st.title("?? Asistente de Consulta para Normativas Fotovoltaicas")
st.write("Esta aplicaci��n te permite hacer consultas en lenguaje natural sobre tus documentos de normativa. Sube tus PDFs, haz una pregunta y obt��n una respuesta basada en ellos.")

# --- CONFIGURACI��N DE LA API KEY Y GESTI��N DE BD ---
with st.sidebar:
    st.header("Configuraci��n")
    google_api_key = st.text_input("Ingresa tu API Key de Google AI", type="password")
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    else:
        st.warning("Por favor, ingresa tu API Key de Google AI para continuar.")
        st.stop()
    
    st.divider()

    st.subheader("Gesti��n de la Base de Datos")
    if st.button("Reiniciar y borrar base de datos"):
        if os.path.exists(DIRECTORIO_PERSISTENTE):
            with st.spinner("Borrando base de datos..."):
                shutil.rmtree(DIRECTORIO_PERSISTENTE)
            st.success("Base de datos borrada. La aplicaci��n se recargar��.")
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
    documentos_cargados = []
    for archivo in os.listdir(ruta_documentos):
        if archivo.endswith('.pdf'):
            ruta_completa = os.path.join(ruta_documentos, archivo)
            loader = PyPDFLoader(ruta_completa)
            documentos_cargados.extend(loader.load())

    if not documentos_cargados:
        st.warning("No se encontraron archivos PDF en el directorio.")
        return None

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
    """Carga la cadena de consulta y recuperaci��n (RetrievalQA) con prompt de experto en espa?ol."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vectordb = FAISS.load_local(DIRECTORIO_PERSISTENTE, embeddings, allow_dangerous_deserialization=True)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, convert_system_message_to_human=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": 7})
    
    template = """
    Act��a como un experto en normativa fotovoltaica. Tu tarea es analizar el siguiente contexto extra��do de documentos normativos y responder la pregunta del usuario de manera clara, profesional y concisa en espa?ol.

    No te limites a repetir el texto. Sintetiza la informaci��n, haz deducciones l��gicas basadas en los art��culos proporcionados y ofrece una conclusi��n pr��ctica. Si el texto no aborda directamente la pregunta, ind��calo, pero tambi��n explica las posibles interpretaciones o art��culos relacionados que podr��an aplicarse al caso.

    Contexto normativo:
    {context}

    Pregunta del usuario: {question}

    Respuesta de experto:
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

# --- L��GICA PRINCIPAL DE LA APLICACI��N ---

if not os.path.exists(DIRECTORIO_PERSISTENTE):
    st.warning("Base de datos vectorial no encontrada. Debes cargar documentos para crear una.")
    with st.sidebar:
        st.subheader("Cargar Documentos")
        uploaded_files = st.file_uploader(
            "Sube tus archivos PDF de normativas aqu��",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("Procesar y Crear Base de Datos"):
            if uploaded_files:
                # --- BLOQUE CON MANEJO DE ERRORES A?ADIDO ---
                try:
                    # 1. Guardar archivos en disco
                    for uploaded_file in uploaded_files:
                        with open(os.path.join(DIRECTORIO_DOCUMENTOS, uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # 2. Procesar documentos y crear la BD
                    with st.spinner("Procesando documentos... Esta operaci��n puede tardar varios minutos."):
                        cargar_y_procesar_documentos(DIRECTORIO_DOCUMENTOS)

                    st.success("?Base de datos creada con ��xito!")
                    st.info("La aplicaci��n se recargar�� para usar la nueva base de datos.")
                    st.rerun()

                except Exception as e:
                    st.error(f"Ocurri�� un error al procesar los documentos: {e}")
                # ----------------------------------------------
            else:
                st.error("No has subido ning��n archivo.")
else:
    qa_chain = cargar_cadena_qa()
    
    st.header("Haz tu Consulta ??")
    pregunta_usuario = st.text_area("Escribe aqu�� tu pregunta sobre la normativa:")

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
                            st.info(f"**Fuente:** {nombre_archivo} | **P��gina:** {doc.metadata.get('page', 'N/A', 0) + 1}")
                            st.caption(doc.page_content)

                except Exception as e:
                    st.error(f"Ocurri�� un error: {e}")
        else:
            st.warning("Por favor, escribe una pregunta.")
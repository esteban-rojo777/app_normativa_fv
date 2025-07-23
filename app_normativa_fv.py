# --- PARCHE PARA SQLITE ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --------------------------

# --- PARCHE PARA ASYNCIO ---
import nest_asyncio
nest_asyncio.apply()
# ---------------------------
import shutil

import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente de Normativa FV", page_icon="💡", layout="wide")
st.title("💡 Asistente de Consulta para Normativas Fotovoltaicas")
st.write("Esta aplicación te permite hacer consultas en lenguaje natural sobre tus documentos de normativa. Sube tus PDFs, haz una pregunta y obtén una respuesta basada en ellos.")

# --- CONFIGURACIÓN DE LA API KEY ---
with st.sidebar:
    st.header("Configuración")
    google_api_key = st.text_input("Ingresa tu API Key de Google AI", type="password")
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    else:
        st.warning("Por favor, ingresa tu API Key de Google AI para continuar.")
        st.stop()
        
        st.divider() # Añade una línea divisoria

    st.subheader("Gestión de la Base de Datos")
    if st.button("Reiniciar y borrar base de datos"):
        if os.path.exists(DIRECTORIO_PERSISTENTE):
            with st.spinner("Borrando base de datos..."):
                shutil.rmtree(DIRECTORIO_PERSISTENTE)
            st.success("Base de datos borrada. La aplicación se recargará.")
            st.rerun()
        else:
            st.info("No hay ninguna base de datos para borrar.")

# --- DIRECTORIOS Y RUTAS ---
DIRECTORIO_PERSISTENTE = "db_chroma_streamlit"
DIRECTORIO_DOCUMENTOS = "documentos_normativos"

# Crear directorio para documentos si no existe
os.makedirs(DIRECTORIO_DOCUMENTOS, exist_ok=True)


# --- FUNCIONES CLAVE (CON CACHÉ PARA EFICIENCIA) ---

@st.cache_resource
def cargar_y_procesar_documentos(ruta_documentos):
    """Carga y procesa los PDFs para crear la base de datos vectorial."""
    st.info(f"Buscando documentos en '{ruta_documentos}'...")
    documentos_cargados = []
    for archivo in os.listdir(ruta_documentos):
        if archivo.endswith('.pdf'):
            ruta_completa = os.path.join(ruta_documentos, archivo)
            loader = PyPDFLoader(ruta_completa)
            documentos_cargados.extend(loader.load())

    if not documentos_cargados:
        return None

    st.info("Dividiendo documentos en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    fragmentos = text_splitter.split_documents(documentos_cargados)

    st.info("Creando embeddings y la base de datos vectorial (puede tardar un momento)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=DIRECTORIO_PERSISTENTE
    )
    st.success("¡Base de datos vectorial creada y guardada con éxito!")
    return vectordb

def cargar_cadena_qa():
    """Carga la cadena de consulta y recuperación (RetrievalQA)."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma(persist_directory=DIRECTORIO_PERSISTENTE, embedding_function=embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, convert_system_message_to_human=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# --- LÓGICA PRINCIPAL DE LA APLICACIÓN ---

# Verificar si la base de datos ya existe. Si no, la crea.
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
                for uploaded_file in uploaded_files:
                    with open(os.path.join(DIRECTORIO_DOCUMENTOS, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                with st.spinner("Procesando documentos..."):
                    cargar_y_procesar_documentos(DIRECTORIO_DOCUMENTOS)
                st.rerun() # Recarga la app para reflejar el nuevo estado
            else:
                st.error("No has subido ningún archivo.")
else:
    # Si la base de datos ya existe, cargamos la cadena de consulta
    qa_chain = cargar_cadena_qa()
    
    st.header("Haz tu Consulta 💬")
    pregunta_usuario = st.text_area("Escribe aquí tu pregunta sobre la normativa:")

    if st.button("Obtener Respuesta"):
        if pregunta_usuario:
            with st.spinner("Buscando en la normativa y generando respuesta..."):
                try:
                    respuesta = qa_chain.invoke(pregunta_usuario) # <--- NUEVA LÍNEA CON INVOKE

                    st.subheader("Respuesta:")
                    st.write(respuesta["result"])

                    with st.expander("Ver fuentes utilizadas en la normativa"):
                        for doc in respuesta["source_documents"]:
                            # Extraer el nombre del archivo de la ruta
                            nombre_archivo = os.path.basename(doc.metadata.get('source', 'N/A'))
                            st.info(f"**Fuente:** {nombre_archivo} | **Página:** {doc.metadata.get('page', 'N/A') + 1}")
                            st.caption(doc.page_content)

                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")
        else:
            st.warning("Por favor, escribe una pregunta.")
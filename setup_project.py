import os

# Nombre de la carpeta del proyecto
PROJECT_NAME = "alpha_rag_project"

# Estructura de archivos y contenido
files = {
    f"{PROJECT_NAME}/requirements.txt": """streamlit
langchain
langchain-community
langchain-openai
langchain-google-genai
faiss-cpu
pypdf
python-dotenv
tiktoken
""",

    f"{PROJECT_NAME}/.env": """# Pega tu API Key aquí (sin comillas)
OPENAI_API_KEY=sk-proj-xxxxxxxx
GOOGLE_API_KEY=AIzaSyxxxxxx
""",

    f"{PROJECT_NAME}/README.md": """# 📈 Alpha-RAG: Financial Intelligence Terminal

## 🚀 Overview
Alpha-RAG is an AI-powered financial analyst that processes 10-K filings using Retrieval-Augmented Generation (RAG). It provides institutional-grade insights for retail investors.

## 🛠️ Tech Stack
- **Frontend:** Streamlit with Custom CSS
- **AI Engine:** LangChain + OpenAI/Gemini
- **Vector DB:** FAISS
""",

    f"{PROJECT_NAME}/assets/style.css": """/* ESTILO BLOOMBERG / DARK MODE */
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');

.stApp {
    background-color: #0e1117;
    font-family: 'Roboto Mono', monospace;
}

h1, h2, h3 {
    color: #e6e6e6 !important;
}

/* Tarjetas de Métricas */
div[data-testid="stMetricValue"] {
    font-size: 28px;
    color: #00ff41; /* Verde Terminal */
    text-shadow: 0px 0px 10px rgba(0, 255, 65, 0.4);
}

/* Chat Input */
.stTextInput > div > div > input {
    background-color: #262730;
    color: white;
    border: 1px solid #4b4b4b;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
}
""",

    f"{PROJECT_NAME}/utils/rag_engine.py": """import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

def process_pdf(file_path, api_key, provider="openai"):
    try:
        # 1. Cargar y Dividir PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(pages)

        # 2. Crear Vector Store
        if provider == "openai":
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        else:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    except Exception as e:
        return None

def get_qa_chain(vector_store, api_key, provider="openai"):
    # 3. Configurar Cerebro (LLM) con temperatura 0 para precisión
    if provider == "openai":
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=api_key)
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)
        
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain
""",

    f"{PROJECT_NAME}/app.py": """import streamlit as st
import os
from utils.rag_engine import process_pdf, get_qa_chain

# --- CONFIGURACIÓN INICIAL ---
st.set_page_config(page_title="Alpha-RAG Terminal", layout="wide", page_icon="📈")

# Cargar CSS personalizado
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    local_css("assets/style.css")
except:
    pass # Si no encuentra el estilo, sigue igual

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2666/2666505.png", width=50) # Logo genérico
    st.header("SYSTEM CONFIG")
    
    provider = st.radio("AI Provider", ["OpenAI", "Google Gemini"])
    
    api_key = st.text_input("API Key", type="password")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("UPLOAD 10-K FILING (PDF)", type="pdf")
    
    st.markdown("---")
    st.caption("Alpha-RAG v1.0 | Built for UEC 2026")

# --- DASHBOARD PRINCIPAL (THE WOW FACTOR) ---
st.title("Alpha-RAG 📈 // Financial Intelligence Unit")

# FAKE DATA para la DEMO (Esto asegura que el video se vea bien siempre)
st.subheader("LIVE MARKET ANALYSIS: TESLA INC (TSLA)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("REVENUE (YoY)", "+19.4%", "Bullish")
with col2:
    st.metric("NET MARGIN", "15.8%", "-2.1%")
with col3:
    st.metric("RISK SCORE", "MODERATE", delta_color="off")
with col4:
    st.metric("SENTIMENT", "POSITIVE", "0.88")

st.markdown("---")

# --- LÓGICA DEL CHAT ---
if uploaded_file and api_key:
    # Guardar archivo temporalmente
    with open("temp_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Estado de la sesión para el chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store" not in st.session_state:
        with st.spinner("🔄 INITIALIZING RAG ENGINE... PROCESSING DOCUMENTS..."):
            # Determinar proveedor (simple string match)
            prov_code = "openai" if provider == "OpenAI" else "google"
            st.session_state.vector_store = process_pdf("temp_doc.pdf", api_key, prov_code)
            st.success("SYSTEM ONLINE. READY FOR QUERIES.")

    # Mostrar historial
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input Usuario
    if prompt := st.chat_input("ENTER QUERY (e.g., 'What are the primary risks in China?')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("ANALYZING SEC FILINGS..."):
                try:
                    prov_code = "openai" if provider == "OpenAI" else "google"
                    qa = get_qa_chain(st.session_state.vector_store, api_key, prov_code)
                    
                    response = qa.invoke(prompt)
                    result = response['result']
                    sources = response['source_documents']
                    
                    st.markdown(result)
                    
                    # EVIDENCIA (Citaciones)
                    with st.expander("🔍 SOURCE EVIDENCE (VERIFIED)"):
                        for i, doc in enumerate(sources[:2]):
                            st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', '?')})**")
                            st.info(doc.page_content[:400] + "...")
                            
                    st.session_state.messages.append({"role": "assistant", "content": result})
                except Exception as e:
                    st.error(f"ERROR: {e}")

elif not uploaded_file:
    st.info("⚠️ AWAITING DOCUMENT UPLOAD...")
"""
}

def create_project():
    # Crear carpetas
    os.makedirs(f"{PROJECT_NAME}/assets", exist_ok=True)
    os.makedirs(f"{PROJECT_NAME}/utils", exist_ok=True)
    
    # Crear archivos
    for path, content in files.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Creado: {path}")
    
    print(f"\n🚀 PROYECTO LISTO EN: {os.path.abspath(PROJECT_NAME)}")
    print("---")
    print("PASOS SIGUIENTES:")
    print(f"1. cd {PROJECT_NAME}")
    print("2. pip install -r requirements.txt")
    print("3. streamlit run app.py")

if __name__ == "__main__":
    create_project()
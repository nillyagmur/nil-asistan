"""
🤖 RAG Tabanlı CV Chatbot - OpenAI/Google Gemini Versiyonu
Bu uygulama Nil Yağmur Muslu'nun CV bilgilerini kullanarak sorulara cevap verir.
API Key artık güvenli bir şekilde .env veya Hugging Face Secrets üzerinden alınır.
"""

# ===============================
# BÖLÜM 1: Gerekli Kütüphaneler
# ===============================
import os
from dotenv import load_dotenv  # .env dosyasından API key yüklemek için
import warnings
warnings.filterwarnings('ignore')  # gereksiz uyarıları gizle

import streamlit as st  # web arayüzü
from langchain.text_splitter import RecursiveCharacterTextSplitter  # metni parçalara bölmek için
from langchain_community.vectorstores import FAISS  # vektör veri tabanı
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # embedding modeli
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI  # Google Gemini LLM
import pickle  # veri tabanı kaydetmek ve yüklemek için

# ===============================
# BÖLÜM 2: API Key Yükleme
# ===============================
# .env dosyasını yükle (lokalde)
load_dotenv()

# Hugging Face Secrets veya .env üzerinden API key al
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ===============================
# BÖLÜM 3: Sabitler
# ===============================
DATA_FILE = "data.txt"  # CV verisi
FAISS_INDEX_PATH = "./faiss_index.pkl"  # FAISS index dosyası
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ===============================
# BÖLÜM 4: Fonksiyonlar
# ===============================
@st.cache_resource  # tekrar tekrar hesaplamamak için
def load_vector_store():
    """
    FAISS vektör veritabanını yükler veya oluşturur.
    Eğer diskte mevcutsa yükler, yoksa data.txt'den oluşturur.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        with open(FAISS_INDEX_PATH, 'rb') as f:
            vectorstore = pickle.load(f)
        return vectorstore
    else:
        with open(DATA_FILE, 'r', encoding='utf-8') as file:
            text = file.read()

        # Metni chunk'lara ayır
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n### ", "\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)

        # Embeddings oluştur
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # FAISS index oluştur
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

        # Kaydet
        with open(FAISS_INDEX_PATH, 'wb') as f:
            pickle.dump(vectorstore, f)

        return vectorstore

def get_response(question: str, vectorstore):
    """
    Kullanıcı sorusuna RAG + Google Gemini kullanarak cevap üretir.
    """
    try:
        # LLM yapılandırması
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY  # gizli anahtar kullanılıyor
        )

        # En alakalı chunk'ları bul
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(question)

        # Context oluştur
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Prompt oluştur
        prompt = f"""Sen Nil Yağmur Muslu'nun kişisel CV asistanısın.
BAĞLAM:
{context}
SORU: {question}
YANITLAMA KURALLARI:
1. Sadece bağlamı kullan
2. Türkçe yanıtla
3. Samimi ve doğal ol
4. Bilgi yoksa "Bu konuda bilgim yok" de
5. Kısa ve öz ol
YANIT:"""

        # Cevap üret
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"❌ Hata: {str(e)}"

# ===============================
# BÖLÜM 5: Web Arayüzü
# ===============================
def main():
    st.set_page_config(
        page_title="Nil Yağmur Muslu - CV Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🤖 Nil Yağmur Muslu - CV Chatbot")
    st.markdown("### Kişisel Asistan")
    st.info("💬 **Merhaba!** Ben Nil Yağmur Muslu'nun CV asistanıyım. Sorularını sorabilirsin.")

    # Vector store yükle
    try:
        with st.spinner("⏳ Vector store yükleniyor..."):
            vectorstore = load_vector_store()
        st.success("✅ Sistem hazır!")
    except Exception as e:
        st.error(f"❌ Hata: {e}")
        return

    # Chat geçmişi
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat geçmişini göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Kullanıcı girişi
    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Asistan cevabı
        with st.chat_message("assistant"):
            with st.spinner("💭 Düşünüyorum..."):
                response = get_response(prompt, vectorstore)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'><p>💻 Nil Yağmur Muslu'nun kişisel CV asistanı</p></div>",
        unsafe_allow_html=True
    )

# Eğer doğrudan çalıştırılırsa
if __name__ == "__main__":
    main()


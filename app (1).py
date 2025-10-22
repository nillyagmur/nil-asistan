"""
RAG Tabanlı CV Chatbot - Gemini Versiyonu
Nil Yağmur Muslu'nun CV bilgilerini kullanarak sorulara cevap verir.
"""

# ===============================
# BÖLÜM 1: Gerekli Kütüphanelerin Yüklenmesi
# ===============================
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

# Google Gemini için
from langchain_google_genai import ChatGoogleGenerativeAI

# ===============================
# BÖLÜM 2: Streamlit Sayfa Konfigürasyonu
# ===============================
st.set_page_config(
    page_title="Nil Yağmur Muslu - CV Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# BÖLÜM 3: Sabit Değişkenler
# ===============================
DATA_FILE = "data.txt"
FAISS_INDEX_PATH = "./faiss_index.pkl"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Gemini API anahtarını Hugging Face "Secrets" kısmından alacağız
# Hugging Face üzerinde: Settings -> Variables -> add new variable
# Key: GOOGLE_API_KEY, Value: AIzaSyAYvYqwvqia4qkrmTwd7oxxBMXd4Y3sbeE
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ===============================
# BÖLÜM 4: Fonksiyonlar
# ===============================

@st.cache_resource
def load_vector_store():
    """FAISS vector store'u yükler veya oluşturur."""
    if os.path.exists(FAISS_INDEX_PATH):
        with open(FAISS_INDEX_PATH, 'rb') as f:
            vectorstore = pickle.load(f)
        return vectorstore
    else:
        with open(DATA_FILE, 'r', encoding='utf-8') as file:
            text = file.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n### ", "\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

        with open(FAISS_INDEX_PATH, 'wb') as f:
            pickle.dump(vectorstore, f)

        return vectorstore


def get_response(question: str, vectorstore):
    """Kullanıcı sorusuna Gemini kullanarak RAG yanıtı üretir."""
    try:
        llm = ChatGoogleGenerativeAI(
            api_key=GOOGLE_API_KEY,
            model="gemini-2.0-flash",
            temperature=0.3,
            max_output_tokens=512
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
Sen Nil Yağmur Muslu'nun kişisel CV asistanısın. Verilen bağlam bilgilerini kullanarak soruları yanıtla.
BAĞLAM:
{context}

SORU: {question}

YANITLAMA KURALLARI:
1. Sadece verilen bağlam bilgilerini kullan.
2. Türkçe olarak yanıtla.
3. Samimi ama profesyonel bir üslup kullan.
4. Eğer bilgi bağlamda yoksa, "Bu konuda bilgim yok" de.
5. Kısa ve net yanıtlar ver.
YANIT:
"""

        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"❌ Hata: {e}"


# ===============================
# BÖLÜM 5: Ana Uygulama
# ===============================
def main():
    st.title("🤖 Nil Yağmur Muslu - CV Chatbot")
    st.markdown("### Kişisel Asistan")
    st.markdown("---")

    st.info("💬 **Merhaba!** Ben Nil Yağmur Muslu'nun CV asistanıyım. Onun hakkında merak ettiklerinizi sorabilirsiniz.")

    try:
        with st.spinner("⏳ Vector store yükleniyor..."):
            vectorstore = load_vector_store()
        st.success("✅ Sistem hazır!")
    except Exception as e:
        st.error(f"❌ Hata: {e}")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("💭 Düşünüyorum..."):
                response = get_response(prompt, vectorstore)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'><p>💻 Nil Yağmur Muslu'nun kişisel CV asistanı</p></div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()


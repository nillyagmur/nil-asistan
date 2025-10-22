# nil-asistan
Akbank GenAI Bootcamp için RAG tabanlı CV Chatbot Projesi
# Akbank GenAI Bootcamp - RAG Tabanlı CV Chatbot Projesi (Gemini Versiyonu)

Bu proje, **Akbank GenAI Bootcamp: Yeni Nesil Proje Kampı** kapsamında geliştirilmiştir. Proje, RAG (Retrieval-Augmented Generation) mimarisini kullanarak CV'm hakkında sorulan sorulara yanıt veren interaktif bir chatbot uygulamasını içermektedir.

**Proje Sahibi:** Nil Yağmur Muslu

---

## 🚀 Projenin Canlı Demosu

Projenin çalışan web arayüzüne (web arayüzü) aşağıdaki linkten erişebilirsiniz. Bu link, projenin Hugging Face Spaces üzerinde dağıtılmış (deploy edilmiş) halidir.

**➡️ UYGULAMA LİNKİ:** https://huggingface.co/spaces/nily123/nil-asistan
<img width="1882" height="916" alt="{4F08CBC7-E97D-45D2-9ECC-08E0FBA139C5}" src="https://github.com/user-attachments/assets/c0b317fb-c6da-46a9-9a8b-3d54b0b3762c" />

---

## 🎯 1. Projenin Amacı

Bu projenin temel amacı, belirli bir bağlam (Nil Yağmur Muslu'nun CV'si) üzerinde uzmanlaşmış, RAG tabanlı bir chatbot geliştirmektir. Chatbot, kullanıcıların CV hakkındaki sorularına, yalnızca sağlanan metin (`data.txt`) içerisindeki bilgilere dayanarak doğru ve tutarlı yanıtlar vermek üzere tasarlanmıştır.

## 📦 2. Veri Seti Hazırlama

Veri seti olarak, "Nil Yağmur Muslu" adına ait CV bilgileri kullanılmıştır. Bu bilgiler, `data.txt` adlı bir metin dosyasına ham metin olarak işlenmiştir. Veri seti, eğitim, iş deneyimleri, yetenekler ve projeler gibi standart CV bölümlerini içermektedir.

## 🛠️ 3. Çözüm Mimarisi ve Kullanılan Teknolojiler

Proje, LangChain çatısı altında bir RAG mimarisi kullanılarak geliştirilmiştir. Çözüm mimarisi aşağıdaki bileşenlerden oluşmaktadır:

1.  **Web Arayüzü:** `Streamlit`
    * Kullanıcı ile interaktif bir sohbet arayüzü oluşturmak için kullanıldı.

2.  **LLM (Generation Model):** `Google Gemini (gemini-pro)`
    * Kullanıcı sorusu ve FAISS'ten alınan ilgili bağlam (context), Google'ın Gemini modeline bir prompt ile gönderilerek nihai cevap üretildi.
    
3.  **Embedding Modeli:** `Hugging Face (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)`
    * Metin parçalarını (chunks) yüksek boyutlu vektörlere dönüştürmek için kullanıldı.

4.  **Vektör Veritabanı:** `FAISS (Facebook AI Similarity Search)`
    * Elde edilen vektörler, sunucu üzerinde hızlı ve verimli bir arama (similarity search) yapabilmek için `FAISS` index'i olarak saklandı.

5.  **Framework:** `LangChain`
    * Veri yükleme, parçalama (`RecursiveCharacterTextSplitter`), embedding modelini çağırma, FAISS veritabanını yönetme ve LLM'e (Gemini) prompt gönderme gibi tüm RAG akışını (pipeline) yönetmek için kullanıldı.

6.  **Dağıtım (Deployment):** `Hugging Face Spaces`
    * Uygulamanın canlı bir web linki olarak sunulması için kullanıldı.

## 📖 4. Çalışma Kılavuzu

Projenin yerel (local) bir bilgisayarda veya sunucuda çalıştırılması için gereken adımlar aşağıdadır.

```bash
git clone https://github.com/nillyagmur/nil-asistan.git
cd nil-asistan
```
```python -m venv venv

venv\Scripts\activate  # Windows için
# source venv/bin/activate  # MacOS/Linux için
```
```
pip install -r requirements.txt
```
 API Anahtarını Ayarlayın: Anahtarınızı .streamlit/secrets.toml dosyası oluşturarak (lokaldeyseniz) veya Hugging Face Secrets bölümüne (dağıtımsanız) GOOGLE_API_KEY adıyla eklemeniz gerekmektedir.
 ```
streamlit run app.py
 ```
### 5. Sonuçlar ve Öğrenimler

Geliştirilen chatbot, sağlanan `data.txt` bağlamına sadık kalarak, CV ile ilgili sorulan sorulara başarılı bir şekilde yanıt vermektedir. data.txt de sorulan soruya cevap bulamazsa da uydurmayıp bilgim yok demektedir.

Geliştirme süreci, özellikle Python kütüphaneleri arasındaki sürüm uyumsuzlukları ("Dependency Hell") konusunda önemli öğrenimler sağladı. `pydantic`, `faiss-cpu`, `langchain` ve `sentence-transformers` kütüphaneleri arasında yaşanan çakışmalar, `requirements.txt` dosyasında spesifik versiyonların sabitlenmesi (`pydantic<2`) ve uyumsuz olanların serbest bırakılması gibi yöntemlerle çözüldü.

Ayrıca, `pickle` dosyalarının (örn: `faiss_index.pkl`) farklı sistemler arası uyumsuzluğu tespit edilmiş; çözüm olarak index dosyasının sunucya yüklenmesi yerine, `app.py` içinde `@st.cache_resource` kullanılarak ilk çalıştırmada sunucunun kendisinde sıfırdan oluşturulması sağlanmıştır.



# nil-asistan
Akbank GenAI Bootcamp için RAG tabanlı CV Chatbot Projesi
Akbank GenAI Bootcamp - RAG Tabanlı CV Chatbot Projesi (Gemini Versiyonu)
Bu proje, Akbank GenAI Bootcamp: Yeni Nesil Proje Kampı kapsamında geliştirilmiştir. Proje, RAG (Retrieval-Augmented Generation) mimarisini kullanarak bir CV hakkında sorulan sorulara yanıt veren interaktif bir chatbot uygulamasını içermektedir.


Proje Sahibi: Nil Yağmur Muslu

🚀 Projenin Canlı Demosu
Projenin çalışan web arayüzüne (web arayüzü)  aşağıdaki linkten erişebilirsiniz. Bu link, projenin Hugging Face Spaces üzerinde dağıtılmış (deploy edilmiş) halidir.


➡️ UYGULAMA LİNKİ:https://huggingface.co/spaces/nily123/nil-asistan

🎯 1. Projenin Amacı
Bu projenin temel amacı, belirli bir bağlam (Nil Yağmur Muslu'nun CV'si) üzerinde uzmanlaşmış, RAG tabanlı bir chatbot geliştirmektir. Chatbot, kullanıcıların CV hakkındaki sorularına, yalnızca sağlanan metin (data.txt) içerisindeki bilgilere dayanarak doğru ve tutarlı yanıtlar vermek üzere tasarlanmıştır.

📦 2. Veri Seti Hazırlama
Veri seti olarak, "Nil Yağmur Muslu" adına ait CV bilgileri kullanılmıştır. Bu bilgiler, data.txt adlı bir metin dosyasına ham metin olarak işlenmiştir. Veri seti, eğitim, iş deneyimleri, yetenekler ve projeler gibi standart CV bölümlerini içermektedir.


(Not: Bootcamp proje gereksinimleri doğrultusunda, veri seti (data.txt) bu depoya (repository) eklenmemiştir.)

🛠️ 3. Çözüm Mimarisi ve Kullanılan Teknolojiler
Proje, LangChain çatısı altında bir RAG mimarisi  kullanılarak geliştirilmiştir. Çözüm mimarisi aşağıdaki bileşenlerden oluşmaktadır:


Web Arayüzü: Streamlit

Kullanıcı ile interaktif bir sohbet arayüzü oluşturmak için kullanıldı.


LLM (Generation Model): Google Gemini (gemini-pro) 

Kullanıcı sorusu ve FAISS'ten alınan ilgili bağlam (context), Google'ın Gemini modeline bir prompt ile gönderilerek nihai cevap üretildi.


Embedding Modeli: Hugging Face (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) 

Metin parçalarını (chunks) yüksek boyutlu vektörlere dönüştürmek için kullanıldı.


Vektör Veritabanı: FAISS (Facebook AI Similarity Search) 

Elde edilen vektörler, sunucu üzerinde hızlı ve verimli bir arama (similarity search) yapabilmek için FAISS index'i olarak saklandı.


Framework: LangChain 

Veri yükleme, parçalama (RecursiveCharacterTextSplitter), embedding modelini çağırma, FAISS veritabanını yönetme ve LLM'e (Gemini) prompt gönderme gibi tüm RAG akışını (pipeline) yönetmek için kullanıldı.

Dağıtım (Deployment): Hugging Face Spaces

Uygulamanın canlı bir web linki olarak sunulması için kullanıldı.

📖 4. Çalışma Kılavuzu
Projenin yerel (local) bir bilgisayarda veya sunucuda çalıştırılması için gereken adımlar  aşağıdadır.


1. Depoyu Klonlayın:

Bash

git clone https://github.com/[KullaniciAdiniz]/[RepoAdiniz].git
cd [RepoAdiniz]
2. Sanal Ortam (Virtual Env) Oluşturun ve Aktive Edin: 

Bash

python -m venv venv
venv\Scripts\activate  # Windows için
# source venv/bin/activate  # MacOS/Linux için
3. Gerekli Paketleri Yükleyin:  Tüm bağımlılıklar requirements.txt dosyasında listelenmiştir.

Bash

pip install -r requirements.txt
4. API Anahtarını Ayarlayın: Bu proje Google Gemini API anahtarı gerektirir. Anahtarınızı .streamlit/secrets.toml dosyası oluşturarak (lokaldeyseniz) veya Hugging Face Secrets bölümüne (dağıtımdaysanız) GOOGLE_API_KEY adıyla eklemeniz gerekmektedir.

5. Uygulamayı Çalıştırın:  (Çalıştırmadan önce data.txt dosyasını ana dizine eklediğinizden emin olun.)

Bash

streamlit run app.py
📊 5. Elde Edilen Sonuçlar ve Öğrenimler
Geliştirilen chatbot, sağlanan data.txt bağlamına sadık kalarak, CV ile ilgili sorulan sorulara başarılı bir şekilde yanıt verebilmektedir.

Geliştirme süreci, özellikle Python kütüphaneleri arasındaki sürüm uyumsuzlukları ("Dependency Hell") konusunda önemli öğrenimler sağlamıştır. pydantic, faiss-cpu, langchain ve sentence-transformers kütüphaneleri arasında yaşanan çakışmalar, requirements.txt dosyasında spesifik versiyonların sabitlenmesi (pydantic<2) ve uyumsuz olanların serbest bırakılması gibi yöntemlerle çözülmüştür.

Ayrıca, pickle dosyalarının (örn: faiss_index.pkl) farklı sistemler arası uyumsuzluğu tespit edilmiş; çözüm olarak index dosyasının sunucuya yüklenmesi yerine, app.py içinde @st.cache_resource kullanılarak ilk çalıştırmada sunucunun kendisinde sıfırdan oluşturulması sağlanmıştır.

# KawalTani Hybrid RAG — Rekomendasi Agronomi Padi Berbasis Sensor IoT Real-Time

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Sentence-Transformers](https://img.shields.io/badge/Embedding-MiniLM--L12--v2%20(384d)-orange.svg)](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
[![LLM Support](https://img.shields.io/badge/LLMs-Gemini%20%7C%20GPT--4o--mini%20%7C%20Llama%203-green.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/license-Academic%20Research-lightgrey.svg)](LICENSE)

Repository ini memuat kode sumber implementasi penuh, dataset evaluasi, basis pengetahuan agronomi standar, serta antarmuka web interaktif dari penelitian Tesis Magister Informatika:

> **Judul:** *Implementasi Retrieval-Augmented Generation (RAG) Berbasis Data Sensor IoT Real-Time untuk Sistem Rekomendasi Tindakan Agronomi Tanaman Padi pada Platform KawalTani*  
> **Peneliti:** Ahmad Subadri (NIM: 24206052019)  
> **Program Studi:** Magister Informatika, Fakultas Sains dan Teknologi, Universitas Islam Negeri Sunan Kalijaga Yogyakarta (2026)

---

## 📑 Daftar Isi
- [Latar Belakang & Masalah](#-latar-belakang--masalah)
- [Arsitektur Sistem (5 Lapisan)](#-arsitektur-sistem-5-lapisan)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Algoritma & Metode Ilmiah](#-algoritma--metode-ilmiah)
- [Struktur Direktori Proyek](#-struktur-direktori-proyek)
- [Panduan Instalasi & Menjalankan](#-panduan-instalasi--menjalankan)
- [Hasil Pengujian Empiris & Evaluasi Tesis](#-hasil-pengujian-empiris--evaluasi-tesis)
- [Lisensi & Sitasi](#-lisensi--sitasi)

---

## 🌾 Latar Belakang & Masalah

Platform cerdas pertanian presisi **KawalTani** mengumpulkan data sensor IoT tanah secara berkala (*Nitrogen, Fosfor, Kalium, pH, Suhu, Kelembaban, Konduktivitas Listrik/EC, Salinitas, dan TDS*). Namun, sistem chatbot tanya-jawab konvensional sebelumnya memiliki tiga kelemahan kritis:
1. **Halusinasi Faktual Spasiotemporal**: LLM generatif standar tidak memiliki akses langsung ke baris database sensor terkini pada area atau rentang tanggal spesifik.
2. **Kelemahan Rata-Rata (*Hidden Anomaly Trap*)**: Kondisi defisiensi ekstrem selama beberapa hari (misalnya Nitrogen anjlok ke $20\text{ mg/kg}$) dapat tertutupi oleh nilai rata-rata periode yang tampak "normal" ($69\text{ mg/kg}$), sehingga sistem lama keliru menyimpulkan "tanah dalam kondisi aman".
3. **Ketiadaan Hubungan Multidimensi**: Rekomendasi agronomi membutuhkan perpaduan simultan antara fase tumbuh tanaman (*Hari Setelah Tanam / HST*), jadwal protokol budidaya (`tr_plant_handling_copy`), dan pedoman pemupukan berimbang standar Kementerian Pertanian (Kementan/Balitbangtan) & *International Rice Research Institute* (IRRI).

Sistem **Hybrid RAG KawalTani** menyelesaikan permasalahan di atas melalui integrasi 3 jalur *retrieval* (*SQL Structured Data, Dense Vector Semantic Search, dan Adaptive Anomaly Routing*) yang dipadukan sebelum generasi jawaban oleh *Large Language Model*.

---

## 🏛️ Arsitektur Sistem (5 Lapisan)

Arsitektur sistem dibangun dalam 5 lapisan hierarkis sesuai standar perancangan sistem cerdas:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  LAPISAN 5: ANTARMUKA PENGGUNA (FastAPI REST API, WebSocket /ws/chat, Scientific UI)   │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼───────────────────────────────────────────────┐
│  LAPISAN 4: ORKESTRASI & GENERASI LLM (Context Fusion, Multi-Provider: Gemini/GPT/Ollama)│
└──────────────────┬──────────────────────────────────────────┬──────────────────────────┘
                   │                                          │
┌──────────────────▼───────────────────┐  ┌───────────────────▼──────────────────────────┐
│  LAPISAN 3: HYBRID RETRIEVAL         │  │  LAPISAN 2: MANAJEMEN PENGETAHUAN            │
│  • Jalur 1: Structured SQL           │  │  • 8 Dokumen Panduan Agronomi Padi (Kementan)│
│  • Jalur 2: Dense Semantic Vector    │  │  • Chunking Semantik & Metadata Tagging      │
│  • Jalur 3: Adaptive Anomaly Routing │  │  • Incremental Watermark Temporal Sync       │
└──────────────────┬───────────────────┘  └───────────────────┬──────────────────────────┘
                   │                                          │
┌──────────────────▼──────────────────────────────────────────▼──────────────────────────┐
│  LAPISAN 1: SUMBER DATA & PENYIMPANAN                                                  │
│  • Relational DB: MySQL / MariaDB (tm_sensor_read, tm_plant, tr_plant_handling_copy)   │
│  • Vector Store: Multilingual MiniLM-L12-v2 (384 Dimensi, Exact Cosine Inner-Product)  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Teknologi yang Digunakan

### 1. Bahasa Pemrograman & Pustaka Inti
- **Python 3.10 / 3.11**: Bahasa utama backend.
- **PyTorch 2.x & Torchvision**: *Runtime tensor* untuk eksekusi model *deep learning*.
- **Sentence-Transformers (5.4+)**: *Framework* inferensi *dense embedding* berbasis HuggingFace.
- **NumPy & Pandas**: Manipulasi matriks vektor, agregasi *sliding-window*, dan analisis deret waktu.
- **SciPy (scipy.stats)**: Pengujian statistik inferensial (*Paired Student's t-test* dan *Wilcoxon Signed-Rank Test*).

### 2. Basis Data & Penyimpanan
- **MySQL 8.0+ / MariaDB 10.5+ (InnoDB Engine)**: Penyimpanan data sensor tabular IoT, tabel relasi unit (`tr_unit`), metadata tanaman (`tm_plant`), dan protokol pemupukan (`tr_plant_handling_copy`).
- **B-Tree Composite Indexing**: Indeks gabungan `idx_ds_readdate(ds_id, read_date)` untuk pencarian *range query* dan titik waktu terbaru dalam latensi sub-milidetik.
- **Exact Vector Index (`ExactFlatIPIndex` / `Faiss`)**: Mesin pencarian vektor berbasis perkalian dalam (*inner-product*) pada vektor ternormalisasi $L_2$.

### 3. Backend & API Service
- **FastAPI (0.110+)**: Kerangka kerja API berbasis asinkron berkecepatan tinggi dengan validasi schema Pydantic v2.
- **Starlette & AnyIO**: *Asynchronous toolkit* dan *worker thread management*.
- **Uvicorn (ASGI Server)**: Server HTTP/1.1 dan WebSocket tingkat produksi.
- **WebSockets**: Komunikasi dua arah *real-time* untuk asisten percakapan responsif.

### 4. Mesin Generasi Bahasa (Large Language Models)
- **Google Gemini API (`gemini-3.5-flash-lite`)**: Model generatif utama dengan latensi inferensi sangat rendah (~2,3 detik) dan jendela konteks luas.
- **OpenAI API (`gpt-4o-mini`)**: Model pembanding *cloud* untuk pengujian A/B testing.
- **Ollama (`llama3:8b`)**: Model *open-weights on-premise* untuk skenario *edge computing* tanpa ketergantungan internet.
- **Groq API (`llama-3.1-8b-instant`)**: Model inferensi LPU berkecepatan tinggi.

### 5. Evaluasi Kualitas & Metrik
- **RAGAS (Retrieval Augmented Generation Assessment System)**:
  - *Faithfulness* (Kebenaran faktual jawaban terhadap konteks sensor).
  - *Answer Relevancy* (Tingkat relevansi rekomendasi terhadap pertanyaan pengguna).
  - *Context Precision* (Ketepatan konteks yang ditarik mesin retrieval).
  - *Context Recall* (Kelengkapan fakta sensor & agronomi yang berhasil dicakup).

### 6. Antarmuka Web Interaktif (*Scientific UI*)
- **HTML5, CSS3, & Modern Vanilla JavaScript (Zero External CDN Dependency)**: Bekerja mandiri tanpa ketergantungan koneksi CDN pihak ketiga.
- **Native SVG Visualization**: Rendering grafik deret waktu interaktif multi-parameter dengan dukungan interaksi seret rentang waktu (*drag-to-select range query*).
- **Academic/Editorial Paper Theme**: Palet warna alami terinspirasi tanah dan padi (*warm paper, forest green, clay red, grain gold*) dengan mode presentasi sidang tesis (*Presentation Mode* via tombol `P`).

---

## 🧠 Algoritma & Metode Ilmiah

Sistem ini mengimplementasikan kombinasi metode analitik, temu kembali informasi, dan statistika inferensial:

### 1. Algoritma Klasifikasi Intent Spasiotemporal (Jalur Hybrid Routing)
Mengekstraksi parameter sensor $\mathcal{P} \subseteq \{\text{N}, \text{P}, \text{K}, \text{pH}, \text{temp}, \text{hum}, \text{con}, \text{salin}, \text{tds}\}$, area $\mathcal{A} \subseteq \{1..6\}$, perangkat $\mathcal{D}$, serta jendela waktu temporal $\mathcal{W}$.
- **Parser Bahasa Alami Temporal Bahasa Indonesia**: Mengonversi ekspresi tanggal formal (*"15 Januari 2025 s/d 30 Januari 2025"*), tanggal tanpa tahun berulang (*"13 Januari - 20 Januari 2026"*), rentang relatif (*"tujuh hari terakhir"*), dan jendela Hari Setelah Tanam (*"H60 s/d H69"*).
- **Aturan Routing 3 Jalur**:
  $$\text{Route}(q) = \begin{cases} 
  \text{SQL}, & \text{jika } (\mathcal{P} \neq \emptyset \lor \mathcal{A} \neq \emptyset \lor \mathcal{W} \neq \emptyset) \land \neg \text{Advisory}(q) \\
  \text{Hybrid}, & \text{jika } (\mathcal{P} \neq \emptyset \lor \mathcal{A} \neq \emptyset) \land \text{Advisory}(q) \\
  \text{Vector}, & \text{jika murni konseptual agronomi (tanpa parameter/lokasi sensor)}
  \end{cases}$$

### 2. Algoritma Agregasi Deret Waktu & Data-to-Text Verbalization
Mengubah data tabular `tm_sensor_read` menjadi representasi teks terstruktur yang kaya konteks semantik:
- **Statistik Deskriptif**: Menghitung rerata ($\mu$), nilai minimum, maksimum, dan simpangan baku populasi ($\sigma$).
- **Sliding-Window Temporal Trend**: Membagi deret waktu harian menjadi dua paruh ($t_1$ dan $t_2$) untuk mendeteksi gradien perubahan $\Delta\% = \frac{\bar{x}_2 - \bar{x}_1}{|\bar{x}_1|} \times 100\%$.
- **Penyelarasan Fase HST & Protokol Tanam**: Menghitung $\text{HST} = t_{\text{ref}} - t_{\text{tanam}}$, menentukan fase tumbuh (*vegetatif, generatif, atau pemasakan*), serta menautkan jadwal dosis pemupukan dari tabel `tr_plant_handling_copy` (H+7 NPK 250 Kg/Ha, H+30 Pemupukan Susulan, H+120 Panen).

### 3. Algoritma Pencarian Semantik Dense Vector (Cosine Inner-Product)
- Dokumen agronomi dipecah menjadi potongan teks (*chunks*) berukuran 180 kata dengan tumpang-tindih (*overlap*) 40 kata.
- Teks dienkode menggunakan model transformer multibahasa $\mathbf{e} = f_{\theta}(\text{text}) \in \mathbb{R}^{384}$.
- Vektor dinormalisasi secara $L_2$ Euclidean:
  $$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} = \frac{\mathbf{v}}{\sqrt{\sum_{i=1}^{384} v_i^2}}$$
- Kemiripan semantik dihitung melalui operasi *dot product* yang setara dengan *Cosine Similarity*:
  $$\text{Sim}(q, d) = \hat{\mathbf{q}} \cdot \hat{\mathbf{d}}^T = \sum_{i=1}^{384} \hat{q}_i \hat{d}_i$$
- Pengambilan $k$ dokumen teratas menggunakan *partial sorting* argpartition $\mathcal{O}(N + k \log k)$.

### 4. Algoritma Eskalasi Anomali Tersembunyi (Adaptive Routing)
Mencegah jebakan halusinasi rata-rata normal (*Hidden Anomaly Trap*):
- Jika rute awal terklasifikasi sebagai $\text{SQL}$ (pertanyaan faktual), sistem memindai distribusi nilai di luar ambang batas:
  $$N_{\text{below}} = \sum \mathbb{I}(x_i < \text{min\_norm}), \quad N_{\text{above}} = \sum \mathbb{I}(x_i > \text{max\_norm})$$
- Bila $N_{\text{below}} > 0$ atau $N_{\text{above}} > 0$, status sensor otomatis disematkan tag `PERHATIAN: [X] dari [Y] pembacaan di luar ambang` dan rute dinaikkan secara adaptif:
  $$\text{Route}_{\text{final}} = \text{Hybrid}$$
  sehingga dokumen penanganan defisiensi/toksisitas dari Kementan/IRRI otomatis diikutsertakan ke dalam prompt LLM.

### 5. Algoritma Incremental Temporal Indexing (Watermark-based)
Memungkinkan sistem menyinkronkan data sensor baru dari MySQL ke ruang vektor tanpa melakukan *re-indexing* dokumen lama dari awal:
- Membaca penanda waktu terakhir yang telah terindeks (*high-watermark*) $t_{\text{watermark}}$ dari `watermark.json`.
- Melakukan agregasi mingguan berbasis ISO *year-week* (`YEARWEEK(read_date, 1)`):
  $$\Delta \mathcal{R} = \{r \in \text{tm\_sensor\_read} \mid r.\text{read\_date} > t_{\text{watermark}}\}$$
- Mengonversi ringkasan multi-parameter mingguan menjadi teks naratif dan menambahkan vektor baru ke indeks aktif dalam kompleksitas waktu $\mathcal{O}(|\Delta \mathcal{R}|)$.

### 6. Pengujian Hipotesis Statistik (Uji Inferensial H1 & H2)
Untuk membuktikan secara ilmiah bahwa skor kualitas sistem usulan lebih tinggi daripada baseline:
- **Paired Student's t-test (Satu Arah)**:
  $$t = \frac{\bar{d}}{s_d / \sqrt{n}}, \quad \bar{d} = \frac{1}{n}\sum_{i=1}^n (x_{\text{rag}, i} - x_{\text{base}, i})$$
- **Wilcoxon Signed-Rank Test (Non-parametrik)**:
  $$W = \min(W^+, W^-), \quad W^+ = \sum_{d_i > 0} \text{rank}(|d_i|)$$
- **Kriteria Keputusan**: Hipotesis diterima apabila nilai signifikansi $p\text{-value} < 0{,}05$ ($\alpha = 5\%$).

---

## 📁 Struktur Direktori Proyek

```text
implementasi_rag_kawaltani/
├── .env.example                 # Template variabel lingkungan (API Key & DB host)
├── .gitignore                   # Proteksi kredensial, cache, dan file biner
├── jalankan_ui.bat              # Skrip 1-klik Windows untuk menjalankan antarmuka
├── requirements.txt             # Daftar dependensi Python
├── run_server.py                # Peluncur server FastAPI & pemanasan model
├── run_evaluation.py            # CLI audit retrieval, collect LLM, RAGAS, & uji hipotesis
├── run_context_tradeoff.py      # Skrip eksperimen kuantitatif trade-off (Tujuan #4)
├── kawaltani_rag/               # Paket utama sistem RAG KawalTani
│   ├── __init__.py
│   ├── config.py                # Konfigurasi terpusat (DB, embedding, LLM, window)
│   ├── intent.py                # Algoritma klasifikasi intent 3 jalur & parser temporal
│   ├── sql_retriever.py         # Jalur 1: Structured retrieval & data-to-text verbalization
│   ├── embeddings.py            # Sentence-Transformers MiniLM-L12-v2 encoder
│   ├── vector_store.py          # Jalur 2: Mesin pencarian vektor exact inner-product
│   ├── incremental_indexer.py   # Sinkronisasi inkremental mingguan berbasis watermark
│   ├── pipeline.py              # Jalur 3: Fusi konteks, routing adaptif, & mode komparasi
│   ├── llm.py                   # Multi-provider client (Gemini, OpenAI, Ollama, Groq)
│   ├── api.py                   # REST API, WebSocket /ws/chat, & endpoint visualisasi
│   └── static/
│       └── index.html           # Antarmuka web interaktif ilmiah (Zero CDN dependency)
├── vector_docs/                 # 8 Dokumen pedoman agronomi standar Balitbangtan & IRRI
│   ├── 01_nitrogen.md           # Defisiensi, gejala visual, dosis urea, & waktu aplikasi
│   ├── 02_fosfor.md             # Peran perakaran padi, gejala kerdil, & pupuk SP-36
│   ├── 03_kalium.md             # Ketahanan rebah, pengisian malai, & aplikasi KCl
│   ├── 04_ph_tanah.md           # Pengapuran dolomit, keracunan besi/aluminium
│   ├── 05_suhu_kelembaban.md    # Cekaman suhu, kebutuhan air macak-macak
│   ├── 06_salinitas_ec_tds.md   # Ambang salinitas dS/m, flushing air tawar
│   ├── 07_fase_pertumbuhan_hst.md # Fase vegetatif, reproduktif, & pematangan bulir
│   ├── 08_jadwal_pemupukan.md   # Jadwal pemupukan berimbang spesifik lokasi
│   └── references.md            # Rujukan resmi Permentan, Katam Terpadu, & IRRI
├── data/                        # Dataset evaluasi kuantitatif
│   ├── qa_dataset_100.csv       # 100 pasangan QA uji (40 nutrisi, 30 rekomendasi, 20 lingkungan, 10 komparatif)
│   ├── qa_vector_general_10.csv # 10 kueri konseptual umum untuk pengujian jalur vector
│   └── ground_truth_anomalies.json # Daftar episode anomali tersembunyi berlabel ground-truth
├── sql/                         # Skrip skema & data MySQL
│   ├── fix_data_inconsistencies_v2.sql # Harmonisasi satuan tr_unit & indeks komposit
│   ├── synthetic_tm_sensor_read.sql    # 384.912 baris data sensor terkalibrasi secara fisik
│   └── kawaltani_simpul_telkom_original_backup.sql # Cadangan database asli
├── index/                       # Indeks vektor neural terkompilasi (384-dim)
│   ├── chunks.json              # 42 chunk agronomi + 110 chunk temporal mingguan
│   ├── agronomi_vecs.npy        # Matriks bobot embedding numpy float32
│   ├── meta.json                # Metadata dimensi & konfigurasi indexer
│   └── watermark.json           # Watermark sinkronisasi sensor MySQL
└── results/                     # Hasil eksperimen empiris Bab IV
    ├── audit_retrieval.json     # Metrik akurasi routing & kelengkapan konteks
    └── tradeoff_context_vs_latency.csv # Matriks trade-off panjang konteks vs latensi
```

---

## 🚀 Panduan Instalasi & Menjalankan

### 1. Prasyarat Lingkungan
- Komputer berbasis Windows, Linux, atau macOS.
- Python versi 3.10 atau 3.11 telah terpasang.
- Layanan MySQL / MariaDB (misalnya via Laragon atau XAMPP) aktif pada `localhost:3306`.

### 2. Kloning Repository & Pembuatan Virtual Environment
```bash
git clone https://github.com/<username-anda>/kawaltani-hybrid-rag.git
cd kawaltani-hybrid-rag
python -m venv venv

# Aktivasi virtual environment (Windows):
.\venv\Scripts\activate
# Aktivasi virtual environment (Linux/macOS):
source venv/bin/activate

# Pasang pustaka dependensi:
pip install -r requirements.txt
```

### 3. Konfigurasi Lingkungan (`.env`)
Salin file `.env.example` menjadi `.env`:
```bash
cp .env.example .env
```
Buka `.env` dan sesuaikan parameter kredensial MySQL dan API Key LLM yang digunakan:
```env
KT_DB_HOST=localhost
KT_DB_PORT=3306
KT_DB_USER=root
KT_DB_PASSWORD=
KT_DB_NAME=kawaltani_rag_tesis

KT_LLM_PROVIDER=gemini
KT_LLM_MODEL=gemini-3.5-flash-lite
GEMINI_API_KEY=masukkan_api_key_google_ai_studio_anda
```

### 4. Menjalankan Antarmuka Web Interaktif
- **Di Windows**: Cukup klik dua kali file [`jalankan_ui.bat`](jalankan_ui.bat).
- **Atau via Terminal**:
  ```bash
  python -u run_server.py
  ```
Buka peramban di [**http://127.0.0.1:8000**](http://127.0.0.1:8000) untuk mengakses dashboard ilmiah, visualisasi deret waktu data lahan, dan uji coba interaktif.

---

## 📊 Hasil Pengujian Empiris & Evaluasi Tesis

### A. Pengujian Audit Hybrid Retrieval (100 QA Utama + 10 QA Konseptual)
Hasil verifikasi nyata terhadap $385.312$ baris pembacaan sensor pada basis data:

| Parameter Pengujian | Target Hipotesis | Hasil Terverifikasi | Status |
|---|---|---|---|
| Kelengkapan Konteks (*Context Coverage*) | $100\%$ | **100 / 100 (100.0%)** | Terpenuhi |
| Validitas Konteks Sensor SQL | $100\%$ | **100 / 100 (100.0%)** | Terpenuhi |
| Akurasi Routing Efektif | $> 90\%$ | **100 / 100 (100.0%)** | Terpenuhi |
| Routing Kueri Konseptual (Jalur Vector) | $> 90\%$ | **10 / 10 (100.0%)** | Terpenuhi |
| Median Latensi Retrieval | $< 5.000\text{ ms}$ (H3) | **51.25 ms** (Rerata: 113.32 ms) | Terpenuhi |
| Deteksi Anomali Tersembunyi (*Adaptive Routing*) | Aktif otomatis | **100% Lulus Uji** | Terpenuhi |

### B. Analisis Trade-Off Panjang Konteks vs. Latensi (Tujuan Penelitian ke-4)
Pengujian matriks kombinasi $\text{top\_k} \in \{2, 4, 6, 8\}$ dokumen agronomi terhadap $\text{max\_daily\_points} \in \{3, 7, 14\}$ deret waktu pada file [`results/tradeoff_context_vs_latency.csv`](results/tradeoff_context_vs_latency.csv):

| Konfigurasi | Rerata Token Prompt | Latensi Retrieval (Rerata) | Latensi P95 | Cakupan Fakta Acuan |
|---|:---:|:---:|:---:|:---:|
| $K=2, D=3$ | $436.4\text{ tok}$ | $125.05\text{ ms}$ | $404.80\text{ ms}$ | $100.00\%$ |
| $K=2, D=7$ | $439.7\text{ tok}$ | $163.34\text{ ms}$ | $642.10\text{ ms}$ | $100.00\%$ |
| $K=2, D=14$ | $439.8\text{ tok}$ | $106.65\text{ ms}$ | $298.90\text{ ms}$ | $100.00\%$ |
| **$K=4, D=14$ (Bawaan Sistem)** | **$574.2\text{ tok}$** | **$175.64\text{ ms}$** | **$741.40\text{ ms}$** | **$100.00\%$** |
| $K=8, D=14$ | $821.5\text{ tok}$ | $83.78\text{ ms}$ | $256.00\text{ ms}$ | $100.00\%$ |

> **Temuan Kunci Bab IV**: Konfigurasi $K=4, D=14$ memberikan titik temu paling optimal (*sweet spot*), di mana seluruh informasi dosis, jadwal pemupukan, dan konteks anomali tercakup lengkap ($100\%$) dengan ukuran prompt yang tetap efisien ($574$ token), sehingga menekan biaya token inferensi LLM tanpa mengorbankan ketepatan rekomendasi.

---

## 📜 Lisensi & Sitasi

Kode sumber ini dikembangkan khusus untuk keperluan riset akademik pada program Magister Informatika UIN Sunan Kalijaga Yogyakarta.

Bila Anda menggunakan atau mengutip arsitektur dan dataset dalam penelitian Anda, silakan sitasi naskah proposal tesis acuan:

```bibtex
@mastersthesis{subadri2026kawaltani,
  author  = {Ahmad Subadri},
  title   = {Implementasi Retrieval-Augmented Generation (RAG) Berbasis Data Sensor IoT Real-Time untuk Sistem Rekomendasi Tindakan Agronomi Tanaman Padi pada Platform KawalTani},
  school  = {Program Studi Magister Informatika, Universitas Islam Negeri Sunan Kalijaga},
  year    = {2026},
  address = {Yogyakarta, Indonesia}
}
```

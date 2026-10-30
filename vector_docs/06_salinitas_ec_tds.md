---
judul: Pengelolaan Konduktivitas Listrik (EC), Salinitas, dan TDS pada Lahan Padi
kategori: kondisi_lingkungan
parameter_terkait: con, konduktivitas, ec, salin, salinitas, garam, tds, zat terlarut
---

## Hubungan EC, Salinitas, dan TDS pada Tanah Sawah

Konduktivitas Listrik (*Electrical Conductivity* / **EC**), **Salinitas**, dan *Total Dissolved Solids* (**TDS**) adalah tiga parameter sensor yang saling berkaitan erat dalam mengukur konsentrasi ion terlarut (baik ion hara pupuk maupun garam natrium klorida) di dalam larutan tanah sawah:
- Nilai EC dan TDS yang terlalu rendah menandakan miskinnya ion hara terlarut di zona perakaran (*nutrient depletion*).
- Sebaliknya, nilai EC, salinitas, atau TDS yang terlalu tinggi menandakan akumulasi garam berlebih (*salinity stress* atau *over-fertilization*) yang menyebabkan akar padi gagal menyerap air akibat tekanan osmotik terbalik (*kekeringan fisiologis*).

## Satuan Pengukuran dan Ambang Batas Sensor KawalTani

Penting memperhatikan kesetaraan satuan sensor agar tidak terjadi salah tafsir:
- **1 mS/cm = 1 dS/m = 1.000 µS/cm** (`us/c`).
- Pada konfigurasi KawalTani:
  1. **Konduktivitas Listrik (`soil_con`):**
     - Rentang ideal larutan tanah sawah non-salin: **1,0 – 3,0 mS/cm** (atau setara **1.000 – 3.000 µS/cm**; nilai 200 – 1.500 µS/cm tergolong aman/rendah garam, sedangkan **> 3.000 µS/cm [> 3,0 mS/cm]** mulai memicu cekaman salinitas pada padi).
  2. **TDS (`soil_tds`):**
     - Rentang acuan: **500 – 2.000 ppm (mg/L)**. Nilai < 500 ppm menunjukkan konsentrasi mineral/hara terlarut relatif encer, sedangkan > 2.000 ppm menunjukkan kepekatan zat terlarut tinggi.
  3. **Salinitas (`soil_salin`):**
     - Dalam satuan mg/L (ppm): air/larutan tanah sawah aman berada pada kisaran **< 1.000 mg/L** (ambang kritis cekaman garam padi umumnya mulai tampak di atas 1.500–2.000 mg/L atau > 2–3 ppt). Pada sensor dengan ambang indeks 0–100, kenaikan tajam di atas ambang menandakan intrusi atau akumulasi garam.

## Dampak Cekaman Salinitas / EC Tinggi pada Tanaman Padi

Padi tergolong tanaman yang **peka terhadap salinitas**, terutama pada **fase bibit muda (0–21 HST)** dan **fase pembungaan (55–75 HST)**:
1. Ujung daun muda menggulung, memutih, lalu mengering terbakar (*white leaf tip*).
2. Pertumbuhan anakan terhenti tidak merata (*spotty growth* di petakan sawah).
3. Malai keluar tidak sempurna, banyak bulir putih hampa (*whitehead* fisiologis), dan hasil panen dapat turun 25–50% bila EC melampaui 4 mS/cm (4.000 µS/cm).

## Rekomendasi Tindakan Pengelolaan

### Tindakan Saat EC atau TDS Terlalu Rendah (Kekurangan Ion Hara)
1. Periksa nilai spesifik sensor **Nitrogen (N), Fosfor (P), dan Kalium (K)** pada area yang sama.
2. Bila NPK memang berada di bawah ambang normal, lakukan pemupukan susulan terukur sesuai jadwal fase tumbuh dan tambahkan pupuk organik/mineral untuk meningkatkan ketersediaan elektrolit hara di larutan tanah.

### Tindakan Saat EC, Salinitas, atau TDS Terlalu Tinggi (Akumulasi Garam)
1. **Pencucian Garam dengan Irigasi Air Tawar (*Leaching & Flushing*):** Masukkan air tawar berkualitas baik ke petakan sawah hingga menggenang 5–7 cm, biarkan melarutkan garam, lalu buang melalui saluran *outlet* drainase. Ulangi secara berkala.
2. **Hentikan Pupuk Berindeks Garam Tinggi:** Tunda penggunaan KCl dan Urea dosis besar sekaligus; hindari pupuk klorida berlebih saat EC tinggi.
3. **Aplikasi Pembenah Tanah (Gypsum & Bahan Organik):** Berikan **Gypsum ($CaSO_4$, 1–2 ton/ha)** pada lahan terdampak salinitas natrium agar ion $Ca^{2+}$ menggantikan $Na^+$ pada kompleks jerapan tanah sehingga mudah tercuci air drainase, disertai kompos jerami (2–5 ton/ha).
4. **Pemilihan Varietas Toleran Salinitas (Musim Tanam Berikutnya):** Untuk lahan rawan salin, gunakan varietas padi toleran salinitas rekomendasi Kementan seperti **Inpari 34 Salin Agritan** atau **Inpari 35 Salin Agritan**.


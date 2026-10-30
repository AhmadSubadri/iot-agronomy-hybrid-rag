---
judul: Jadwal dan Protokol Pemupukan Berimbang Tanaman Padi Sawah
kategori: rekomendasi_pemupukan
parameter_terkait: pupuk, pemupukan, dosis, jadwal, urea, npk, sp-36, kcl, organik
---

## Prinsip Pemupukan Berimbang Spesifik Lokasi (5 Tepat)

Rekomendasi pemupukan padi sawah mengacu pada prinsip **5 Tepat**: **Tepat Jenis, Tepat Dosis, Tepat Waktu, Tepat Cara, dan Tepat Sasaran** berdasarkan pembacaan sensor tanah aktual serta usia tanaman (HST). Pemupukan tidak boleh dilakukan secara seragam tanpa melihat status hara tanah di masing-masing area petakan.

## Protokol Jadwal Penanganan Tanaman pada Platform KawalTani

Berdasarkan tabel jadwal penanganan (`tr_plant_handling_copy`) pada sistem KawalTani dan panduan teknis Kementerian Pertanian RI untuk padi sawah:

1. **Pra-Tanam (H-14 s/d H-7 Sebelum Tanam):**
   - Aplikasikan **Pupuk Organik / Kompos Jerami Matang (2.000 kg/ha atau 2 ton/ha)** saat pengolahan tanah.
   - Bila sensor pH tanah menunjukkan tanah masam (< 5,5), berikan **Kapur Pertanian / Dolomit (1–2 ton/ha)** bersamaan pengolahan tanah terakhir.

2. **Pemupukan Dasar — H+7 HST (Toleransi 7–14 HST):**
   - **Jenis & Dosis Acuan KawalTani:** **NPK Majemuk 250 kg/ha** + alokasi **Urea** tahap awal (sekitar **75–100 kg/ha** dari kuota Urea musim berjalan, disesuaikan pembacaan sensor `soil_nitro`).
   - Bila kadar Fosfor tanah sangat rendah (< 150 mg/kg), tambahkan suplemen **SP-36 (50–75 kg/ha)** di awal karena P dibutuhkan untuk perakaran dini.

3. **Pemupukan Susulan — H+30 HST (Toleransi 25–45 HST):**
   - **Jenis & Dosis Acuan KawalTani:** **NPK Majemuk 250 kg/ha** + **Urea susulan (100–125 kg/ha)** yang dibagi pada fase anakan aktif (~28–30 HST) dan menjelang primordia malai (~40–45 HST).
   - **Penyesuaian Dinamis Berbasis Sensor:**
     - Jika pada H+30 sensor `soil_nitro` sudah tinggi (> 100–120 mg/kg), **kurangi dosis Urea susulan 30–50%** untuk mencegah rebah dan penyakit blas.
     - Jika sensor `soil_nitro` anjlok di bawah 45 mg/kg pada fase vegetatif, segera berikan koreksi Urea tanpa menunggu jadwal berikutnya terlewat.
     - Jika sensor `soil_pot` menjelang bunting (35–45 HST) turun mendekati atau di bawah 100 mg/kg, tambahkan **KCl 50 kg/ha** untuk menjamin pengisian bulir gabah.

4. **Panen — H+120 HST (Toleransi ±5 Hari):**
   - Keringkan petakan sawah 10–14 hari sebelum panen (sekitar H+106 s/d H+110). Panen dilakukan saat 90–95% gabah pada malai telah menguning bernas.

## Teknik Aplikasi Pupuk yang Efisien di Lapangan

1. **Kondisi Air Macak-macak:** Selalu kurangi tinggi air petakan menjadi 1–2 cm sebelum menabur pupuk butiran (Urea, NPK, SP-36, KCl), dan tutup pintu pengeluaran (*outlet*) selama minimal 3–4 hari agar hara terserap koloid tanah dan tidak terbuang ke saluran pembuangan.
2. **Waktu Penaburan:** Taburkan pupuk pada pagi hari setelah embun di daun mengering (pukul 08.00–10.00 WIB) atau sore hari (pukul 15.30–17.00 WIB) agar butiran Urea tidak menempel dan membakar helai daun basah serta meminimalkan penguapan panas siang hari.
3. **Larangan Pencampuran Langsung:** Jangan mencampur pupuk Urea langsung dengan Kapur/Dolomit atau abu dapur karena reaksi basa kuat akan menguapkan nitrogen menjadi gas amonia seketika.


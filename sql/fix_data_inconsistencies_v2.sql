-- ============================================================================
-- fix_data_inconsistencies_v2.sql
-- Dijalankan HANYA pada database duplikat eksperimen `kawaltani_rag_tesis`
-- (Database asli `kawaltani_simpul_telkom` tetap utuh 100% sebagai arsip asli)
-- ============================================================================

-- 1. Harmonisasi simbol satuan pada `tr_unit` (perbaikan karakter rusak & salah ketik satuan)
UPDATE tr_unit SET unit_symbol = '' WHERE unit_id IN ('ph', 'soil_ph1', 'soil_ph2', 'soil_ph3');
UPDATE tr_unit SET unit_symbol = 'mg/kg' WHERE unit_id IN ('nitro', 'phos', 'pot', 'soil_nitro1', 'soil_nitro2', 'soil_phos1', 'soil_phos2', 'soil_pot1', 'soil_pot2');
UPDATE tr_unit SET unit_symbol = 'µS/cm' WHERE unit_id IN ('soil_con1', 'soil_con2');
UPDATE tr_unit SET unit_symbol = 'mS/cm' WHERE unit_id = 'con';
UPDATE tr_unit SET unit_symbol = 'mg/L' WHERE unit_id IN ('salin', 'soil_salin1', 'soil_salin2');
UPDATE tr_unit SET unit_symbol = '°C' WHERE unit_id IN ('temp', 'soil_temp1', 'soil_temp2', 'sol_temp');

-- 2. Seragamkan ambang batas Nitrogen (N) semua area ke 45-120 mg/kg (sesuai Bab 2.2.5 Proposal)
UPDATE td_device_sensors
SET ds_min_norm_value = 45, ds_max_norm_value = 120
WHERE ds_id IN ('soil_nitro1', 'soil_nitro2', 'soil_nitro3', 'soil_nitro4', 'soil_nitro5', 'soil_nitro6', 'soil_nitro');

-- 3. Sesuaikan ambang batas EC & Salinitas Area 1-2 dengan satuan sensor riil di `tr_unit`
--    soil_con1/2 melapor dalam µS/cm (200 - 2000 µS/cm setara 0.2 - 2.0 mS/cm untuk sawah non-salin)
UPDATE td_device_sensors
SET ds_min_norm_value = 200, ds_max_norm_value = 2000
WHERE ds_id IN ('soil_con1', 'soil_con2');

--    soil_salin1/2 melapor dalam mg/L (0 - 1000 mg/L aman untuk air/larutan sawah padi)
UPDATE td_device_sensors
SET ds_min_norm_value = 0, ds_max_norm_value = 1000
WHERE ds_id IN ('soil_salin1', 'soil_salin2');

-- 4. Sesuaikan batas atas kelembapan tanah sawah beririgasi (30 - 85 %) agar kondisi lembap sawah normal tidak dianggap error palsu
UPDATE td_device_sensors
SET ds_min_norm_value = 30, ds_max_norm_value = 85
WHERE ds_id IN ('soil_hum1', 'soil_hum2', 'soil_hum3', 'soil_hum4', 'soil_hum5', 'soil_hum6');

-- 5. Lengkapi teks rekomendasi tindakan agronomis yang sebelumnya NULL atau salah copy-paste (khusus pH & salinitas)
UPDATE td_device_sensors
SET max_danger_action = 'Lakukan pengairan pencucian (flushing) dengan air tawar, perbaiki saluran drainase, dan tunda pupuk berindeks garam tinggi (KCl)'
WHERE ds_id LIKE 'soil_salin%';

UPDATE td_device_sensors
SET min_danger_action = 'Aplikasikan kapur pertanian/dolomit (1-2 ton/ha) untuk menaikkan pH tanah masam dan atur drainase berkala',
    max_danger_action = 'Lakukan pengairan teratur (flushing), tambahkan bahan organik/kompos matang, dan gunakan pupuk sumber N bersifat masam fisiologis (ZA)'
WHERE ds_id LIKE 'soil_ph%';

-- 6. Perbaikan pemetaan koordinat lokasi & perangkat (tetap mempertahankan PL02 -> TELU0300 sebagai template siklus tanam Area 1 & 2 agar perhitungan HST berfungsi)
UPDATE tm_device SET site_id = 'SITE000' WHERE dev_id = 'TELU0300';
UPDATE tm_site SET site_lon = 110.68781333578296, site_lat = -7.609834670387249 WHERE site_id = 'SITE001';
UPDATE tm_plant SET pl_lat = -7.609834670387249, pl_lon = 110.68781333578296 WHERE pl_id = 'PL03';
UPDATE tm_plant SET dev_id = NULL WHERE pl_id = 'PL01';

-- 7. Optimasi Indeks Komposit Spasio-Temporal (Kontribusi Rumusan Masalah #1: Minimasi Latensi SQL Retrieval)
CREATE INDEX idx_ds_readdate ON tm_sensor_read (ds_id, read_date);


# Qwen Safety Classifier

Fine-tuning **Qwen3.5-0.8B-Instruct** dengan **QLoRA** untuk klasifikasi keamanan konten berbahasa Indonesia berdasarkan **UU ITE Pasal 28 Ayat 2**.

- **Input**: gambar atau frame video  
- **Output**: reasoning detail dalam Bahasa Indonesia + label `SAFE` / `UNSAFE`  
- **Klien**: Pemerintah Indonesia

---

## Struktur Dataset

```
dataset/
├── metadata.csv              ← sumber kebenaran utama
└── Kontent/
    ├── class_0/
    ├── class_1/
    ├── class_2/
    ├── class_3/
    └── class_4/
```

### Format metadata.csv

```
image_name,CLASSIFICATION,kategori,REASONING
1_126.jpg,UNSAFE,...,"Berdasarkan ..."
```

### Konvensi penamaan gambar

`X_Y.jpg` → file berada di folder `Kontent/class_X/` dengan nama `X_Y.jpg`

Contoh: `1_126.jpg` → `dataset/Kontent/class_1/1_126.jpg`

---

## Struktur Repositori

```
qwen-safety-classifier/
├── config/
│   ├── config_base.yaml        ← path dataset, prompt sistem, val_split
│   ├── config_trl.yaml         ← hyperparameter pelatihan TRL
│   └── config_unsloth.yaml     ← hyperparameter pelatihan Unsloth
├── requirements.txt
├── 00_extract_frames.py        ← video → tensor temporal (.pt)
├── 01_prepare_dataset.py       ← CSV + folder → train/val JSON
├── train/
│   ├── train_trl.py            ← pelatihan: HuggingFace TRL
│   └── train_unsloth.py        ← pelatihan: Unsloth (lebih hemat VRAM)
├── eval/
│   ├── eval_trl.py             ← evaluasi checkpoint TRL
│   └── eval_unsloth.py         ← evaluasi checkpoint Unsloth
├── merge_trl.py                ← gabungkan adapter LoRA ke model penuh
├── dataset/
│   ├── metadata.csv
│   ├── Kontent/class_0..4/
│   ├── train_data.json         ← dibuat oleh 01_prepare_dataset.py
│   └── val_data.json
└── output/
    ├── trl_checkpoint/
    ├── trl_merged/
    ├── unsloth_checkpoint/
    └── unsloth_merged/
```

---

## Alur Pipeline Lengkap

```
Dataset mentah ──────────────────────────────────┐
(class_0..4 + metadata.csv)                      │
                                                  │
Video (opsional) ──► 00_extract_frames.py ───────┤
                     └─ .pt tensor per video      │
                                                  ▼
                              01_prepare_dataset.py
                              (config_base.yaml)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            train_data.json                  val_data.json
                    │                               │
        ┌───────────┴──────────┐                   │
        ▼                      ▼                   │
  train_trl.py         train_unsloth.py ◄──────────┘
  (config_trl.yaml)    (config_unsloth.yaml)
        │                      │
        ▼                      ▼
  trl_checkpoint/      unsloth_checkpoint/
  final_adapter        final_adapter
        │                      │
        ▼                      ▼
  merge_trl.py         (auto-merge di Unsloth)
  trl_merged/          unsloth_merged/
        │                      │
        └──────────┬───────────┘
                   ▼
          eval_trl.py / eval_unsloth.py
                   │
                   ▼
          output/eval_results/
          eval_*.json
```

---

## Panduan Cepat

### 1. Instalasi dependensi

```bash
pip install -r requirements.txt
```

Untuk Unsloth (Kaggle/Colab CUDA 12.1):

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 2. (Opsional) Ekstraksi frame video

Jika dataset berisi file video di dalam folder kelas:

```bash
python 00_extract_frames.py --fps 1
```

Skrip ini membaca setiap video, mengambil sampel frame sesuai FPS yang ditentukan, menyimpannya sebagai tensor temporal `(T, C, H, W)` dalam format `.pt`, lalu menulis `metadata_videos.csv`. Format `.pt` mempertahankan hubungan antar-frame sehingga model dapat memanfaatkan jalur *temporal attention* milik Qwen3.5.

### 3. Persiapan dataset

```bash
# Verifikasi semua gambar ada di disk sebelum menulis JSON
python 01_prepare_dataset.py --verify
```

Skrip ini membaca `dataset/metadata.csv`, memetakan path gambar menggunakan konvensi `X_Y.jpg`, lalu menulis:
- `dataset/train_data.json` (80%)
- `dataset/val_data.json` (20%)

### 4. Pelatihan

**Opsi A — TRL standar (~6 GB VRAM)**

```bash
python train/train_trl.py
```

**Opsi B — Unsloth (~3 GB VRAM, 2–2.7× lebih cepat)**

```bash
python train/train_unsloth.py
```

**Mode smoke test** (uji cepat end-to-end tanpa data penuh):

```bash
python train/train_trl.py debug=on
python train/train_unsloth.py debug=on
```

Kedua skrip membaca semua hyperparameter dari file konfigurasi masing-masing.

### 5. (Opsional) Gabungkan adapter ke model penuh

Diperlukan hanya untuk deployment atau saat menggunakan `eval_trl.py --merged`:

```bash
python merge_trl.py \
  --adapter output/trl_checkpoint/final_adapter \
  --output  output/trl_merged
```

Model Unsloth secara otomatis menyimpan versi merged saat pelatihan selesai.

### 6. Evaluasi

```bash
# TRL — adapter saja
python eval/eval_trl.py --checkpoint output/trl_checkpoint/final_adapter

# TRL — model merged
python eval/eval_trl.py --checkpoint output/trl_merged --merged

# Unsloth
python eval/eval_unsloth.py --checkpoint output/unsloth_checkpoint/final_adapter

# Batasi ke N sampel saja
python eval/eval_trl.py --checkpoint output/trl_checkpoint/final_adapter --n 50
```

---

## Detail Model

| Parameter | Nilai |
|-----------|-------|
| Model dasar | Qwen3.5-0.8B-Instruct |
| Metode | QLoRA (base 4-bit + adapter 16-bit) |
| LoRA rank | 8, alpha 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Thinking mode | Nonaktif (output langsung) |
| Max new tokens | 128–256 |
| Bahasa output | Indonesia |

---

## Metrik Evaluasi (Urutan Prioritas)

| Metrik | Jenis | Alasan |
|--------|-------|--------|
| **Recall (UNSAFE)** | Klasifikasi | Prioritas — konten berbahaya tidak boleh lolos |
| F1-Score | Klasifikasi | Menangani ketidakseimbangan kelas |
| AUC-ROC | Klasifikasi | Tidak bergantung pada threshold |
| BERTScore | Kualitas reasoning | Evaluasi teks Indonesia |
| Train/Val Loss | Kesehatan pelatihan | Deteksi overfitting |

**Recall > Precision** — lebih baik alarm palsu daripada melewatkan konten berbahaya.

---

## Konfigurasi

Semua hyperparameter ada di folder `config/`. Bagian utama:

```yaml
# config_base.yaml — digunakan semua skrip
dataset:
  metadata_csv: "dataset/metadata.csv"
  image_root:   "dataset/Kontent"
  val_split:    0.2
  seed:         42

prompt:
  system: |
    Kamu adalah sistem klasifikasi konten yang membantu pemerintah Indonesia
    menegakkan UU ITE Pasal 28 Ayat 2 ...

# config_trl.yaml / config_unsloth.yaml — khusus pelatihan
model:
  name: "Qwen/Qwen3.5-0.8B"
  max_seq_length: 512

lora:
  r: 8
  alpha: 16

training:
  learning_rate: 2.0e-4
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

---

## Penanganan Video

Skrip `00_extract_frames.py` mengekstraksi video menjadi **tensor temporal** bukan frame gambar biasa. Ini penting karena:

- Model Qwen3.5 memiliki jalur *temporal attention* khusus (`pixel_values_videos` + `video_grid_thw`)
- Frame yang disimpan sebagai gambar terpisah kehilangan hubungan antar-waktu
- Format `.pt` menyimpan seluruh video dalam satu paket `(T, C, H, W)` sehingga model dapat memperhatikan gerakan dan konteks lintas-frame

Saat pelatihan dan evaluasi, `VLMDataCollator` (TRL) dan `VideoAwareCollator` (Unsloth) secara otomatis mendeteksi file `.pt` dan menyuntikkan tensor video ke dalam batch.

---

## Timeline Proyek

| Hari | Tugas |
|------|-------|
| 1–5 | Persiapan dataset (tim lain) + setup repositori + uji dengan 13 sampel |
| 6–7 | Latih v1: TRL + Unsloth |
| 8 | Evaluasi v1 |
| 9–10 | Latih v2 (hyperparameter yang disesuaikan) |
| 11–12 | Evaluasi akhir + perbandingan TRL vs Unsloth |
| 13–14 | Merge model + skrip inferensi + dokumentasi serah terima |

---

## Daftar Periksa Sebelum Mulai

- [ ] Buat Organisasi HuggingFace, atur repositori dataset ke private, undang semua anggota tim
- [ ] Konfirmasi dengan tim dataset: kolom CSV harus `image_name, CLASSIFICATION, kategori, REASONING`
- [ ] Jalankan `python 01_prepare_dataset.py --verify` dan konfirmasi 0 baris dilewati
- [ ] Periksa VRAM GPU: ≥3 GB untuk Unsloth, ≥6 GB untuk TRL

---

## Referensi

- [Qwen3.5-0.8B-Instruct](https://huggingface.co/Qwen/Qwen3.5-0.8B-Instruct)
- [TRL SFT Trainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [TRL VLM Cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl)
- [Unsloth Qwen3.5 Guide](https://unsloth.ai/docs/models/qwen3.5)
- [PEFT LoRA Docs](https://huggingface.co/docs/peft/en/conceptual_guides/lora)
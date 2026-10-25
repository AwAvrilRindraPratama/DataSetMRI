# Klasifikasi Multi-Kelas Tumor Otak Citra MRI Menggunakan ResNet50 dengan Interpretasi Grad-CAM

**Peneliti:** Aw. Avril Rindra Pratama (`2311102441032`)  
**Program Studi:** S1 Teknik Informatika — Universitas Muhammadiyah Kalimantan Timur (UMKT)  
**Pembimbing:** Abdul Rahim, S.Kom., M.Kom.

---

## Struktur Repositori

```text
DataSetMRI/
├── Sc/                             # Direktori Kode Program Penelitian (Source Code)
│   ├── train_resnet50_gradcam.py   # Kode utama eksperimen (5-Fold CV + Grad-CAM)
│   └── python.py                   # Salinan skrip utama
├── Training/                       # 1.280 citra MRI T1-CE (4 kelas @ 320 citra)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── Testing/                        # 320 citra MRI T1-CE (4 kelas @ 80 citra)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── requirements.txt                # Spesifikasi pustaka sesuai Tabel II.3 Proposal
└── README.md
```

---

## Cara Menjalankan Eksperimen

```bash
pip install -r requirements.txt
python Sc/train_resnet50_gradcam.py --dataset_dir . --output_dir ./outputs_5fold
```

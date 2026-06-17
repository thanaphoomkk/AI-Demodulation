# AI Demodulation 📡🤖

โปรเจค **Deep Learning สำหรับ Demodulation สัญญาณวิทยุ** ด้วยโมเดล **CNN + LSTM**  
รับ IQ samples เข้ามาแล้วถอดรหัสกลับเป็นบิต `0/1` และเปรียบเทียบกับ Classical Matched Filter

รองรับสัญญาณดิจิทัล 5 ประเภท: **BPSK, QPSK, QAM8, QAM16, QAM64**  
และ Analog: **AM, FM** (สร้าง/แสดงสัญญาณ)

🌐 **Demo:** https://ai-demodulation.onrender.com

---

## 📂 โครงสร้างโปรเจค

```
ai-demodulation/
├── app.py                # Flask Web App (API + routes)
├── generate_signal.py    # สร้างสัญญาณจำลอง + เพิ่ม AWGN noise
├── demod_classical.py    # Classical Demodulation (RRC Matched Filter + Euclidean Decision)
├── model.py              # โมเดล CNN + LSTM
├── train.py              # เทรนโมเดล
├── requirements.txt      # รายการ library
├── render.yaml           # config สำหรับ deploy บน Render.com
├── models/               # โมเดลที่เทรนแล้ว (.keras)
│   ├── bpsk_demod.keras
│   ├── qpsk_demod.keras
│   ├── qam8_demod.keras
│   ├── qam16_demod.keras
│   └── qam64_demod.keras
├── templates/
│   └── index.html        # Web UI (4 แท็บ)
└── results/              # กราฟผลการเทรน
```

---

## 🚀 วิธีรันบนเครื่อง

### 1. ติดตั้ง library

```bash
pip install -r requirements.txt
```

### 2. รัน Web App

```bash
python app.py
```

เปิดเบราว์เซอร์ที่ `http://localhost:5000`

### 3. เทรนโมเดลใหม่ (ถ้าต้องการ)

```bash
python train.py --mod bpsk
python train.py --mod qpsk
python train.py --mod qam8
python train.py --mod qam16
python train.py --mod qam64 --epochs 40 --n_train 8000
```

ผลลัพธ์ → `models/<mod>_demod.keras` และ `results/<mod>_training.png`

---

## 🧠 หลักการทำงาน

### สัญญาณ IQ (generate_signal.py)

```
bits → Symbol Mapper → RRC Pulse Shaping → upsample (sps=8) → + AWGN noise → IQ signal
```

- สุ่มบิต → แปลงเป็น complex symbol ตาม constellation แต่ละ modulation
- ใส่ Root-Raised-Cosine (RRC) filter เพื่อ pulse shaping
- เพิ่ม AWGN noise ตาม SNR ที่กำหนด (ฝึกด้วย -5 ถึง 25 dB)

---

### Classical Demodulation (demod_classical.py)

```
IQ noisy
  → RRC Matched Filter    (RRC × RRC = RC → ไม่มี ISI + maximize SNR)
  → Downsample            (เอา sample ตรงกลาง symbol)
  → Euclidean Distance    (วัดระยะหาจุด constellation ที่ใกล้สุด)
  → bits
```

**ข้อจำกัด:**
- ตัดสินใจ **ทีละ symbol โดดๆ** ไม่สนบริบทรอบข้าง
- ใช้สูตรคณิตศาสตร์คงที่ ปรับตัวกับ noise รูปแบบอื่นไม่ได้

---

### AI Demodulation — CNN + LSTM (model.py)

```
IQ noisy (512 samples, 2 channels)
  → Conv1D × 2 + BatchNorm        (ดึง feature pattern จากคลื่น)
  → Reshape → 64 time steps
  → Bidirectional LSTM             (อ่านบริบท ← → ระหว่าง symbol)
  → TimeDistributed Dense + sigmoid
  → bits (ทุก symbol พร้อมกัน)
```

**ข้อได้เปรียบ:**
- LSTM เห็น **บริบทของ symbol รอบข้าง** — ช่วยตัดสินใจที่ SNR ต่ำได้ดีกว่า
- เรียนรู้จาก 5,000–8,000 ตัวอย่าง ที่ SNR = -5 ถึง 25 dB
- ไม่ต้องรู้สูตร RRC หรือ constellation ล่วงหน้า — หาเองจากข้อมูล

---

### เปรียบเทียบ AI vs Classical

| | Classical | AI (CNN+LSTM) |
|--|-----------|---------------|
| ใช้ข้อมูลจาก | symbol เดียว | 512 samples (64 symbols) |
| ตัดสินใจด้วย | สูตร math คงที่ | pattern ที่เรียนรู้ |
| SNR สูง (>15 dB) | ดีมาก | ดีพอกัน |
| SNR ต่ำ (-5 ถึง 5 dB) | BER สูง | BER ต่ำกว่า (ใช้บริบทช่วย) |
| ปรับตัวกับ noise ได้ไหม | ไม่ได้ | ได้ (เรียนรู้มาแล้ว) |

---

## 📊 ผลการเทรนโมเดล

โมเดลทั้งหมดเทรนด้วย SNR range **-5 ถึง 25 dB** เพื่อให้ทำงานได้ดีในทุกสภาวะ

| Modulation | bits/symbol | Accuracy | BER |
|------------|-------------|----------|-----|
| BPSK | 1 | 99.95% | 0.00048 |
| QPSK | 2 | 99.44% | 0.0056 |
| QAM8 | 3 | 97.70% | 0.023 |
| QAM16 | 4 | 96.66% | 0.033 |
| QAM64 | 6 | 92.85% | 0.072 |

> BER ของ QAM64 สูงกว่า BPSK เพราะมี 64 constellation points อยู่ใกล้กัน noise นิดเดียวก็ผิดได้

---

## 🌐 Web App — 4 แท็บ

### 1. สร้างสัญญาณจำลอง
- เลือก Modulation, SNR, จำนวนสัญลักษณ์, Carrier freq
- แสดง waveform, constellation diagram, decode result bar chart
- เปรียบเทียบบิต AI vs Classical vs Source พร้อม BER

### 2. อัปโหลดไฟล์ IQ
- รับไฟล์ `.csv` รูปแบบ `I,Q` ต่อบรรทัด
- ถอดรหัสด้วย AI และ Classical

### 3. 🔬 Waveform Lab
- แสดง **3 panel**: clean signal + bit labels, noisy signal, decode result bar chart
- เปรียบเทียบบิตต้นทาง / AI decoded / Classical decoded พร้อม diff highlight
- พล็อต **BER vs SNR curve** อัตโนมัติเมื่อกด Run

### 4. 📊 Comparison
- **Real-time BER**: ปรับ SNR → เห็นตารางทุก modulation ทันที ว่า AI หรือ Classical ดีกว่า
- **BER vs SNR ทุก Mod**: กราฟ 5 subplot เปรียบเทียบ AI vs Classical ทุก modulation พร้อมกัน

---

## ⚙️ Architecture โมเดล

```
Input:  (None, 512, 2)          ← 512 samples, 2 channels (I, Q)
Conv1D(64, kernel=5) + BN
Conv1D(64, kernel=5) + BN
Reshape → (None, 64, 512)       ← แบ่งเป็น 64 time steps (1 symbol ต่อ step)
Bidirectional LSTM(128)
Dropout(0.2)
TimeDistributed Dense(64)
TimeDistributed Dense(bps) + sigmoid
Output: (None, 64, bps)         ← บิตของทุก symbol พร้อมกัน
```

- **Loss:** Binary Crossentropy
- **Optimizer:** Adam (lr=0.001, ReduceLROnPlateau)
- **Early Stopping:** patience=6 บน val_binary_accuracy
- **Parameters:** ~694K (2.65 MB ต่อโมเดล)

---

## 🔧 Hyperparameters

| Parameter | Value | ความหมาย |
|-----------|-------|----------|
| `sps` | 8 | samples ต่อสัญลักษณ์ |
| `symbols` | 64 | สัญลักษณ์ต่อ 1 ตัวอย่าง |
| `SNR range (train)` | -5 ถึง 25 dB | ครอบคลุมทุกสภาวะ noise |
| `RRC beta` | 0.35 | Roll-off factor |

---

## ⚠️ หมายเหตุ

- AM/FM เป็น analog modulation ใช้สำหรับแสดงสัญญาณเท่านั้น ไม่มีโมเดล AI
- ถ้า Render ไม่มี TensorFlow จะ fallback ใช้ Classical อัตโนมัติ
- โมเดลต้องการ input ขนาด 512 samples — ถ้าสัญญาณสั้นกว่า จะ zero-pad อัตโนมัติ

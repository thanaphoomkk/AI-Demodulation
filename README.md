# AI Demodulation 📡🤖

โปรเจค **Deep Learning สำหรับ Demodulation สัญญาณวิทยุ** ด้วยโมเดล **CNN + LSTM**
รับ IQ samples เข้ามาแล้วถอดรหัสกลับเป็นบิต `0/1`

รองรับสัญญาณ 5 ประเภท: **BPSK, QPSK, QAM16, AM, FM**

---

## 📂 โครงสร้างโปรเจค

```
ai-demodulation/
├── requirements.txt      # รายการ library
├── install_deps.py       # ติดตั้ง library อัตโนมัติ
├── generate_signal.py    # สร้างสัญญาณจำลอง + เพิ่ม noise
├── model.py              # โมเดล CNN + LSTM
├── train.py              # เทรนโมเดล + วาดกราฟ loss/accuracy/BER
├── test.py               # ทดสอบกับสัญญาณทุกประเภท + แสดงผลความแม่นยำ
├── models/               # โมเดลที่เทรนแล้ว (สร้างอัตโนมัติ)
└── results/              # กราฟผลลัพธ์ (สร้างอัตโนมัติ)
```

---

## 🚀 วิธีรันโปรเจค

### 1. ติดตั้ง library (อัตโนมัติ)

เลือกวิธีใดวิธีหนึ่ง:

```bash
python install_deps.py
```

หรือ

```bash
pip install -r requirements.txt
```

Library ที่ใช้: `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `tensorflow`

> 💡 ไม่จำเป็นต้องมี GPU ก็รันได้ (TensorFlow จะใช้ CPU อัตโนมัติ แต่จะช้ากว่า)

### 2. ทดลองสร้างสัญญาณ (ตัวเลือก)

```bash
python generate_signal.py
```

### 3. เทรนโมเดล

```bash
python train.py                 # ค่าเริ่มต้น = QPSK
python train.py --mod bpsk      # เทรน BPSK
python train.py --mod qam16     # เทรน 16-QAM
python train.py --mod qpsk --epochs 30 --batch 128
```

ผลลัพธ์หลังเทรน:
- โมเดล → `models/<mod>_demod.keras`
- กราฟ loss / accuracy / BER → `results/<mod>_training.png`

### 4. ทดสอบโมเดล

```bash
python test.py                  # ทดสอบทุกโมเดลที่เทรนไว้
python test.py --mod qpsk       # ทดสอบเฉพาะ QPSK
```

ผลลัพธ์:
- ภาพสัญญาณทุกประเภท → `results/all_signals.png`
- กราฟ BER vs SNR → `results/ber_vs_snr.png`
- ตาราง Accuracy / BER ที่ระดับ SNR ต่าง ๆ (แสดงบนหน้าจอ)

---

## 🧠 รายละเอียดทางเทคนิค

### สัญญาณ (`generate_signal.py`)
- **Digital (BPSK/QPSK/QAM16):** สุ่มบิต → map เป็นสัญลักษณ์เชิงซ้อน → upsample + Root-Raised-Cosine pulse shaping → เพิ่ม AWGN ตามค่า SNR
- **Analog (AM/FM):** สร้างสัญญาณข้อความจำลอง แล้ว modulate เป็น complex baseband
- IQ ถูกเก็บเป็น 2 ช่อง `[I, Q]`

### โมเดล (`model.py`)
```
Input (T*sps, 2)
   → Conv1D ×2 + BatchNorm        (ดึงฟีเจอร์จากรูปคลื่น)
   → Reshape เป็นราย symbol (T)
   → Bidirectional LSTM           (เรียนรู้ลำดับเวลา)
   → TimeDistributed Dense (sigmoid)
Output (T, bits_per_symbol)        → บิต 0/1
```
- Loss: `binary_crossentropy`
- Metric: `binary_accuracy`  →  **BER = 1 − accuracy**

### พารามิเตอร์เริ่มต้น
| พารามิเตอร์ | ค่า | ความหมาย |
|---|---|---|
| `sps` | 8 | samples ต่อสัญลักษณ์ |
| `symbols` | 64 | สัญลักษณ์ต่อ 1 ตัวอย่าง |
| `SNR (train)` | 0–20 dB | สุ่มแต่ละตัวอย่างให้ทนต่อ noise หลายระดับ |

---

## 📊 ผลลัพธ์ที่คาดหวัง

- กราฟ **Loss** ลดลงเรื่อย ๆ และ **Accuracy** เพิ่มขึ้นตาม epoch
- กราฟ **BER vs SNR** ลดลงเมื่อ SNR สูงขึ้น (พฤติกรรมตามทฤษฎีการสื่อสาร)
- BPSK/QPSK มักได้ BER ต่ำกว่า QAM16 ที่ SNR เดียวกัน (เพราะ QAM16 หนาแน่นกว่า)

---

## ⚠️ หมายเหตุ

- AM/FM เป็น analog modulation จึงใช้ในส่วน **สร้าง/แสดงภาพสัญญาณ** เป็นหลัก
  ส่วนการถอดรหัสเป็นบิต (demodulation ด้วย AI) จะทำกับ digital modulation (BPSK/QPSK/QAM16)
- ปรับขนาดชุดข้อมูล / epoch ได้ตามทรัพยากรเครื่องด้วย argument ของ `train.py`

---

## 🔧 ขั้นตอนแบบเร็ว (Quick Start)

```bash
python install_deps.py          # 1. ติดตั้ง
python train.py --mod qpsk      # 2. เทรน
python test.py --mod qpsk       # 3. ทดสอบ
```

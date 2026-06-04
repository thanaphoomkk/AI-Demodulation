"""
model.py
========
โมเดล CNN + LSTM สำหรับ AI Demodulation

แนวคิด:
    - Input  : IQ samples รูปร่าง (T*sps, 2)   [I, Q]
    - CNN    : ดึง local features จากรูปคลื่น IQ
    - Reshape: จัดกลุ่มตัวอย่างกลับเป็นราย "สัญลักษณ์" (T สัญลักษณ์)
    - LSTM   : เรียนรู้ความสัมพันธ์เชิงเวลาในลำดับสัญลักษณ์
    - Output : บิต 0/1 ของแต่ละสัญลักษณ์ (sigmoid) -> (T, bits_per_symbol)

โมเดลถูกเทรนแบบ multi-label ต่อสัญลักษณ์ ด้วย binary_crossentropy
"""

from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(symbols_per_example: int = 64,
                sps: int = 8,
                bits_per_symbol: int = 2,
                cnn_filters: int = 64,
                lstm_units: int = 128) -> tf.keras.Model:
    """
    สร้างโมเดล CNN+LSTM

    พารามิเตอร์:
        symbols_per_example : จำนวนสัญลักษณ์ต่อ 1 ตัวอย่าง (T)
        sps                 : samples ต่อสัญลักษณ์
        bits_per_symbol     : จำนวนบิตที่ output ต่อสัญลักษณ์
    """
    T = symbols_per_example
    input_len = T * sps

    inputs = layers.Input(shape=(input_len, 2), name="iq_input")

    # ---------- CNN: ดึงฟีเจอร์จากรูปคลื่น ----------
    x = layers.Conv1D(cnn_filters, kernel_size=5, padding="same",
                      activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(cnn_filters, kernel_size=5, padding="same",
                      activation="relu")(x)
    x = layers.BatchNormalization()(x)
    # หลัง CNN: shape = (input_len, cnn_filters)

    # ---------- Reshape: รวม sps samples ของแต่ละสัญลักษณ์เข้าด้วยกัน ----------
    # (T*sps, cnn_filters) -> (T, sps*cnn_filters)
    x = layers.Reshape((T, sps * cnn_filters))(x)

    # ---------- LSTM: เรียนรู้ลำดับเวลา ----------
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True)
    )(x)
    x = layers.Dropout(0.3)(x)

    # ---------- Output: บิต 0/1 ต่อสัญลักษณ์ ----------
    x = layers.TimeDistributed(layers.Dense(64, activation="relu"))(x)
    outputs = layers.TimeDistributed(
        layers.Dense(bits_per_symbol, activation="sigmoid"), name="bit_output"
    )(x)

    model = models.Model(inputs, outputs, name="CNN_LSTM_Demodulator")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()

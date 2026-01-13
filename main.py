import uvicorn
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
import os

# --- BAGIAN 1: PENGATURAN LOKASI MODEL ---
# Kita gunakan Absolute Path agar file .h5 pasti ketemu
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Pastikan nama file di folder model adalah 'model_tbc_final.h5'
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_tbc_final.h5")

print("="*50)
print(f"Versi TensorFlow: {tf.__version__}")
print(f"Target Model    : {MODEL_PATH}")

app = FastAPI(title="API Deteksi TBC (Final)")
model = None

# --- BAGIAN 2: LOAD MODEL ---
try:
    if os.path.exists(MODEL_PATH):
        print("✅ FILE DITEMUKAN! Sedang memuat model...")
        # compile=False digunakan agar tidak error masalah optimizer
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("🚀 SUKSES! Model berhasil dimuat dan siap digunakan.")
    else:
        print("❌ FILE TIDAK ADA.")
        print(f"   Pastikan file ada di: {MODEL_PATH}")
except Exception as e:
    print("❌ ERROR SAAT LOAD MODEL.")
    print(f"Pesan Error: {e}")

print("="*50)

# --- BAGIAN 3: FUNGSI PROSES GAMBAR ---
def proses_gambar(file_bytes):
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    
    # PENTING: Ubah ukuran ke 320x320 (Sesuai error shape tadi)
    img = img.resize((320, 320)) 
    
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- BAGIAN 4: ENDPOINT PREDIKSI ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "MODEL BELUM SIAP. Cek terminal server."}
    
    try:
        contents = await file.read()
        processed = proses_gambar(contents)
        prediction = model.predict(processed)
        score = float(prediction[0][0])
        
        # --- PERBAIKAN LOGIKA (DIBALIK) ---
        # Berdasarkan tes tadi: Skor 99% dianggap Normal padahal gambar TBC.
        # Artinya Logika Lama Terbalik.
        # KITA TUKAR POSISINYA SEKARANG:
        
        if score > 0.5:
            # Jika skor tinggi, berarti Normal (Sehat)
            label = "Normal (Sehat)"
            confidence = score * 100
        else:
            # Jika skor rendah, berarti TBC (Sakit)
            label = "TBC (Sakit)"
            confidence = (1 - score) * 100
            
        return {
            "filename": file.filename,
            "prediksi": label,
            "keyakinan": f"{confidence:.2f}%"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
import requests

# Ganti dengan nama gambar yang ada di folder Anda
FILE_GAMBAR = "Normal-95.png" 

url = "http://127.0.0.1:8000/predict"

try:
    print("⏳ Sedang mengirim gambar ke API...")
    with open(FILE_GAMBAR, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
    
    print("\n=== HASIL API ===")
    print(response.json())
    
except FileNotFoundError:
    print(f"❌ Error: File gambar '{FILE_GAMBAR}' tidak ditemukan! Masukkan gambar dulu.")
except Exception as e:
    print(f"❌ Error koneksi: {e}")
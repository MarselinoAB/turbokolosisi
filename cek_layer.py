import tensorflow as tf
import os

# Ganti dengan nama file model Anda yang sebenarnya
model_path = os.path.join("Model", "densenet_model.keras") 

try:
    model = tf.keras.models.load_model(model_path)
    print("\n=== 10 Layer Terakhir di Model Anda ===")
    # Kita lihat 10 layer terakhir untuk mencari layer Conv/Concat terakhir
    for i, layer in enumerate(model.layers[-20:]): 
        print(f"{i}. {layer.name} ({layer.__class__.__name__})")
        
    print("\nCARI NAMA LAYER DI ATAS YANG TIPE-NYA 'Concatenate' ATAU 'Conv2D' TERAKHIR SEBELUM 'GlobalAveragePooling'")
except Exception as e:
    print(f"Error loading model: {e}")
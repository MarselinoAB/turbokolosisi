import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import base64

# DenseNet121 biasanya menggunakan input 224x224
IMG_SIZE = (224, 224)

def preprocess_image(image_bytes):
    """Mengubah bytes gambar menjadi array numpy untuk model."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    
    # Preprocessing khusus DenseNet (biasanya rescaling 1/255 atau tf.keras.applications.densenet.preprocess_input)
    # Di sini kita gunakan rescaling standar 0-1 seperti notebook umumnya
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, np.array(img) # Kembalikan juga gambar asli untuk overlay

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Fungsi inti untuk membuat Grad-CAM Heatmap.
    """
    # Buat model yang memetakan input ke layer konvolusi terakhir dan output prediksi
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Hitung gradien output kelas terhadap feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalisasi heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_img, alpha=0.4):
    """
    Menggabungkan heatmap dengan gambar asli dan mengubahnya jadi base64 string.
    """
    # Resize heatmap agar sesuai dengan gambar asli
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Konversi gambar asli ke format OpenCV (BGR)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

    # Gabungkan (overlay)
    superimposed_img = heatmap * alpha + original_img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

    # Encode ke JPG lalu ke Base64 string agar bisa dikirim via API
    _, buffer = cv2.imencode('.jpg', superimposed_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64
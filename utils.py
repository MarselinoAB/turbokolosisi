import numpy as np
from PIL import Image
import io
import tensorflow as tf
import cv2
import base64

# PENTING: Jika error shape, ganti angka ini jadi (512, 512)
IMG_SIZE = (512, 512)

def preprocess_image(image_bytes):
    # Buka gambar
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Resize sesuai input model
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    # Normalisasi (0-1)
    img_array = img_array / 255.0
    # Tambah dimensi (1, 224, 224, 3)
    img_expanded = np.expand_dims(img_array, axis=0)
    return img_expanded, np.array(img)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Buat model gradien
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_img, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    superimposed_img = heatmap * alpha + original_img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    _, buffer = cv2.imencode('.jpg', superimposed_img)
    return base64.b64encode(buffer).decode('utf-8')
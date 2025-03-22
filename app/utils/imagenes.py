import base64
from fastapi import FastAPI, HTTPException
import numpy as np
import cv2
from PIL import Image
import io
import tensorflow as tf

# ===============================
# Función para decodificar Base64 a imagen
# ===============================
def decode_base64_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al decodificar la imagen: {str(e)}")

# ===============================
# Función para preprocesar la imagen
# ===============================
def preprocess_image(image, target_size=(299, 299)):
    img = cv2.resize(image, target_size)
    img = img.astype("float32") / 255.0  # Normalizar
    img = np.expand_dims(img, axis=0)  # Expandir dimensiones para batch
    return img

# Función para aplicar Grad-CAM

def make_gradcam_heatmap(img_array, model):
    # Obtener la capa base InceptionV3
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.models.Model):
            base_model = layer
            break
        
    # Obtener la última capa convolucional
    last_conv_layer = base_model.get_layer("mixed10")

    # Modelo para obtener salida de capa convolucional
    grad_model_conv = tf.keras.models.Model(
        inputs = base_model.inputs,
        outputs = last_conv_layer.output
    )

    with tf.GradientTape() as tape:
        # Obtener activación de última capa convolucional
        conv_output = grad_model_conv(img_array)
        tape.watch(conv_output)

        # Pasar por el resto del modelo
        x = conv_output
        for layer in model.layers:
            if isinstance(layer, tf.keras.models.Model):
                continue
            x = layer(x)

        preds = x
        class_channel = preds[:, 0]  # Para clasificación binaria

    # Calcular gradientes
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Generar mapa de calor
    heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalizar
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.5):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)





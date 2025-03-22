
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

from tensorflow.keras.models import load_model
from utils.imagenes import decode_base64_image, preprocess_image

class CancerClassifier:
    def __init__(self, model_path: str):
        """
        Inicializa el clasificador cargando el modelo.

        :param model_path: Ruta del archivo del modelo (.h5).
        """
        self.model = load_model(model_path)



    def preprocess_image_model(self, image_base64: str):
        """
        Preprocesa una imagen para hacerla compatible con el modelo.

        :param image: Imagen en formato PIL.
        :return: Imagen preprocesada como un array de NumPy.
        """
        # Convertir imagen Base64 a NumPy array
        image = decode_base64_image(image_base64)

        # Preprocesar imagen
        imagen_procesada = preprocess_image(image)
        return imagen_procesada

    def predict(self, img_array: np.ndarray):
        """
        Realiza una predicción en la imagen preprocesada.

        :param img_array: Imagen preprocesada como un array de NumPy.
        :return: Diccionario con el resultado de la predicción.
        """
        # Hacer la predicción
        preds = self.model.predict(img_array, verbose=0)
        pred_prob = float(preds[0][0])
        pred_class = 1 if pred_prob > 0.5 else 0
        etiqueta = 'Maligno' if pred_class == 1 else 'Benigno'

        return  pred_prob , etiqueta

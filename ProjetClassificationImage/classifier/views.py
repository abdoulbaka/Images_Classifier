from django.shortcuts import render

import os
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Charger le modèle au démarrage du serveur
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'chat_chien_model.h5')
model = load_model(MODEL_PATH)

# Fonction pour prédire une image uploadée
def predict_image(request):
    if request.method == 'POST' and request.FILES['image']:
        img = request.FILES['image']
        fs = FileSystemStorage()
        img_path = fs.save(img.name, img)  # Sauvegarder l’image
        img_path = fs.url(img_path)

        # 🔹 Charger l’image et la préparer pour la prédiction
        img_full_path = os.path.join(fs.location, img.name)
        img = image.load_img(img_full_path, target_size=(224, 224))  # Redimensionner
        img_array = image.img_to_array(img)  # Convertir en tableau
        img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
        img_array = preprocess_input(img_array)  # Normaliser comme pour ResNet

        # 🔹 Faire la prédiction
        prediction = model.predict(img_array)
        class_name = "Chien" if prediction[0][0] > 0.5 else "Chat"  # Sigmoid → binaire

        return render(request, 'classifier/result.html', {'class_name': class_name, 'img_path': img_path})

    return render(request, 'classifier/upload.html')


def home(request):
    return render(request, 'classifier/home.html') 


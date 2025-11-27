# backend/main.py

import uvicorn
import base64
import numpy as np
import io
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

# --- CONFIGURACIÓN DEL MODELO ---
TFLITE_MODEL_PATH = "modelos/ml_model.tflite"
INPUT_SHAPE = (224, 224)  # mismo tamaño usado en entrenamiento

CLASS_LABELS = {
    0: "Lesión Benigna (Bajo Riesgo)",
    1: "Lesión Maligna (Alto Riesgo)",
}

# --- CARGA DEL MODELO TFLITE ---
try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"✅ Modelo TFLite cargado correctamente: {TFLITE_MODEL_PATH}")
except Exception as e:
    print(f"ERROR cargando TFLite: {e}")
    interpreter = None

# --- FASTAPI ---
app = FastAPI(title="DermaCheck AI API", version="1.0.0")

# --- MIDDLEWARE CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELO DE DATOS DE ENTRADA ---
class PielCheckRequest(BaseModel):
    image_base64: str  # Imagen codificada en Base64
    symptoms: str
    description: str

# --- PREPROCESAMIENTO DE IMAGEN ---
def preprocess_image(image_base64: str) -> np.ndarray:
    """
    Convierte la imagen base64 en un array float32 normalizado [0,1] 
    y con shape (1, 224, 224, 3) para TFLite.
    """
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Modelo de IA no cargado.")

    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(INPUT_SHAPE)
        image_array = np.array(image, dtype=np.float32) / 255.0  # Normalización [0,1]
        image_array = np.expand_dims(image_array, axis=0)  # Añadir batch
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando la imagen: {e}")

# --- INFERENCIA TFLITE ---
def run_tflite_inference(image_array: np.ndarray, symptoms: str, description: str) -> Dict[str, Any]:
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Modelo de IA no cargado.")

    try:
        input_index = input_details[0]['index']
        interpreter.set_tensor(input_index, image_array.astype(np.float32))
        interpreter.invoke()

        output_index = output_details[0]['index']
        predictions = interpreter.get_tensor(output_index)

        probability_of_malignancy = float(predictions[0][0])
        predicted_class_index = 1 if probability_of_malignancy >= 0.5 else 0
        confidence = probability_of_malignancy if predicted_class_index == 1 else (1.0 - probability_of_malignancy)
        diagnosis_label = CLASS_LABELS.get(predicted_class_index, "Desconocido")
        is_high_risk = (predicted_class_index == 1)

        full_description = (
            f"Diagnóstico Principal: **{diagnosis_label}** (Confianza: {confidence*100:.2f}%). "
            f"Síntomas reportados: {symptoms}. Descripción adicional: {description}. "
            f"Evaluación de Riesgo: {'ALTO' if is_high_risk else 'BAJO'}. "
            f"Nota: evaluación de IA, no sustituye diagnóstico médico."
        )

        return {
            "diagnosisResult": diagnosis_label,
            "confidenceScore": confidence,
            "isHighRisk": is_high_risk,
            "fullResultDescription": full_description
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {e}")

# --- ENDPOINT PRINCIPAL ---
@app.post("/skin-check/analyze")
async def analyze_skin_lesion(request: PielCheckRequest):
    image_array = preprocess_image(request.image_base64)
    result = run_tflite_inference(
        image_array,
        symptoms=request.symptoms,
        description=request.description
    )
    return result

# --- EJECUTAR SERVIDOR ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000) # uvicorn main:app --host 0.0.0.0 --port 10000
    #uvicorn main:app --host 127.0.0.1 --port 8000
    
    
    # Cuando tengo el backend actualizado pasos primero crear un entorno (ya lo tengo)
    # instalar las librerias necesarias para el backend
    # Generar el requirement.txt con: pip freeze > requirements.txt

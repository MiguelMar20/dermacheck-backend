# app/main.py

import os
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

# -----------------------------------------------------------
# CONFIGURACIÓN DEL MODELO
# -----------------------------------------------------------

# Ruta correcta dentro del deploy en Render
TFLITE_MODEL_PATH = "app/modelos/ml_model.tflite"

INPUT_SHAPE = (224, 224)

CLASS_LABELS = {
    0: "Lesión Benigna (Bajo Riesgo)",
    1: "Lesión Maligna (Alto Riesgo)",
}

# -----------------------------------------------------------
# CARGAR MODELO TFLITE
# -----------------------------------------------------------

try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"✅ Modelo TFLite cargado correctamente: {TFLITE_MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR cargando modelo TFLite: {e}")
    interpreter = None

# -----------------------------------------------------------
# FASTAPI
# -----------------------------------------------------------

app = FastAPI(title="DermaCheck AI API", version="1.0.0")

# CORS (permite acceso desde Flutter)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------
# ENDPOINT DE PRUEBA
# -----------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "DermaCheck backend activo!"}

# -----------------------------------------------------------
# MODELO DE REQUEST
# -----------------------------------------------------------

class PielCheckRequest(BaseModel):
    image_base64: str
    symptoms: str
    description: str

# -----------------------------------------------------------
# PREPROCESAMIENTO IMAGEN
# -----------------------------------------------------------

def preprocess_image(image_base64: str) -> np.ndarray:
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Modelo de IA no cargado.")

    try:
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(INPUT_SHAPE)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando la imagen: {e}")

# -----------------------------------------------------------
# INFERENCIA
# -----------------------------------------------------------

def run_tflite_inference(image_array: np.ndarray, symptoms: str, description: str) -> Dict[str, Any]:
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Modelo de IA no cargado.")

    try:
        interpreter.set_tensor(input_details[0]["index"], image_array.astype(np.float32))
        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]["index"])

        probability = float(predictions[0][0])
        predicted_class = 1 if probability >= 0.5 else 0
        confidence = probability if predicted_class == 1 else 1 - probability
        label = CLASS_LABELS[predicted_class]

        description_full = (
            f"Diagnóstico: {label} (Confianza: {confidence*100:.2f}%). "
            f"Síntomas: {symptoms}. Descripción: {description}. "
            f"Riesgo: {'ALTO' if predicted_class == 1 else 'BAJO'}. "
            f"Nota: IA, no sustituye diagnóstico médico."
        )

        return {
            "diagnosisResult": label,
            "confidenceScore": confidence,
            "isHighRisk": predicted_class == 1,
            "fullResultDescription": description_full,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {e}")

# -----------------------------------------------------------
# ENDPOINT PRINCIPAL
# -----------------------------------------------------------

@app.post("/skin-check/analyze")
async def analyze_skin_lesion(request: PielCheckRequest):
    image_array = preprocess_image(request.image_base64)
    return run_tflite_inference(image_array, request.symptoms, request.description)

# -----------------------------------------------------------
# ARRANCAR SERVIDOR (Render usa $PORT)
# -----------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

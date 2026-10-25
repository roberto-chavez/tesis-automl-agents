import os
from google import genai

# Configura tu cliente (reemplaza tu clave o asegúrate de usar la variable de entorno)
client = genai.Client()

model_name = "gemini-3.8-flash"

try:
    print(f"Enviando mensaje de prueba al modelo {model_name}...")
    
    response = client.models.generate_content(
        model=model_name,
        contents="Hola Gemini, dime cual es la mejor cualidad de Makima.",
    )
    
    print("\n--- Respuesta del modelo ---")
    print(response.text)
    print("----------------------------\n")
    print("¡Todo funciona correctamente!")

except Exception as e:
    print(f"\n❌ Ocurrió un error: {e}")
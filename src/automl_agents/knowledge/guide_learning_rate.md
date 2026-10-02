# Rangos de tasa de aprendizaje

Con backbone congelado y optimizador Adam, la cabeza nueva admite tasas de aprendizaje relativamente altas, entre 3e-4 y 3e-3. Al entrenar toda la red desde cero conviene un rango menor, entre 1e-4 y 1e-3. Tasas de 1e-2 suelen ser inestables. La tasa de aprendizaje es el hiperparámetro con mayor impacto y debe buscarse en escala logarítmica.

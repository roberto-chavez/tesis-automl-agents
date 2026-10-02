# Descarte temprano y parada anticipada

Detener las configuraciones con mal desempeño parcial ahorra la mayor parte del cómputo de una búsqueda. Estrategias tipo Hyperband asignan pocos recursos a muchas configuraciones y continúan solo con las mejores. La parada anticipada por paciencia sobre la métrica de validación evita épocas inútiles y restaura los mejores pesos.

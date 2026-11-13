# Consumo de recursos frente a línea base (R4.1, R4.3)

- Entrenamientos ejecutados: **1**, de los cuales **0** (0.0 %) se podaron temprano.
- Épocas usadas en todos los intentos: **1**. En intentos completados/podados: 1 de 1 máximas; reducción **0.0 %** por poda y parada anticipada. Los fallos y cancelaciones no cuentan como ahorro por parada anticipada.
- Tiempo medido: 15.2 s (15.25 s/época en promedio).

| Línea base (estimada) | Configs | Épocas | Horas est. | Reducción de épocas |
|---|---|---|---|---|
| Búsqueda exhaustiva, espacio completo | 990 | 990 | 4.19 | 99.9 % |
| Búsqueda exhaustiva, espacio acotado por RAG | 24 | 24 | 0.1 | 95.8 % |

> Líneas base estimadas: configuraciones x épocas máximas x costo medio por época medido. El costo real depende de las arquitecturas y recursos; no se midió una búsqueda exhaustiva.

## Tiempos reales por etapa (segundos)
| Etapa | Segundos medidos |
|---|---|
| perfil | 0.17207969999981287 |
| rag | 0.9539356000004773 |
| candidatas | 45.78581049999957 |
| entrenar | 43.651580300000205 |
| controlar | 0.00014229999942472205 |
| evaluar | 4.1099005999994915 |
| benchmark | 0.023656000000300992 |
| reportar | 0.49740079999992304 |
| empaquetar | 15.844442600000548 |
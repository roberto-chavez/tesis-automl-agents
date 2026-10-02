# Desbalance de clases

Cuando una clase tiene mucho más ejemplos que otra, la exactitud global engaña. Se recomienda ponderar la función de pérdida con pesos de clase inversamente proporcionales a la frecuencia, usar aumento de datos en las clases minoritarias y evaluar con F1 macro, matriz de confusión y recall por clase en lugar de solo exactitud.

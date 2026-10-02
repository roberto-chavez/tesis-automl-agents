# Transferencia de aprendizaje con pesos ImageNet

Reutilizar un backbone preentrenado en ImageNet y entrenar solo una cabeza de clasificación nueva es la estrategia más eficiente cuando hay pocas imágenes por clase. Se congela el backbone, se agrega pooling global, dropout y una capa densa softmax. Reduce mucho el tiempo de entrenamiento y la necesidad de datos etiquetados frente a entrenar desde cero.

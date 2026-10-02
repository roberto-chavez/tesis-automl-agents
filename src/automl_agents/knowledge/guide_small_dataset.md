# Datasets pequeños: pocas imágenes por clase

Con menos de unos 1000 imágenes de entrenamiento o pocas decenas por clase, conviene elegir arquitecturas ligeras con transferencia de aprendizaje, usar aumento de datos y dropout alto de 0.2 a 0.4 para evitar sobreajuste. Los modelos grandes tienden a sobreajustar con pocos datos y no aportan mejora. Usar validación estratificada.

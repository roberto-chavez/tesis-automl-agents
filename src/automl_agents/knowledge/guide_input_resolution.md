# Resolución de entrada

Cada arquitectura tiene una resolución nativa, 224x224 en la mayoría y 299x299 en Xception e InceptionV3. Subir la resolución mejora el detalle pero aumenta el costo cuadráticamente. Si las imágenes del usuario son pequeñas, aumentar artificialmente su tamaño no aporta información. Con recursos limitados conviene usar la resolución nativa mínima.

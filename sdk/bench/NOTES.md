# Metricas pendientes (para mas adelante)

Metricas que decidimos NO implementar todavia, con el porque y el como, para
retomarlas cuando agreguemos las redes mlp/cnn o al redactar la tesis.

## Fidelidad de ranking / confianza (especifica de cada red)

`agreement` (top-1) es binaria: solo dice si el argmax cifrado coincide con el
del modelo en claro, no captura degradacion parcial. Para redes multiclase
(mlp, cnn) conviene medir cuanto preserva la salida cifrada el *orden* y la
*confianza* de las clases:

- correlacion de rango (Spearman) entre los logits cifrados y los del modelo en claro,
- divergencia KL entre la softmax cifrada y la del modelo en claro,
- agreement top-k.

Pendiente: en el playground solo hay 2 clases, asi que `agreement` top-1 basta.
Agregar al introducir mlp/cnn (10 clases), donde dos modelos pueden coincidir en
top-1 pero diferir mucho en el resto del ranking.

## Metricas de evaluacion ML estandar

Ademas de `accuracy`, reportar por red: F1, precision/recall por clase, matriz de
confusion y AUC, sobre las predicciones cifradas vs las etiquetas. Es la
evaluacion clasica de un clasificador.

Pendiente: agregar al reportar resultados por red concreta en el documento de la
tesis (mas informativo con datasets reales/desbalanceados que con el sintetico
del playground).

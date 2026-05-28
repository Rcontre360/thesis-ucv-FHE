# Notas del benchmark

## Orden cronologico de fases por backend (semantica de los deltas)

Cada backend mide tres fases con su propio `Measure`, y reporta para cada una:
`<fase>_vram_mb`, `<fase>_vram_alloc_mb`, `<fase>_ram_mb`, mas
`<fase>_vram_delta_mb` y `<fase>_vram_alloc_delta_mb` = pico de esta fase −
pico de la fase **cronologicamente anterior** en ese backend. La primera fase
ejecutada (`= 0`) no tiene predecesor.

| Backend       | Orden cronologico (1ra → ultima)         | Fase con delta = 0 |
|---------------|-------------------------------------------|--------------------|
| pytorch_plain | infer (no hay keygen ni compile)          | infer              |
| sdk           | keygen → compile → infer                  | keygen             |
| concrete-ml   | compile → keygen → infer                  | compile            |
| orion         | keygen → compile → infer                  | keygen             |

Los nombres de columna coinciden entre backends, pero el delta de `keygen` en
SDK/Orion es `0` (es la primera fase), mientras que en concrete-ml el delta de
`keygen` es `keygen − compile` (porque compile corre primero). Para
interpretar deltas en una vista cruzada, mirar esta tabla.

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

# Regenera las graficas de resultados del Capitulo 4 a partir de los CSV del
# banco de pruebas. Requiere matplotlib y pandas; el entorno del paquete
# (sdk/.env) ya los trae.
#
#   make              regenera las seis graficas
#   make mlp          solo el caso de regresion
#   make cnn          solo el caso de clasificacion

PYTHON ?= ../../sdk/.env/bin/python
GUION   = graficas_resultados.py
BANCO   = ../../sdk/bench
IMAGENES = ../images

CSV = $(BANCO)/mlp/results/results_mlp.csv \
      $(BANCO)/mlp/results/profile_sdk_mlp.csv \
      $(BANCO)/cnn/results/results_cnn.csv \
      $(BANCO)/cnn/results/profile_sdk_cnn.csv

PNG = $(IMAGENES)/res_calidad_mlp.png \
      $(IMAGENES)/res_rendimiento_mlp.png \
      $(IMAGENES)/res_perfil_mlp.png \
      $(IMAGENES)/res_calidad_cnn.png \
      $(IMAGENES)/res_rendimiento_cnn.png \
      $(IMAGENES)/res_perfil_cnn.png

.PHONY: all mlp cnn limpiar

all: $(PNG)

$(PNG): $(GUION) estilo.py formato.py $(CSV)
	$(PYTHON) $(GUION)

mlp:
	$(PYTHON) $(GUION) mlp

cnn:
	$(PYTHON) $(GUION) cnn

limpiar:
	rm -f $(PNG)

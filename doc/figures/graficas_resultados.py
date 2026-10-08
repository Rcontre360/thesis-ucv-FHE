#!/usr/bin/env python3
"""Genera las graficas de resultados del Capitulo 4 a partir de los CSV del banco.

Lee sdk/bench/{mlp,cnn}/results/ y escribe los PNG en doc/images/. Los valores
que aparecen en los rotulos son los mismos que recogen las tablas del capitulo,
con la misma precision y el mismo formato numerico (coma decimal, espacio fino
para los millares).

    python3 graficas_resultados.py           # las seis graficas
    python3 graficas_resultados.py mlp       # solo el caso de regresion
"""

import sys
from math import ceil, log10
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import estilo
from estilo import ACENTO, APAGADO, NEUTRO, TINTA
from formato import cientifica, eje_decimal, megabytes, numero, porcentaje, segundos

RAIZ = Path(__file__).resolve().parents[2]
BANCO = RAIZ / "sdk" / "bench"
SALIDA = RAIZ / "doc" / "images"

# Orden de presentacion y nombre con que aparece cada implementacion en el texto.
IMPLEMENTACIONES = [
    ("pytorch_plain", "PyTorch"),
    ("sdk", "OpalML"),
    ("orion", "Orion"),
    ("concrete-ml", "Concrete ML"),
]
CIFRADAS = IMPLEMENTACIONES[1:]
PROPUESTA = "sdk"

CASOS = {
    "mlp": {
        "titulo_calidad": "Calidad ($R^2$)",
        "eje_calidad": "$R^2$",
        "col_aproximado": "approx_r2",
        "col_cifrado": "r2",
        "decimales": 3,
        "capas": {"Linear": "Lineal", "ReLU": "ReLU"},
    },
    "cnn": {
        "titulo_calidad": "Calidad (exactitud)",
        "eje_calidad": "Exactitud",
        "col_aproximado": "approx_accuracy",
        "col_cifrado": "accuracy",
        "decimales": 2,
        "capas": {"Conv2D": "Convolucional", "ReLU": "ReLU", "Linear": "Lineal"},
    },
}


def cargar(caso):
    resultados = pd.read_csv(BANCO / caso / "results" / f"results_{caso}.csv")
    perfil = pd.read_csv(BANCO / caso / "results" / f"profile_sdk_{caso}.csv")
    return resultados.set_index("backend"), perfil


def memoria(fila):
    """Memoria maxima observada entre la compilacion y la inferencia.

    En la GPU se usa la memoria efectivamente reservada por la biblioteca
    (conjunto de trabajo) cuando el \"backend\" la reporta; si no, el total del
    dispositivo. Es el criterio con que se construyeron las tablas.
    """
    vram = 0.0
    for fase in ("compile", "infer"):
        reservada = fila[f"{fase}_vram_alloc_mb"]
        total = fila[f"{fase}_vram_mb"]
        vram = max(vram, reservada if reservada > 0 else total)
    ram = max(fila["compile_ram_mb"], fila["infer_ram_mb"])
    return vram, ram


def colores(claves):
    return [ACENTO if c == PROPUESTA else NEUTRO for c in claves]


def tope_log(valores, decadas=1):
    return 10 ** (ceil(log10(max(valores))) + decadas)


def guardar(fig, nombre):
    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / f"{nombre}.png"
    fig.savefig(destino, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  {destino.relative_to(RAIZ)}")


# --------------------------------------------------------------------------- #
# Calidad y fidelidad
# --------------------------------------------------------------------------- #

def grafica_calidad(caso, datos):
    cfg = CASOS[caso]
    dec = cfg["decimales"]
    claves = [c for c, _ in CIFRADAS]
    etiquetas = [n for _, n in CIFRADAS]
    referencia = datos.loc["pytorch_plain", cfg["col_aproximado"]]

    aproximado = [datos.loc[c, cfg["col_aproximado"]] for c in claves]
    # Concrete ML no publica una cifra bajo cifrado: repite la del modelo aproximado.
    cifrado = [
        None if c == "concrete-ml" else datos.loc[c, cfg["col_cifrado"]] for c in claves
    ]
    mae = [datos.loc[c, "output_mae"] for c in claves]

    fig, (izq, der) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    x = range(len(claves))
    ancho = 0.36

    b1 = izq.bar([i - ancho / 2 for i in x], aproximado, ancho, color=NEUTRO)
    b2 = izq.bar(
        [i + ancho / 2 for i in x],
        [v if v is not None else 0 for v in cifrado],
        ancho,
        color=ACENTO,
    )
    izq.axhline(referencia, color=TINTA, linestyle="--", linewidth=1.8, zorder=1)
    izq.annotate(
        f"texto plano {numero(referencia, 3)}",   # como en la tabla del capitulo
        xy=(1.0, referencia),
        xycoords=("axes fraction", "data"),
        xytext=(0, 12),
        textcoords="offset points",
        ha="right",
        color=APAGADO,
        fontsize=12,
    )

    estilo.rotular(izq, b1, [numero(v, dec) for v in aproximado])
    estilo.rotular(
        izq, b2, ["n/d" if v is None else numero(v, dec) for v in cifrado]
    )

    izq.set_title(cfg["titulo_calidad"])
    izq.set_ylabel(cfg["eje_calidad"])
    izq.set_ylim(0, max(referencia, max(aproximado)) * 1.45)
    izq.set_xticks(list(x), etiquetas)
    izq.yaxis.set_major_formatter(eje_decimal(1))
    estilo.solo_horizontal(izq)

    b3 = der.bar(list(x), mae, 0.55, color=colores(claves))
    der.set_yscale("log")
    der.set_ylim(1e-4, tope_log(mae))
    der.set_title("Fidelidad frente a PyTorch")
    der.set_ylabel("MAE (escala logarítmica)")
    der.set_xticks(list(x), etiquetas)
    estilo.rotular(der, b3, [cientifica(v) for v in mae])
    estilo.solo_horizontal(der)

    fig.legend(
        handles=[
            Patch(facecolor=NEUTRO, label="Aproximado, sin cifrar"),
            Patch(facecolor=ACENTO, label="Bajo cifrado"),
            Line2D([], [], color=TINTA, linestyle="--", linewidth=1.8,
                   label="Referencia en texto plano"),
        ],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.09),
        labelcolor=APAGADO,
    )
    fig.tight_layout()
    guardar(fig, f"res_calidad_{caso}")


# --------------------------------------------------------------------------- #
# Rendimiento
# --------------------------------------------------------------------------- #

def grafica_rendimiento(caso, datos):
    todas = [c for c, _ in IMPLEMENTACIONES]
    nombres = [n for _, n in IMPLEMENTACIONES]
    claves_cif = [c for c, _ in CIFRADAS]
    nombres_cif = [n for _, n in CIFRADAS]

    latencia = [datos.loc[c, "latency_s"] for c in todas]
    claves_s = [datos.loc[c, "keygen_s"] for c in claves_cif]
    memorias = [memoria(datos.loc[c]) for c in todas]
    vram = [m[0] for m in memorias]
    ram = [m[1] for m in memorias]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(11.0, 4.3))

    x = range(len(todas))
    barras = a.bar(list(x), latencia, 0.6, color=colores(todas))
    a.set_yscale("log")
    a.set_ylim(1e-4, 1e6)
    a.set_title("Inferencia")
    a.set_ylabel("Segundos por muestra")
    a.set_xticks(list(x), nombres, rotation=22, ha="right")
    estilo.rotular(a, barras, [segundos(v) for v in latencia])
    estilo.solo_horizontal(a)

    y = range(len(claves_cif))
    barras = b.bar(list(y), claves_s, 0.6, color=colores(claves_cif))
    b.set_yscale("log")
    b.set_ylim(1e-1, 1e3)
    b.set_title("Generación de claves")
    b.set_ylabel("Segundos")
    b.set_xticks(list(y), nombres_cif, rotation=22, ha="right")
    estilo.rotular(b, barras, [segundos(v) for v in claves_s])
    estilo.solo_horizontal(b)

    ancho = 0.38
    b1 = c.bar([i - ancho / 2 for i in x], vram, ancho, color=NEUTRO, label="VRAM")
    b2 = c.bar([i + ancho / 2 for i in x], ram, ancho, color=ACENTO, label="RAM")
    c.set_yscale("log")
    c.set_ylim(1e-1, 1e7)
    c.set_title("Memoria")
    c.set_ylabel("MB")
    c.set_xticks(list(x), nombres, rotation=22, ha="right")
    estilo.rotular(c, b1, [megabytes(v) for v in vram], fontsize=11)
    estilo.rotular(c, b2, [megabytes(v) for v in ram], fontsize=11)
    c.legend(loc="upper left", ncol=2, labelcolor=APAGADO)
    estilo.solo_horizontal(c)

    fig.tight_layout()
    guardar(fig, f"res_rendimiento_{caso}")


# --------------------------------------------------------------------------- #
# Perfilado por capa
# --------------------------------------------------------------------------- #

def grafica_perfil(caso, perfil):
    nombres = CASOS[caso]["capas"]
    tiempos = perfil["time_s"].tolist()
    total = sum(tiempos)
    etiquetas = [
        f"{i}. {nombres.get(n, n)}"
        for i, n in zip(perfil["layer_idx"], perfil["layer_name"])
    ]

    fig, ax = plt.subplots(figsize=(7.7, 4.3))
    x = range(len(tiempos))
    barras = ax.bar(list(x), tiempos, 0.5, color=ACENTO)
    estilo.rotular(
        ax,
        barras,
        [f"{segundos(t)} s\n{porcentaje(100 * t / total)}" for t in tiempos],
        fontsize=12,
        linespacing=1.3,
    )
    ax.set_ylabel("Segundos")
    ax.set_ylim(0, max(tiempos) * 1.3)
    ax.set_xticks(list(x), etiquetas, rotation=18, ha="right")
    ax.yaxis.set_major_formatter(eje_decimal(0))
    estilo.solo_horizontal(ax)
    fig.tight_layout()
    guardar(fig, f"res_perfil_{caso}")


def main():
    estilo.aplicar()
    casos = sys.argv[1:] or list(CASOS)
    for caso in casos:
        if caso not in CASOS:
            raise SystemExit(f"caso desconocido: {caso} (use {' o '.join(CASOS)})")
        print(f"{caso}:")
        datos, perfil = cargar(caso)
        grafica_calidad(caso, datos)
        grafica_rendimiento(caso, datos)
        grafica_perfil(caso, perfil)


if __name__ == "__main__":
    main()

"""Paleta y estilo comunes a las graficas de resultados.

Monocromo deliberado: el documento se imprime en blanco y negro, de modo que la
distincion entre series descansa en la luminosidad y no en el tono.
"""

import matplotlib as mpl

ACENTO = "#3f4c54"   # serie destacada: la biblioteca propuesta, RAM, barras del perfil
NEUTRO = "#7d8994"   # serie de contraste: implementaciones de referencia, VRAM
REJILLA = "#e3e8ec"
EJES = "#b8c0c6"
TINTA = "#1c2429"
APAGADO = "#4d5a63"

FAMILIA = ["DejaVu Sans"]   # tiene U+2212 y el espacio fino; Noto Sans carece del menos


def aplicar():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FAMILIA,
        "font.size": 13,
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": EJES,
        "axes.labelcolor": APAGADO,
        "axes.titlecolor": TINTA,
        "axes.titlesize": 15,
        "axes.titlepad": 14,
        "axes.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": REJILLA,
        "grid.linewidth": 1.0,
        "xtick.color": APAGADO,
        "ytick.color": APAGADO,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.bottom": False,
        "ytick.left": True,
        "legend.frameon": False,
        "legend.fontsize": 13,
        "savefig.facecolor": "white",
    })


def solo_horizontal(ax):
    """Rejilla unicamente en el eje de valores."""
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")


def rotular(ax, barras, textos, **kwargs):
    """Rotulo sobre cada barra, con fondo blanco para no chocar con la rejilla."""
    opciones = dict(
        padding=4,
        color=APAGADO,
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
    )
    opciones.update(kwargs)
    ax.bar_label(barras, labels=textos, **opciones)

"""Formato de valores numericos segun la convencion del documento.

Coma decimal y espacio fino para los millares, igual que en las tablas del
Capitulo 4 (17 996 MB, 0,478, 6,68 s). Los rotulos de las graficas tienen que
coincidir con el texto, de ahi que todo pase por estas funciones.
"""

ESPACIO_FINO = " "


def numero(valor, decimales=2):
    """Devuelve el valor con coma decimal y espacio fino en los millares."""
    texto = f"{valor:,.{decimales}f}"
    entera, _, decimal = texto.partition(".")
    entera = entera.replace(",", ESPACIO_FINO)
    return f"{entera},{decimal}" if decimales else entera


def segundos(valor):
    """Tiempos: dos decimales, o cuatro cuando el valor es menor que 0,01."""
    return numero(valor, 4 if valor < 0.01 else 2)


def megabytes(valor):
    """Memoria: entero con espacio fino, o un decimal por debajo de 1 MB."""
    return numero(valor, 1 if valor < 1 else 0)


def porcentaje(valor, decimales=1):
    """Porcentaje con espacio fino antes del signo, como recomienda el SI."""
    return f"{numero(valor, decimales)}{ESPACIO_FINO}%"


def cientifica(valor):
    """Notacion cientifica en mathtext, con coma decimal: $6{,}2 \\times 10^{-4}$."""
    if valor == 0:
        return "0"
    exponente = 0
    mantisa = abs(valor)
    while mantisa >= 10:
        mantisa /= 10
        exponente += 1
    while mantisa < 1:
        mantisa *= 10
        exponente -= 1
    mantisa = round(mantisa, 1)
    if mantisa >= 10:          # el redondeo puede desbordar: 9,96 -> 10,0
        mantisa /= 10
        exponente += 1
    signo = "-" if valor < 0 else ""
    return rf"${signo}{mantisa:.1f}".replace(".", "{,}") + rf" \times 10^{{{exponente}}}$"


def eje_decimal(decimales=1):
    """Formateador de ejes para matplotlib, con coma decimal."""
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _pos: numero(v, decimales))

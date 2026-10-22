"""Reducción del modelo P con dos tipos a un Schelling clásico (Proposición 1).

Con dos tipos, para un agente de tipo i con n > 0 vecinos ocupados y fracción
de vecinos distintos f = d/n:

    I = h_i + c_i · f,   con h_i = H[i, i] (diagonal), hb_i = H[i, otro]
                         y c_i = hb_i - h_i (pendiente).

La incomodidad es una función afín de f, así que las decisiones (cómodo,
mejora) solo dependen del signo de c_i y de un umbral reducido θ_i:

- c_i > 0 (homofílico):   cómodo si f <= θ_i, con θ_i = (τ_i - h_i) / c_i.
- c_i < 0 (heterofílico): cómodo si g <= θ_i, con g = 1 - f y
  θ_i = (τ_i - hb_i) / |c_i|.
- c_i = 0 (neutral): la incomodidad es constante.

Con iota = max_j H[i, j] una casilla aislada vale lo mismo que f = 1 (o g = 1),
es decir, ξ_i = 1 para todos los tipos. La reducción es una transformación
afín positiva de cada fila de H, que no cambia la dinámica.
"""

import dataclasses
from fractions import Fraction

from .config import Config

REGIMENES = ("inerte", "activo", "siempre_incomodo", "congelado")


def _umbrales(cfg: Config) -> list:
    return [Fraction(int(x), cfg.escala) for x in cfg.tau_ent]


def _comprobar(cfg: Config) -> None:
    if cfg.modelo != "P":
        raise ValueError("la reducción solo vale para el modelo P")
    if cfg.n != 2:
        raise ValueError("la reducción solo vale para dos tipos")
    H = cfg.H_extendida
    iota = [Fraction(int(x), cfg.escala) for x in cfg.iota_ent]
    if any(iota[i] != max(H[i][:2]) for i in range(2)):
        raise ValueError("la reducción supone iota = max_j H[i, j] (iota='max')")


def parametros_reducidos(cfg: Config) -> list:
    """Parámetros reducidos de cada tipo del modelo P con dos tipos.

    Devuelve una lista con un dict por tipo (en el orden del código: tipo 1 y
    tipo 2, que en el artículo son 1 y -1) con:
    - `s`: orientación, +1 (homofílico), -1 (heterofílico) o 0 (neutral).
    - `theta`: umbral reducido (Fraction), o None si el tipo es neutral.
    - `regimen`: "inerte" (nunca incómodo), "activo", "siempre_incomodo"
      (incómodo en cualquier vecindad) o "congelado" (neutral e incómodo: no
      puede mejorar nunca).
    """
    _comprobar(cfg)
    H = cfg.H_extendida
    tau = _umbrales(cfg)
    resultado = []
    for i in range(2):
        h, hb = H[i][i], H[i][1 - i]
        c = hb - h
        if c == 0:
            resultado.append({"s": 0, "theta": None,
                              "regimen": "inerte" if h <= tau[i] else "congelado"})
            continue
        theta = (tau[i] - h) / c if c > 0 else (tau[i] - hb) / -c
        if theta >= 1:
            regimen = "inerte"
        elif theta >= 0:
            regimen = "activo"
        else:
            regimen = "siempre_incomodo"
        resultado.append({"s": 1 if c > 0 else -1, "theta": theta, "regimen": regimen})
    return resultado


def config_reducida(cfg: Config) -> Config:
    """Configuración del Schelling clásico equivalente (misma trayectoria).

    Cada fila de H se sustituye por su versión normalizada:
    - homofílico: (0, 1) en (propio, otro), umbral θ_i  -> I = f;
    - heterofílico: (1, 0), umbral θ_i                  -> I = g;
    - neutral inerte: (0, 0), umbral 0;
    - neutral congelado: (1, 1), umbral 0.
    En todos los casos iota = "max", que en la versión reducida vale 0 o 1.
    """
    filas, umbrales = [], []
    for i, p in enumerate(parametros_reducidos(cfg)):
        if p["s"] == 0:
            valor = 0 if p["regimen"] == "inerte" else 1
            propio, otro, theta = valor, valor, Fraction(0)
        else:
            propio, otro = (0, 1) if p["s"] == 1 else (1, 0)
            theta = p["theta"]
        fila = [0, 0]
        fila[i], fila[1 - i] = propio, otro
        filas.append(fila)
        umbrales.append(theta)
    return dataclasses.replace(cfg, H=filas, tau=umbrales, iota="max")

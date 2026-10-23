from fractions import Fraction

import numpy as np

from conftest import iguales, trayectoria
from schelling_matricial import Config

H2 = [[0, 1], [1, 0]]
H3 = [[Fraction(-1, 2), 1, Fraction(3, 10)], [2, 0, Fraction(1, 5)], [1, Fraction(7, 10), -1]]
BASE_P = dict(L=12, modelo="P", seed=5)


def transformar(H, lam, mu, columnas_vac=True):
    H = np.array([[Fraction(x) for x in f] for f in H], dtype=object)
    if H.shape[1] == H.shape[0]:
        H = np.hstack([H, np.full((H.shape[0], 1), Fraction(0), dtype=object)])
    Hp = H * lam
    if columnas_vac:
        Hp = Hp + mu
    else:
        Hp[:, :-1] = Hp[:, :-1] + mu
    return Hp.tolist()


def casos_P():
    yield dict(poblaciones=(50, 50), H=H2, tau=Fraction(1, 2), iota=0)
    yield dict(poblaciones=(40, 40, 30), H=H3, tau=Fraction(3, 5), iota=Fraction(1, 4))
    yield dict(poblaciones=(15, 15), H=H2, tau=Fraction(1, 3), iota=Fraction(1, 5))


def test_invariancia_P():
    for caso in casos_P():
        for lam, mu in [(Fraction(5, 2), Fraction(-7, 10)), (Fraction(1, 3), 4)]:
            original = Config(**BASE_P, **caso)
            transformado = Config(
                **BASE_P, poblaciones=caso["poblaciones"],
                H=transformar(caso["H"], lam, mu),
                tau=lam * caso["tau"] + mu, iota=lam * caso["iota"] + mu,
            )
            t1, t2 = trayectoria(original), trayectoria(transformado)
            assert len(t1) > 2
            assert iguales(t1, t2)


def test_no_invariancia_de_iota_en_P():
    # Baja densidad: hay vacíos aislados disponibles. Sin transformar iota, la
    # casilla aislada cambia de valor relativo (observación O3). Con λ=1, μ=-1,
    # iota'=0 equivale a una fracción de distintos f=1 en la escala original.
    lam, mu = Fraction(1), Fraction(-1)
    distintas = 0
    for seed in range(5):
        base = dict(L=12, poblaciones=(15, 15), modelo="P", seed=seed)
        original = Config(**base, H=H2, tau=Fraction(1, 2), iota=0)
        sin_iota = Config(**base, H=transformar(H2, lam, mu), tau=lam * Fraction(1, 2) + mu, iota=0)
        distintas += not iguales(trayectoria(original), trayectoria(sin_iota))
    assert distintas > 0


def casos_S():
    yield dict(poblaciones=(50, 50), H=H2, tau=2)
    yield dict(poblaciones=(40, 40, 30), H=[f + [Fraction(1, 10)] for f in H3], tau=Fraction(3, 2))


def test_invariancia_S():
    k = 8
    for caso in casos_S():
        for lam, mu in [(Fraction(5, 2), Fraction(-7, 10)), (Fraction(1, 3), 4)]:
            original = Config(L=12, modelo="S", seed=5, **caso)
            transformado = Config(
                L=12, modelo="S", seed=5, poblaciones=caso["poblaciones"],
                H=transformar(caso["H"], lam, mu), tau=lam * caso["tau"] + k * mu,
            )
            t1, t2 = trayectoria(original), trayectoria(transformado)
            assert len(t1) > 2
            assert iguales(t1, t2)


def test_sumar_solo_a_ocupados_cambia_S():
    k, lam, mu = 8, Fraction(1), Fraction(1, 2)
    distintas = 0
    for seed in range(5):
        base = dict(L=12, poblaciones=(40, 40), modelo="S", seed=seed)
        original = Config(**base, H=H2, tau=2)
        solo_ocupados = Config(**base, H=transformar(H2, lam, mu, columnas_vac=False),
                               tau=lam * 2 + k * mu)
        distintas += not iguales(trayectoria(original), trayectoria(solo_ocupados))
    assert distintas > 0


def test_iota_max():
    # Solo cuentan las columnas de tipos, no la de vacío.
    cfg = Config(L=12, poblaciones=(10, 10, 10), H=[f + [5] for f in H3], modelo="P")
    assert [Fraction(int(x), cfg.escala) for x in cfg.iota_ent] == [1, 2, 1]


def test_invariancia_P_con_iota_max():
    # Con iota="max" no hace falta transformar iota: max_j(λH + μ) = λ·max_j H + μ.
    for caso in casos_P():
        caso = {**caso, "iota": "max"}
        for lam, mu in [(Fraction(5, 2), Fraction(-7, 10)), (Fraction(1, 3), 4)]:
            original = Config(**BASE_P, **caso)
            transformado = Config(
                **BASE_P, poblaciones=caso["poblaciones"], iota="max",
                H=transformar(caso["H"], lam, mu), tau=lam * caso["tau"] + mu,
            )
            t1, t2 = trayectoria(original), trayectoria(transformado)
            assert len(t1) > 2
            assert iguales(t1, t2)


def test_invariancia_S_fila_a_fila():
    # Cada fila extendida (incluida la columna de vacío) se transforma con su
    # propio (λ_i, μ_i), λ_i > 0, y τ_i -> λ_i τ_i + k μ_i: como n + e = k,
    # S -> λ_i S + k μ_i y la dinámica no cambia.
    k = 8
    H = [[Fraction(1, 5), Fraction(9, 10), Fraction(3, 10)],
         [Fraction(2, 5), Fraction(-1, 10), Fraction(-7, 10)]]
    tau = [Fraction(1, 2), Fraction(-1, 5)]
    lam, mu = [Fraction(3), Fraction(1, 4)], [Fraction(-1, 2), Fraction(2)]
    Ht = [[lam[i] * x + mu[i] for x in H[i]] for i in range(2)]
    taut = [lam[i] * tau[i] + k * mu[i] for i in range(2)]
    for seed in range(3):
        base = dict(L=20, poblaciones=(90, 90), modelo="S", seed=seed)
        t1 = trayectoria(Config(**base, H=H, tau=tau))
        assert len(t1) > 2
        assert iguales(t1, trayectoria(Config(**base, H=Ht, tau=taut)))


def test_columna_de_vacio_se_puede_anular_en_S():
    # Caso particular: μ_i = -H[i, vac], λ_i = 1 lleva toda fila a H[i, vac] = 0
    # con umbral τ_i - k H[i, vac].
    k = 8
    H = [[Fraction(1, 5), Fraction(9, 10), Fraction(3, 10)],
         [Fraction(2, 5), Fraction(-1, 10), Fraction(-7, 10)]]
    tau = Fraction(1, 2)
    H0 = [[x - f[2] for x in f] for f in H]
    tau0 = [tau - k * f[2] for f in H]
    for seed in range(3):
        base = dict(L=20, poblaciones=(90, 90), modelo="S", seed=seed)
        assert iguales(trayectoria(Config(**base, H=H, tau=tau)),
                       trayectoria(Config(**base, H=H0, tau=tau0)))

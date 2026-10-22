"""Proposición 1: el modelo P con dos tipos se reduce a un Schelling clásico."""

from fractions import Fraction

import pytest

from conftest import iguales, trayectoria
from schelling_matricial import Config, config_reducida, parametros_reducidos

TAU = Fraction(513, 1000)

# Una matriz por combinación de orientaciones y regímenes (filas: tipo 1, tipo 2).
MATRICES = {
    "homo_homo": [[Fraction(2, 5), Fraction(3, 5)], [Fraction(3, 4), Fraction(3, 10)]],
    "homo_hetero": [[Fraction(2, 5), Fraction(3, 5)], [Fraction(3, 10), Fraction(3, 4)]],
    "hetero_hetero": [[Fraction(3, 5), Fraction(1, 5)], [Fraction(1, 10), Fraction(7, 10)]],
    "homo_neutral_inerte": [[0, 1], [Fraction(2, 5), Fraction(2, 5)]],
    "homo_neutral_congelado": [[0, 1], [Fraction(3, 5), Fraction(3, 5)]],
    "siempre_incomodo": [[Fraction(3, 5), Fraction(4, 5)], [Fraction(4, 5), Fraction(3, 5)]],
}


@pytest.mark.parametrize("nombre", MATRICES)
@pytest.mark.parametrize("poblaciones", [(150, 150), (60, 60)])
def test_misma_trayectoria_que_el_clasico_reducido(nombre, poblaciones):
    # (60, 60) en L=20: baja densidad, con agentes y casillas aisladas.
    for seed in range(3):
        cfg = Config(L=20, poblaciones=poblaciones, H=MATRICES[nombre], tau=TAU, seed=seed)
        t1, t2 = trayectoria(cfg, 60), trayectoria(config_reducida(cfg), 60)
        assert iguales(t1, t2)


def test_parametros_del_ejemplo():
    cfg = Config(L=20, poblaciones=(10, 10), H=MATRICES["homo_homo"], tau=TAU)
    p = parametros_reducidos(cfg)
    assert [x["s"] for x in p] == [1, 1]
    assert [x["theta"] for x in p] == [Fraction(113, 200), Fraction(71, 150)]
    assert [x["regimen"] for x in p] == ["activo", "activo"]


def test_regimenes():
    def regimenes(H, tau):
        cfg = Config(L=20, poblaciones=(10, 10), H=H, tau=tau)
        return [x["regimen"] for x in parametros_reducidos(cfg)]

    # Observación A5: homofílica en la tabla, pero nunca cómoda (θ = -1/2).
    assert regimenes(MATRICES["siempre_incomodo"], Fraction(1, 2)) == ["siempre_incomodo"] * 2
    assert regimenes([[0, 1], [1, 0]], 1) == ["inerte", "inerte"]
    assert regimenes([[0, 1], [Fraction(2, 5), Fraction(2, 5)]], TAU) == ["activo", "inerte"]
    assert regimenes([[0, 1], [Fraction(3, 5), Fraction(3, 5)]], TAU) == ["activo", "congelado"]


def test_invariancia_fila_a_fila():
    # Cada agente solo usa su fila de H, su umbral y su iota: se puede
    # transformar cada fila con su propio (λ_i, μ_i), λ_i > 0.
    H = MATRICES["homo_homo"]
    lam, mu = [Fraction(5), Fraction(1, 3)], [Fraction(-2), Fraction(7, 10)]
    Ht = [[lam[i] * x + mu[i] for x in H[i]] for i in range(2)]
    taut = [lam[i] * TAU + mu[i] for i in range(2)]
    for seed in range(3):
        base = dict(L=20, poblaciones=(60, 60), seed=seed)
        assert iguales(trayectoria(Config(**base, H=H, tau=TAU)),
                       trayectoria(Config(**base, H=Ht, tau=taut)))


def test_errores():
    with pytest.raises(ValueError):
        parametros_reducidos(Config(L=20, poblaciones=(10, 10), H=[[0, 1], [1, 0]], modelo="S"))
    with pytest.raises(ValueError):
        parametros_reducidos(Config(L=20, poblaciones=(10, 10, 10), H=[[0, 1, 1]] * 3))
    with pytest.raises(ValueError):
        parametros_reducidos(Config(L=20, poblaciones=(10, 10), H=[[0, 1], [1, 0]], iota=0))

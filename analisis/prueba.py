from schelling_matricial import Config, Simulacion

cfg = Config(L=40, poblaciones=(700, 700), H=[[0, 1], [1, 0]], tau=0.7, seed=123)
r = Simulacion(cfg).ejecutar()
print(r["motivo_parada"], r["tipo_punto_fijo"], r["barridos"], r["s"])

# -*- coding: utf-8 -*-
"""
IRIS F4b - Ciclo de ingesta de ~2 h (SOLO datos: lectura + filtro + buffer).
El tokenizador y el bucle de entrenamiento los conecta Desarrollador de ia.

Diseño del ciclo:
- Ventana por dataset proporcional a su peso del plan (reparto de lectura).
- Reanudable: cursores por dataset (JSON atomico) + chunks deterministas.
- Tope de buffer: si un dataset alcanza TOPE_GB se omite hasta que el
  consumidor (tokenizador) libere espacio borrando chunks consumidos.
- Al terminar el ciclo escribe estado_ciclo.json con el resumen.

Uso:  python run_cycle.py [segundos]
      python run_cycle.py            # ciclo completo (7200 s)
"""
import sys, time, json
from pathlib import Path
import iris_stream as st

RAIZ = Path(__file__).resolve().parent
ESTADO = RAIZ / "estado_ciclo.json"

# pesos del plan (IRIS_TRAINING_PLAN.md): wikipedia ES 15 / EN 15 / fineweb 25 / c4 30 / oasst2 15
PESOS = {
    "wikipedia_es": 15, "wikipedia_en": 15,
    "fineweb_edu": 25, "c4_es": 30, "oasst2": 15,
}
CICLO_S = 2 * 3600

def correr_ciclo(segundos=CICLO_S):
    fin = time.time() + segundos
    restante = segundos
    resultados = {}
    # ventana proporcional al peso; rotando hasta agotar el ciclo
    total = sum(PESOS.values())
    for nombre, peso in PESOS.items():
        if restante <= 0:
            break
        ventana = restante * peso / total
        t0 = time.time()
        print(f"== ingesta {nombre}: ventana {ventana/60:.0f} min")
        resultados[nombre] = st.correr_ingesta([nombre], ventana)
        restante = max(0.0, fin - time.time())
    estado = {
        "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duracion_s": round(segundos),
        "stats": resultados,
        "buffer_mb": {n: round(st.bytes_dataset(n) / 1e6, 1) for n in PESOS},
    }
    tmp = ESTADO.with_suffix(".tmp")
    tmp.write_text(json.dumps(estado, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(ESTADO)
    print("estado del ciclo:", ESTADO)
    return estado

if __name__ == "__main__":
    seg = int(sys.argv[1]) if len(sys.argv) > 1 else CICLO_S
    e = correr_ciclo(seg)
    print(json.dumps(e["buffer_mb"], indent=1))

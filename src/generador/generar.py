"""Punto de entrada del generador: crea las corridas de una versión del dataset.

Uso:
    python -m src.generador.generar --version v0.2                 # 3 escenarios × 3 réplicas × 52 semanas
    python -m src.generador.generar --version v0.2 --escenario base --semilla 2026 --semanas 8

Cada corrida (escenario y semilla) deja en datos/sinteticos/<versión>/ tres archivos:
    registros_<escenario>_<semilla>.csv.gz   los registros del dataset (51 campos; diccionario de datos)
    eventos_<escenario>_<semilla>.csv.gz     ventanas y fallas de cada equipo (diccionario, Tabla 17)
    resumen_<escenario>_<semilla>.json       configuración, conteos y lo que quedó pendiente al cierre
La v0.1 va a datos/sinteticos/borradores/v0.1/, y las corridas de prueba con otro número de semanas, a
datos/sinteticos/borradores/<versión>_<semanas>_semanas/, para no reemplazar las del dataset; esa carpeta
no se sube al repositorio.

Diseño: docs/Generador_de_datos.docx, secciones 3 a 8.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.generador import escenarios, llegadas, perturbaciones, registro
from src.generador.recursos import MIN_POR_DIA, Laboratorio
from src.simulacion.entorno import Simulador
from src.simulacion.politicas import ReglaGenerador

RAIZ = Path(__file__).resolve().parents[2]
COLUMNAS_EVENTOS = ["escenario", "semilla", "equipo", "configuracion", "evento", "inicio", "fin"]


def carpeta(version: str, semanas: int = escenarios.SEMANAS) -> Path:
    """datos/sinteticos/<versión>. La versión de prueba (v0.1) y las corridas de otro número de semanas van a la
    subcarpeta borradores, que no se sube al repositorio."""
    base = RAIZ / "datos" / "sinteticos"
    if int(semanas) != escenarios.SEMANAS:
        return base / "borradores" / f"{version}_{int(semanas)}_semanas"
    return base / "borradores" / version if version == "v0.1" else base / version


def generar_corrida(escenario: str, semilla: int, version: str = "v0.2", semanas: int = escenarios.SEMANAS,
                    politica=None, verificar: bool = True, laboratorio: Laboratorio | None = None):
    """Una corrida completa: muestras, ensayos, perturbaciones y ejecución en el laboratorio.

    Devuelve (registros, eventos de los equipos, resumen). Los registros cumplen las 17 reglas del
    diccionario (registro.chequear_registro); si verificar es True, la función falla si alguno no las cumple.
    laboratorio: otro modelo de recursos para un análisis de sensibilidad (por omisión, el de las hojas del Excel);
    las reglas de asignación del diccionario siguen leyendo las hojas, así que con otro hay que pasar verificar=False.
    """
    config = escenarios.VERSIONES[version]
    carga = escenarios.ESCENARIOS[escenario]["multiplicador_carga"]
    dias = int(semanas) * 7
    rng = escenarios.generadores(semilla)
    inicio = time.time()
    lab = laboratorio or Laboratorio.desde_excel()
    muestras = llegadas.generar_muestras(dias, carga, rng["llegadas"], rng["urgentes"])
    ensayos = llegadas.generar_ensayos(muestras, rng["ensayos"])
    ventanas = lab.ventanas(dias)
    fallas = perturbaciones.generar_fallas(lab.equipos.values(), dias * MIN_POR_DIA, rng["fallas"])
    sim = Simulador(lab, politica or ReglaGenerador(), dias, rng, forma=config["forma"], rho=config["rho"],
                    alistamiento=config["alistamiento"], escenario=escenario, carga=carga, semilla=semilla)
    filas, eventos, resumen = sim.correr(muestras, ensayos, ventanas, fallas)
    registros = pd.DataFrame(filas, columns=registro.CAMPOS_REGISTRO)
    registros = registros.sort_values(["inicio", "id_actividad"], ignore_index=True)
    eventos = pd.DataFrame(eventos, columns=COLUMNAS_EVENTOS[2:])
    eventos.insert(0, "semilla", semilla)
    eventos.insert(0, "escenario", escenario)
    eventos = eventos.sort_values(["inicio", "equipo"], ignore_index=True)
    reglas = registro.chequear_registro(registros) if len(registros) else pd.DataFrame()
    cumplen = int(reglas.all(axis=1).sum()) if len(reglas) else 0
    resumen.update({
        "version": version, "escenario": escenario, "carga": carga, "semilla": int(semilla), "semanas": int(semanas),
        "configuracion": {k: v for k, v in config.items() if k != "descripcion"},
        "regla_de_despacho": (politica or ReglaGenerador()).nombre,
        "muestras": int(len(muestras)), "urgentes": int(muestras["urgente"].sum()), "ensayos_pedidos": int(len(ensayos)),
        "registros": int(len(registros)), "registros_que_cumplen_las_reglas": cumplen,
        "reanalisis": int(registros["causa_repeticion"].eq("reanálisis").sum()),
        "interrumpidos": int(registros["resultado"].eq("interrumpido").sum()),
        "fallas": int(eventos["evento"].eq("falla").sum()),
        "ventanas": int(eventos["evento"].isin(["mantenimiento", "verificación"]).sum()),
        "segundos": round(time.time() - inicio, 1),
    })
    if verificar and cumplen != len(registros):
        malas = reglas.columns[~reglas.all()].tolist()
        raise AssertionError(f"{len(registros) - cumplen} registros no cumplen las reglas {malas}")
    return registros, eventos, resumen


def guardar_corrida(registros, eventos, resumen, destino: Path):
    destino.mkdir(parents=True, exist_ok=True)
    nombre = f"{resumen['escenario']}_{resumen['semilla']}"
    registros.to_csv(destino / f"registros_{nombre}.csv.gz", index=False, float_format="%.6f")
    eventos.to_csv(destino / f"eventos_{nombre}.csv.gz", index=False, float_format="%.6f")
    (destino / f"resumen_{nombre}.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")


def cargar_version(version: str, escenario=None) -> tuple:
    """Lee los registros y los eventos de una versión (todas las corridas, o las de un escenario)."""
    destino = carpeta(version)
    patron = f"_{escenario}_" if escenario else "_"
    registros = [pd.read_csv(p) for p in sorted(destino.glob("registros*.csv.gz")) if patron in p.name]
    eventos = [pd.read_csv(p) for p in sorted(destino.glob("eventos*.csv.gz")) if patron in p.name]
    resumenes = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(destino.glob("resumen*.json"))
                 if patron in p.name]
    return pd.concat(registros, ignore_index=True), pd.concat(eventos, ignore_index=True), resumenes


def generar_version(version: str, replicas: int = escenarios.REPLICAS, semanas: int = escenarios.SEMANAS,
                    escenarios_elegidos=None, semillas=None):
    destino = carpeta(version, semanas)
    print(f"Las corridas van a {destino}")
    for escenario in escenarios_elegidos or list(escenarios.ESCENARIOS):
        for semilla in semillas or escenarios.semillas(replicas):
            registros, eventos, resumen = generar_corrida(escenario, semilla, version, semanas)
            guardar_corrida(registros, eventos, resumen, destino)
            print(f"{version} {escenario} {semilla}: {resumen['registros']} registros en {resumen['segundos']} s")


def main():
    parser = argparse.ArgumentParser(description="Generador de datos sintéticos (O1)")
    parser.add_argument("--version", choices=sorted(escenarios.VERSIONES), default="v0.2")
    parser.add_argument("--escenario", choices=sorted(escenarios.ESCENARIOS))
    parser.add_argument("--semilla", type=int)
    parser.add_argument("--semanas", type=int, default=escenarios.SEMANAS)
    parser.add_argument("--replicas", type=int, default=escenarios.REPLICAS)
    args = parser.parse_args()
    generar_version(args.version, args.replicas, args.semanas,
                    [args.escenario] if args.escenario else None, [args.semilla] if args.semilla is not None else None)


if __name__ == "__main__":
    main()

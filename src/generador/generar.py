"""Punto de entrada del generador: crea un dataset para un escenario y una semilla.

Uso (cuando esté listo, desde el 30 de octubre):
    python -m src.generador.generar --escenario base --semilla 2026
"""
import argparse

from src.generador.escenarios import ESCENARIOS, SEMILLA_BASE


def main():
    parser = argparse.ArgumentParser(description="Generador de datos sintéticos (O1)")
    parser.add_argument("--escenario", choices=sorted(ESCENARIOS), default="base")
    parser.add_argument("--semilla", type=int, default=SEMILLA_BASE)
    args = parser.parse_args()
    raise NotImplementedError(f"Pendiente: generar dataset {args.escenario} con semilla {args.semilla}")


if __name__ == "__main__":
    main()

"""Descarga las instancias de referencia para el contraste estructural (bloque del 1 de noviembre).

- TLSP (Mischek y Musliu, 2021): conjunto de evaluación con 15 instancias General y
  15 LabStructure (archivos XML). https://dbai.tuwien.ac.at/staff/fmischek/TLSP/
- Martins et al. (2023): 54 instancias JSON de su repositorio público (licencia MIT).

No se suben a git (ver .gitignore): cada quien las baja en su máquina.
Uso, desde la raíz del repositorio:
    python datos/descargar_instancias.py
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

CARPETA_DATOS = Path(__file__).resolve().parent

TLSP_URL = "https://dbai.tuwien.ac.at/staff/fmischek/TLSP/instances/evaluation-set-TLSPS.zip"
MARTINS_URL = ("https://github.com/miguelsem/Minimizing-total-completion-time-in-large-sized-"
               "pharmaceutical-quality-control-scheduling/archive/refs/heads/main.zip")


def bajar(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=120) as respuesta:
            return respuesta.read()
    except Exception as error:  # noqa: BLE001
        sys.exit(f"No se pudo descargar {url}\n{error}\n"
                 "En Mac, si el error es de certificados, instala certifi (pip install certifi) "
                 "o córrelo en Cloud Shell.")


def instancias_tlsp(destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    if any(destino.glob("*/*_tasks.xml")):
        print(f"TLSP: ya estaban en {destino}")
        return
    archivo = zipfile.ZipFile(io.BytesIO(bajar(TLSP_URL)))
    n = 0
    for nombre in archivo.namelist():
        ruta = Path(nombre)
        if ruta.suffix == ".xml" and len(ruta.parts) >= 2:
            salida = destino / ruta.parts[-2] / ruta.name  # conserva General/ y LabStructure/
            salida.parent.mkdir(parents=True, exist_ok=True)
            salida.write_bytes(archivo.read(nombre))
            n += 1
    print(f"TLSP: {n} archivos en {destino}")


def instancias_martins(destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    if any(destino.glob("*.json")):
        print(f"Martins: ya estaban en {destino}")
        return
    archivo = zipfile.ZipFile(io.BytesIO(bajar(MARTINS_URL)))
    n = 0
    for nombre in archivo.namelist():
        ruta = Path(nombre)
        if len(ruta.parts) == 3 and ruta.parts[1] == "instances" and ruta.suffix == ".json":
            (destino / ruta.name).write_bytes(archivo.read(nombre))
            n += 1
        elif len(ruta.parts) == 2 and ruta.name == "LICENSE":
            (destino / "LICENSE_martins.txt").write_bytes(archivo.read(nombre))
    print(f"Martins: {n} instancias en {destino}")


if __name__ == "__main__":
    instancias_tlsp(CARPETA_DATOS / "instancias_tlsp")
    instancias_martins(CARPETA_DATOS / "instancias_martins")

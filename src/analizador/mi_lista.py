"""«Mi lista»: los activos que sigue el usuario, guardados en un archivo local.

Es una app para una persona en su ordenador, así que basta un JSON. (En Streamlit Community
Cloud el archivo se borra al reiniciar la app.)
"""

from __future__ import annotations

import json
from pathlib import Path

RUTA = Path(__file__).resolve().parents[2] / "modelos" / "mi_lista.json"


def en_la_nube() -> bool:
    """True si la app corre en Streamlit Community Cloud (allí el código vive en /mount/src)."""
    return str(Path(__file__).resolve()).startswith("/mount/src")


def cargar(ruta: Path | str = RUTA) -> list[str]:
    ruta = Path(ruta)
    if not ruta.exists():
        return []
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [str(t) for t in datos.get("tickers", [])]


def guardar(tickers: list[str], ruta: Path | str = RUTA) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    unicos = list(dict.fromkeys(tickers))  # sin duplicados, en el orden en que se añadieron
    ruta.write_text(json.dumps({"tickers": unicos}, ensure_ascii=False, indent=2), encoding="utf-8")


def anadir(ticker: str, ruta: Path | str = RUTA) -> list[str]:
    lista = cargar(ruta)
    if ticker not in lista:
        lista.append(ticker)
        guardar(lista, ruta)
    return lista


def quitar(ticker: str, ruta: Path | str = RUTA) -> list[str]:
    lista = [t for t in cargar(ruta) if t != ticker]
    guardar(lista, ruta)
    return lista

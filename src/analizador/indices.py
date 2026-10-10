"""Principales índices bursátiles por país (tickers de Yahoo Finance).

Son la cesta con la que se entrena la IA global y la lista del estudio de índices.
Se excluyen índices sin historial en Yahoo (Chile, Rusia, Tailandia, Filipinas, Egipto).
"""

from __future__ import annotations

# ticker: (país, nombre del índice, región)
INDICES: dict[str, tuple[str, str, str]] = {
    "^GSPC": ("Estados Unidos", "S&P 500", "América"),
    "^DJI": ("Estados Unidos", "Dow Jones", "América"),
    "^IXIC": ("Estados Unidos", "Nasdaq Composite", "América"),
    "^RUT": ("Estados Unidos", "Russell 2000", "América"),
    "^GSPTSE": ("Canadá", "S&P/TSX", "América"),
    "^MXX": ("México", "IPC", "América"),
    "^BVSP": ("Brasil", "Bovespa", "América"),
    "^MERV": ("Argentina", "Merval", "América"),
    "^FTSE": ("Reino Unido", "FTSE 100", "Europa"),
    "^GDAXI": ("Alemania", "DAX", "Europa"),
    "^FCHI": ("Francia", "CAC 40", "Europa"),
    "^IBEX": ("España", "IBEX 35", "Europa"),
    "FTSEMIB.MI": ("Italia", "FTSE MIB", "Europa"),
    "^AEX": ("Países Bajos", "AEX", "Europa"),
    "^SSMI": ("Suiza", "SMI", "Europa"),
    "^BFX": ("Bélgica", "BEL 20", "Europa"),
    "^ATX": ("Austria", "ATX", "Europa"),
    "^OMX": ("Suecia", "OMX Stockholm 30", "Europa"),
    "OSEBX.OL": ("Noruega", "OSEBX", "Europa"),
    "^OMXC25": ("Dinamarca", "OMX Copenhagen 25", "Europa"),
    "PSI20.LS": ("Portugal", "PSI", "Europa"),
    "^STOXX50E": ("Eurozona", "Euro Stoxx 50", "Europa"),
    "XU100.IS": ("Turquía", "BIST 100", "Europa"),
    "^N225": ("Japón", "Nikkei 225", "Asia-Pacífico"),
    "^HSI": ("Hong Kong", "Hang Seng", "Asia-Pacífico"),
    "000001.SS": ("China", "SSE Composite (Shanghái)", "Asia-Pacífico"),
    "^KS11": ("Corea del Sur", "KOSPI", "Asia-Pacífico"),
    "^TWII": ("Taiwán", "TAIEX", "Asia-Pacífico"),
    "^BSESN": ("India", "Sensex", "Asia-Pacífico"),
    "^NSEI": ("India", "Nifty 50", "Asia-Pacífico"),
    "^AXJO": ("Australia", "S&P/ASX 200", "Asia-Pacífico"),
    "^NZ50": ("Nueva Zelanda", "NZX 50", "Asia-Pacífico"),
    "^STI": ("Singapur", "Straits Times", "Asia-Pacífico"),
    "^JKSE": ("Indonesia", "Jakarta Composite", "Asia-Pacífico"),
    "^KLSE": ("Malasia", "FTSE Bursa Malaysia KLCI", "Asia-Pacífico"),
    "^TA125.TA": ("Israel", "TA-125", "Oriente Medio y África"),
    "^J203.JO": ("Sudáfrica", "FTSE/JSE All Share", "Oriente Medio y África"),
}

# Índice que se usa como "mercado global" en las características de la IA.
MERCADO = "^GSPC"


def etiqueta(ticker: str) -> str:
    """'IBEX 35 (España)' para un índice conocido; el propio ticker en otro caso."""
    if ticker in INDICES:
        pais, nombre, _ = INDICES[ticker]
        return f"{nombre} ({pais})"
    return ticker

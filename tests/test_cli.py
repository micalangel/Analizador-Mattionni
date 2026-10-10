import pytest

from analizador import AVISO_LEGAL
from analizador.cli import crear_parser


def test_ayuda_incluye_aviso_legal(capsys):
    with pytest.raises(SystemExit):
        crear_parser().parse_args(["--help"])
    salida = " ".join(capsys.readouterr().out.split())
    assert AVISO_LEGAL in salida


def test_argumentos():
    args = crear_parser().parse_args(["AAPL", "--horizon", "15"])
    assert args.tickers == ["AAPL"] and args.horizonte == 15


def test_mismo_aviso_en_readme_y_metodologia():
    from pathlib import Path

    raiz = Path(__file__).parent.parent
    for doc in ["README.md", "docs/METHODOLOGY.md", "app.py"]:
        texto = (raiz / doc).read_text(encoding="utf-8")
        assert AVISO_LEGAL in texto or "AVISO_LEGAL" in texto, doc

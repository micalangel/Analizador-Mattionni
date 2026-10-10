#!/bin/bash
# Actualización automática de Mattionni Analytics (la programa launchd cada noche en el Mac).
#
#  * Cada noche: estudio de todas las empresas con los precios del día (probabilidades, rangos,
#    salud y volumen) y apunta las predicciones nuevas en el diario de la IA.
#  * Los domingos: además, reentrena la IA con todos los datos hasta hoy.
#  * Sube a GitHub solo los modelos y estudios; la web se actualiza sola al recibirlos.
#
# Uso manual:  scripts/actualizar_noche.sh            (lo mismo que de noche)
#              scripts/actualizar_noche.sh --entrenar (fuerza el reentrenamiento)
#              scripts/actualizar_noche.sh --prueba   (solo comprueba que todo es accesible)

set -uo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RAIZ" || exit 1
PY="$RAIZ/venv/bin/python"
export PYTHONPATH="$RAIZ/src" PYTHONWARNINGS=ignore

echo "===== $(date '+%Y-%m-%d %H:%M') · Mattionni Analytics ====="

if [[ "${1:-}" == "--prueba" ]]; then
  "$PY" --version && git status --short | head -5 && git ls-remote --heads origin main | head -1
  echo "Prueba correcta."
  exit 0
fi

# Traer primero los cambios de los demás (amigos que editan la web).
git pull --rebase --autostash -q origin main || echo "Aviso: no se pudo traer lo último de GitHub."

if [[ "$(date +%u)" == "7" || "${1:-}" == "--entrenar" ]]; then
  echo "-- Reentrenando la IA…"
  "$PY" -m analizador.entrenamiento || { echo "ERROR en el entrenamiento"; exit 1; }
fi

echo "-- Estudio de empresas…"
"$PY" -m analizador.estudio || { echo "ERROR en el estudio"; exit 1; }

# Solo los resultados: nunca el código a medio hacer de quien esté trabajando en el Mac.
git add modelos/global/*.joblib modelos/global/estudio_*.csv modelos/registro/ 2>/dev/null
if git diff --cached --quiet; then
  echo "Sin cambios que subir."
else
  git commit -q -m "Actualización automática del $(date '+%d/%m/%Y')" &&
    git pull --rebase --autostash -q origin main &&
    git push -q origin main && echo "Subido a GitHub: la web se actualizará en unos minutos." ||
    echo "ERROR al subir a GitHub"
fi
echo "Listo a las $(date '+%H:%M')."

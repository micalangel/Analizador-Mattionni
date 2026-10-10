# Analizador-Mattionni

Aplicación web para analizar acciones y ETF con fines **educativos y de investigación**:
indicadores técnicos, ratios fundamentales, backtesting walk-forward con costes,
rangos de precio con incertidumbre declarada y un informe de calibración que comprueba
si esos rangos aciertan tanto como dicen.

> **Aviso:** Esta herramienta tiene fines educativos y de investigación. No constituye asesoramiento financiero. El rendimiento pasado no predice resultados futuros.

## La web en cuatro páginas

| Página | Para qué sirve |
|---|---|
| 🎯 **¿Subirá o bajará?** (portada) | Buscas por nombre (Amazon, Santander, oro, euro, IBEX…) y eliges plazo (5 días, 10 días, 1 mes, 6 meses o 1 año). Ves la probabilidad de que esté más arriba o más abajo, el rango hasta donde podría llegar y cuánto han acertado ambas cosas en el pasado. |
| 🏢 **Empresas** | Unas 270 empresas (EE. UU., España, Europa, Asia, Latinoamérica), divisas, materias primas y criptomonedas, con filtros por país y sector: probabilidad de estar más arriba, rango y si están en caída fuerte. |
| 🤖 **La IA** | Cómo aprende, su **diario de predicciones** (apuntadas antes de saber el resultado y comprobadas al vencer) y el historial de entrenamientos. |
| 🌍 **Índices del mundo** | Los principales índices de 30 países, con su rango a cada plazo y su fiabilidad comprobada. |
| 🧪 **Laboratorio de señales** | Prueba si alguna señal clásica (tendencia, momentum, RSI, rebotes, calendario…) adivina la dirección mejor que el azar, con datos que no se usaron para descubrirla. |
| 🔬 **Análisis avanzado** | Indicadores, fundamentales, resultados y noticias, backtest, calibración detallada e IA por ticker. |

## Qué hace

- **Datos:** descarga precios diarios OHLCV de Yahoo Finance con `yfinance` (`auto_adjust=True`,
  ajustados por dividendos y splits) para uno o varios tickers. Guarda una caché local en CSV
  (`data_cache/`) y trata los datos faltantes (filas sin cierre, fechas duplicadas, zonas horarias).
- **Indicadores:** medias móviles (simple y exponencial), RSI, MACD, bandas de Bollinger,
  volatilidad móvil y drawdown. Cada uno es una función independiente con tests.
- **Fundamentales (opcional):** PER, precio/valor contable, deuda/patrimonio, margen de
  beneficio y rentabilidad del flujo de caja libre, lado a lado.
- **Estudio de resultados financieros:** ingresos, beneficios, BPA, flujo de caja y deuda
  (trimestral y anual), márgenes y crecimiento interanual; historial de BPA estimado frente a
  publicado (sorpresa) y reacción del precio en cada publicación, medida también en desviaciones
  típicas respecto a la volatilidad previa.
- **Noticias:** titulares recientes (fuente, fecha y enlace) y el movimiento del precio en su sesión.
  No se valoran ni resumen las noticias, y coincidir en el tiempo no implica causa.
- **Rangos de pronóstico:** intervalo de precios a *h* días con un nivel de confianza declarado
  (por ejemplo 80 %), a partir de la volatilidad histórica (desviación típica móvil, EWMA o
  GARCH(1,1)). **Siempre un rango, nunca un precio único.**
- **¿Más arriba o más abajo?:** probabilidad de que el precio esté más arriba al final del plazo. Se usa
  la frecuencia histórica del propio activo o la IA de dirección (clasificador entrenado con los
  índices mundiales), la que mejor haya funcionado en sus datos recientes. Fuera de muestra, en los 37
  índices, la IA acierta el lado el 55,5 % (5 días), 58,8 % (1 mes), 65,2 % (6 meses) y 67,5 % (1 año),
  frente a 55,3 %, 58,2 %, 62,7 % y 65,4 % de la frecuencia histórica: aporta poco a corto plazo.
- **Analizar una empresa:** toda su información en pestañas: qué hace, precio objetivo de los analistas y **si han acertado antes con esa empresa** (con una guía de cómo usar esa información), valoración (PER actual frente a **su
  propia media de 10 años** y a su sector, PER esperado, PEG, precio/ventas, EV/EBITDA), rentabilidad y deuda
  (márgenes, ROE, ROA, caja neta, liquidez), crecimiento, dividendos (rentabilidad real del último año, años
  sin bajar), accionariado, opinión de los analistas (como dato de terceros), análisis técnico objetivo (tendencia,
  impulso, volatilidad, soportes y resistencias, rendimiento frente a su índice, velas con medias, volumen y RSI)
  y resultados y noticias.
- **Mi lista, alertas, calendario y comparar:** sigue tus activos favoritos con alertas diarias (caída fuerte,
  volumen inusual, pánico, resultados cercanos, movimientos muy grandes), consulta cuándo presentan resultados y
  cuánto suelen moverse ese día, y compara de 2 a 4 activos lado a lado. `python -m analizador.alertas --notificar`
  muestra las alertas como aviso del sistema en macOS.
- **Simulador de estrategias (educativo):** reglas clásicas (cruce de medias, tendencia, ruptura de máximos, RSI,
  comprar caídas) con sus puntos de compra ▲ y venta ▼, operaciones, costes, comparación con comprar y mantener y
  con el azar, en velas diarias, de 1 hora o de 5 minutos. Todo simulado: no envía órdenes a ningún bróker.
- **Ficha de cada empresa:** salud financiera con semáforos explicados (PER, margen, deuda, flujo de caja,
  crecimiento de ventas y beneficio del último año), volumen de hoy frente a lo normal con los días de volumen
  inusual recientes y lo que pasó después, y noticias recientes.
- **Experimento de volumen:** en ~360 empresas desde 1990, ninguna regla de volumen inusual (en días de subida, de
  bajada, acumulación silenciosa o volumen muy bajo) anticipa subidas de forma fiable; la que más se acerca es el
  volumen alto en un día de caída (+2,6 puntos a 3 meses, no concluyente).
- **Situaciones especiales:** detecta si hoy hay **pánico** (VIX ≥ 30) o una **caída fuerte** (−20 % desde
  máximos) y muestra qué pasó en cada episodio anterior, contado por episodios y no por días. Desde 1990
  ha habido 21 episodios de pánico cerrados: un año después el S&P 500 estaba más arriba en 15 (6 de 11
  antes de 2010 y 9 de 10 desde 2010).
- **IA de dirección con miedo y tipos de interés:** variante que añade el VIX y la curva de tipos de
  EE. UU. Mejora a 1 año (67,8 % de acierto) y empeora a plazos cortos; cada activo elige entre
  frecuencia histórica, IA e IA con miedo y tipos con sus años antiguos.
- **Catálogo con nombres:** ~425 activos (37 índices, ~360 empresas con su sector, divisas, materias primas y
  criptomonedas) con búsqueda por nombre, marca (Zara → Inditex), país o sector; lo que no está en la lista se
  busca en Yahoo Finance.
- **Proyección con fiabilidad comprobada:** para cada plazo se elige el método (EWMA, GARCH,
  volatilidad histórica, histórico de movimientos o IA) con el 70 % más antiguo del histórico y se
  informa de sus aciertos en el 30 % más reciente, que no se usó para elegir.
- **IA global entrenada con índices mundiales:** 37 índices de 30 países desde 1990 (crisis de 1997,
  2000, 2008, 2011 y 2020 incluidas). Un modelo cada dos años, entrenado solo con datos anteriores;
  los modelos se guardan en `modelos/global/` y la web no necesita entrenar nada.
- **Laboratorio de señales:** acierto direccional de 11 señales a 1 día, 1 semana y 1 mes frente a un
  azar con la misma mezcla de «sube/baja», con periodo de descubrimiento (antes de 2010) y de
  comprobación (desde 2010), tamaño efectivo corregido por correlación y corrección de Holm.
- **IA entrenable por ticker:** modelo de aprendizaje automático (gradient boosting de cuantiles con
  scikit-learn, sin redes neuronales) que aprende de cada ticker cuándo los movimientos suelen ser
  mayores o menores de lo que indica la volatilidad reciente. Se entrena solo con el pasado, se
  reentrena periódicamente, se corrige con calibración conformal y compite con los demás métodos
  en el informe de calibración. Muestra en qué datos se ha fijado más.
- **Backtesting walk-forward:** cualquier función de estrategia que recibe los datos hasta el día *t*
  y devuelve una posición. Incluye costes y deslizamiento, y se compara siempre con *comprar y
  mantener* y con una línea base ingenua de posición constante.
- **Informe de calibración:** cobertura real frente a la objetivo, ancho medio del intervalo y
  error (puntuación de intervalo) frente a una referencia ingenua.
- **Métricas:** CAGR, volatilidad anualizada, ratio de Sharpe, máximo drawdown, tasa de acierto,
  número de operaciones y rotación.

## Qué NO hace

- No da recomendaciones de compra o venta, precios objetivo ni clasificaciones de "mejores" valores.
- No predice un precio concreto: solo muestra rangos con su incertidumbre y comprueba si están calibrados.
- No garantiza la exactitud de los datos de Yahoo Finance.
- No opera en mercados ni se conecta a ningún bróker.

## Instalación en macOS, paso a paso

1. Instala Python 3.11 o superior (por ejemplo con [Homebrew](https://brew.sh): `brew install python@3.11`).
2. Abre el Terminal y entra en la carpeta del proyecto:

   ```bash
   cd Analizador-Mattionni
   ```

3. Crea un entorno virtual:

   ```bash
   python3.11 -m venv venv
   ```

4. Actívalo (el indicador `(venv)` aparecerá al principio de la línea):

   ```bash
   source venv/bin/activate
   ```

5. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

## Cómo ejecutarla

Con el entorno activado:

```bash
streamlit run app.py
```

Se abrirá el navegador en `http://localhost:8501`. En la barra lateral puedes elegir los tickers,
el rango de fechas, el horizonte y el nivel de confianza del pronóstico, la estrategia de
backtest, los costes y el deslizamiento.

### Línea de comandos (opcional)

```bash
pip install -e .
python -m analizador AAPL --horizon 10
```

Más opciones: `python -m analizador --help` (por ejemplo `--confianza 0.9`, `--metodo garch`,
`--fundamentales`, `--resultados`, `--ia`). Sin instalar el paquete también funciona con `PYTHONPATH=src python -m analizador AAPL`.

### Actualizar el estudio de empresas (y el diario de la IA)

```bash
python -m analizador.estudio
```

Recalcula las ~290 empresas y activos con todos los núcleos del ordenador (≈ 30 minutos) y apunta las
predicciones del día en `modelos/registro/predicciones.csv`, que la página «La IA» comprueba al vencer.

### Reentrenar la IA global

```bash
python -m analizador.entrenamiento
```

Descarga los índices, entrena los modelos de los cinco plazos y rehace el estudio de índices
(unos 10-15 minutos). Conviene repetirlo de vez en cuando para incorporar datos nuevos.

### Tests

```bash
pytest
```

Los tests usan datos sintéticos pequeños y funcionan sin conexión.

## Cómo leer el informe de calibración

Para cada día del histórico se emite un intervalo usando solo los datos disponibles ese día y se
compara con el precio real *h* días después. La tabla tiene una fila por método más la
**referencia ingenua** ("sin cambio" + cuantiles históricos de los retornos a *h* días).

| Columna | Significado |
|---|---|
| **cobertura** | % de veces que el precio real cayó dentro del intervalo. |
| **objetivo** | El nivel de confianza declarado (p. ej. 80 %). |
| **desviación / tolerancia** | Diferencia entre cobertura y objetivo, y el margen estadístico aceptado. |
| **calibrado** | "Sí" si la desviación está dentro de la tolerancia. |
| **ancho medio %** | Anchura media del intervalo en % del precio de partida. Más estrecho es más informativo, *si* sigue calibrado. |
| **puntuación intervalo %** | Error del intervalo (Winkler): ancho + penalización por cada fallo según lo lejos que cae el precio. Menor es mejor. |
| **MAE centro %** | Error absoluto medio del centro del modelo. Solo informativo. |
| **mejora vs ingenuo** | "Sí" si su puntuación es menor que la de la referencia ingenua en el mismo periodo. |
| **útil** | "Sí" solo si está calibrado **y** mejora a la referencia ingenua. |

Cómo interpretarlo:

- Cobertura **muy por debajo** del objetivo → intervalos demasiado estrechos (exceso de confianza).
- Cobertura **muy por encima** → intervalos demasiado anchos (poco informativos).
- Un método puede estar calibrado y aun así no ser útil si la referencia ingenua consigue
  intervalos igual de fiables con menos error.
- El gráfico de **cobertura móvil** muestra si la calibración es estable o si falla en ciertos
  periodos (por ejemplo, en crisis de volatilidad).

Los detalles están en [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Limitaciones conocidas

- **Datos:** Yahoo Finance no es una fuente oficial; puede tener huecos, errores o cambios de
  formato que rompan `yfinance`. Los fundamentales pueden estar retrasados o incompletos y los ETF
  normalmente no los tienen.
- **Resultados y noticias:** Yahoo Finance solo ofrece unos 5 trimestres y 4 años de estados
  financieros, y unas 25 publicaciones de BPA; con tan pocos casos, las estadísticas de reacción
  son muy inciertas. Las noticias suelen estar en inglés y solo cubren los últimos días.
- **Sesgo de supervivencia:** solo se pueden analizar tickers que existen hoy.
- **Modelo de los intervalos:** supone retornos logarítmicos normales; los mercados reales tienen
  colas más gruesas, por lo que los movimientos extremos se subestiman.
- **GARCH simplificado:** se estima por máxima verosimilitud en una rejilla de parámetros, sin
  optimizador numérico; es adecuado para fines educativos, no para producción.
- **Backtest:** ejecución al precio de cierre, costes proporcionales fijos, sin impuestos, sin
  dividendos en liquidez (ya incluidos en el precio ajustado), sin intereses de la liquidez ni
  coste de financiación en posiciones cortas.
- **IA:** con datos diarios hay pocas muestras realmente independientes; en las pruebas con
  AAPL, SPY y MSFT (2016-2026, horizonte 10 días, 80 %) la IA queda calibrada y mejora a la
  referencia ingenua en SPY y MSFT, pero no a EWMA ni a GARCH. Su configuración se eligió mirando
  esos mismos datos, lo que la favorece ligeramente. En Streamlit Community Cloud los modelos
  guardados se pierden al reiniciarse la app y hay que volver a entrenar (unos 15 s).
- **Dirección del precio:** en el laboratorio, ninguna señal clásica acierta la dirección mejor que
  el azar de forma fiable desde 2010, y varias que funcionaban antes dejaron de hacerlo. Lo
  predecible es la magnitud del movimiento (los rangos), no el sentido.
- **IA global:** está entrenada con índices; en acciones individuales (más saltos por resultados)
  puede ser menos fiable. La web lo comprueba ticker a ticker y, si no está calibrada, elige otro método.
- **Activos recién salidos a bolsa** (p. ej. SpaceX, desde junio de 2026): sin un año de historia no se
  puede estimar ni comprobar la dirección; se muestra el Nasdaq solo como referencia y el rango no está verificado.
- **Exceso de confianza de la IA de dirección:** cuando da probabilidades muy altas (≥ 75 %) acierta *menos*
  que de media, así que no se usa para "opinar solo cuando está segura".
- **Sobreajuste:** probar muchas estrategias o parámetros sobre el mismo histórico produce
  resultados optimistas por azar.
- **Rendimiento:** el backtest llama a la estrategia una vez por día, así que estrategias pesadas
  sobre muchos años pueden tardar unos segundos.

## Deploy: publicar en Streamlit Community Cloud

1. Sube el proyecto a un repositorio **público** de GitHub (el `.gitignore` ya excluye `venv/`,
   `data_cache/` y `outputs/`):

   ```bash
   git add .
   git commit -m "Primera versión"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/Analizador-Mattionni.git
   git push -u origin main
   ```

2. Entra en [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
3. Pulsa **Create app** → **Deploy a public app from GitHub**.
4. Elige el repositorio `Analizador-Mattionni`, la rama `main` y como *Main file path* `app.py`.
5. En **Advanced settings** selecciona Python 3.11 (o superior).
6. Pulsa **Deploy**. Streamlit instalará `requirements.txt` y te dará una URL pública del tipo
   `https://<nombre>.streamlit.app`.

Cada `git push` a `main` vuelve a desplegar la app automáticamente. La caché CSV en la nube es
temporal (se borra al reiniciar), pero `st.cache_data` evita descargas repetidas mientras la app
está en marcha.

## Estructura del proyecto

```
app.py                      Interfaz Streamlit (navegación)
vistas/                     Páginas: proyeccion, indices, laboratorio, analisis, comun
modelos/global/             Modelos de la IA global y estudio de índices
src/analizador/
    data.py                 Descarga, limpieza y caché CSV
    indicators.py           Indicadores técnicos
    fundamentals.py         Ratios fundamentales (opcional)
    resultados.py           Estados financieros y publicaciones de resultados
    noticias.py             Noticias recientes
    eventos.py              Reacción del precio a eventos fechados
    ia.py                   IA entrenable (gradient boosting de cuantiles + calibración conformal)
    entrenamiento.py        IA global con índices mundiales y estudio de índices
    proyeccion.py           Proyección a varios plazos con elección y validación de método
    senales.py              Laboratorio de señales direccionales
    indices.py              Lista de índices por país
    catalogo.py             Nombres de acciones, divisas, materias primas…
    direccion.py            Probabilidad de subida (frecuencia histórica e IA de dirección)
    situaciones.py          Pánico y caídas fuertes, contados por episodios
    estudio.py              Estudio precalculado de todas las empresas y activos
    registro.py             Diario de predicciones e historial de entrenamientos
    volumen.py              Volumen inusual y su experimento
    salud.py                Indicadores de salud financiera con semáforo
    mi_lista.py             Activos que sigue el usuario (archivo local)
    alertas.py              Alertas diarias de «Mi lista»
    calendario.py           Próximas presentaciones de resultados y reacción típica
    simulador.py            Simulador educativo de estrategias de compra y venta
    ficha.py                Datos completos de una empresa (valoración, balance, dividendos, analistas)
    per_historico.py        PER de cada día y su media histórica
    tecnico.py              Análisis técnico objetivo: lecturas, soportes y resistencias
    forecast.py             Intervalos de pronóstico
    calibration.py          Informe de calibración
    backtest.py             Motor walk-forward y estrategias de ejemplo
    metrics.py              Métricas de rendimiento y riesgo
    cli.py, __main__.py     Línea de comandos
tests/                      Tests con pytest (sin conexión)
docs/METHODOLOGY.md         Metodología
```

## Licencia

MIT.

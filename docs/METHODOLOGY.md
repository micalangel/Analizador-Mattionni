# Metodología

> Esta herramienta tiene fines educativos y de investigación. No constituye asesoramiento financiero. El rendimiento pasado no predice resultados futuros.

## 1. Datos

- Fuente: Yahoo Finance mediante `yfinance.download(..., auto_adjust=True)`. Los precios OHLC se
  ajustan por dividendos y splits, así que el retorno del precio de cierre ya incluye los dividendos.
- Limpieza (`data.limpiar_ohlcv`): índice de fechas sin zona horaria, ordenado y sin duplicados;
  se eliminan los días sin cierre o con cierre ≤ 0 (no se interpolan precios); Open/High/Low
  faltantes se rellenan con el cierre del día y el volumen faltante con 0.
- Caché: `data_cache/<TICKER>_<inicio>_<fin>.csv`. Las descargas vacías o fallidas no se guardan.

## 2. Ausencia de *look-ahead*

Todo valor calculado para el día *t* usa solo datos disponibles al cierre de *t*:

- Los indicadores usan ventanas móviles y medias exponenciales hacia atrás (nunca centradas).
- El motor de backtest pasa a la estrategia `datos.iloc[:t+1]`: la estrategia no puede ver filas futuras.
- Los intervalos de pronóstico emitidos en *t* usan retornos hasta *t*. El precio realizado
  (`P_{t+h}`) solo se usa después, para evaluar.
- GARCH se reestima cada 63 días con los retornos disponibles hasta ese día.

Los tests lo comprueban explícitamente: se alteran los precios a partir de un día *k* y se verifica
que ningún indicador, posición, retorno de la estrategia ni intervalo anterior a *k* cambia; además,
se registra lo que recibe la estrategia en cada llamada (`tests/test_backtest.py`).

## 3. Reglas del backtest

- **Posición:** fracción del capital, limitada a [−1, 1] (1 = totalmente invertido, 0 = liquidez,
  −1 = corto). Valores no finitos se tratan como 0.
- **Calendario:** la posición decidida al cierre de *t* se mantiene durante el retorno de *t* a *t+1*:

  `retorno_neto_{t} = posición_{t−1} · r_{t} − coste_{t}`, con `r_t = Close_t / Close_{t−1} − 1`.

- **Historia mínima:** durante los primeros 200 días (por defecto) la posición es 0 y esos días no
  se evalúan. Todas las curvas comparadas empiezan el mismo día.
- **Referencias obligatorias:**
  - *Comprar y mantener*: posición 1 desde el primer día evaluado.
  - *Posición constante* (línea base ingenua): la misma exposición (por defecto 0,5) todos los días,
    sin mirar los datos. Si una estrategia no la supera en Sharpe, su *timing* no añade valor frente
    a simplemente reducir la exposición.
- **Ejecución:** al precio de cierre ajustado. No se modelan impuestos, intereses de la liquidez,
  coste de préstamo en cortos ni límites de liquidez.

## 4. Costes

Cada cambio de posición paga, el mismo día en que se decide:

`coste_t = |posición_t − posición_{t−1}| · (coste + deslizamiento)`

- `coste`: comisión proporcional (por defecto 10 puntos básicos = 0,10 %).
- `deslizamiento`: diferencia entre el precio teórico y el obtenido (por defecto 5 pb = 0,05 %).
- Pasar de 0 a 1 cuesta una vez la suma; pasar de 1 a −1 cuesta el doble.
- Comprar y mantener paga solo la entrada.

## 5. Métricas

| Métrica | Definición |
|---|---|
| CAGR | `(∏(1 + r))^(252/n) − 1` |
| Volatilidad anual | desviación típica muestral de los retornos diarios × √252 |
| Sharpe | media / desviación típica de los retornos diarios × √252 (tipo libre de riesgo 0 por defecto) |
| Máx. drawdown | mínimo de `capital / máximo_acumulado − 1`, incluido el capital inicial |
| Tasa de acierto | % de días con exposición (posición del día anterior ≠ 0) y retorno neto > 0 |
| Nº operaciones | días en que la posición cambia, incluida la entrada inicial |
| Rotación anual | `Σ|Δposición|` dividido por el número de años |

## 6. Intervalos de pronóstico

Se modela el retorno logarítmico a *h* días como normal:

`ln(P_{t+h}/P_t) ~ N(μ·h, σ_h²)` → intervalo `P_t · exp(μ·h ± z·σ_h)`, con `z = Φ⁻¹((1 + confianza)/2)`.

| Método | σ_h | μ |
|---|---|---|
| `vol_historica` | desv. típica de los últimos *N* retornos diarios × √h | 0 |
| `vol_historica_deriva` | igual | media de los últimos *N* retornos |
| `ewma` | varianza EWMA (RiskMetrics, λ = 0,94) × h | 0 |
| `garch` | suma de las varianzas GARCH(1,1) previstas para los próximos *h* días | 0 |
| `ingenuo` (referencia) | cuantiles empíricos de todos los retornos a *h* días observados hasta *t* | centro = último precio ("sin cambio") |

GARCH(1,1): `σ²_{t+1} = ω + α·ε²_t + β·σ²_t`. Se fija ω por *variance targeting* y (α, β) se eligen
por máxima verosimilitud gaussiana en una rejilla (α ∈ [0,02; 0,30], β ∈ [0,60; 0,97], α + β < 0,995).

La aplicación muestra solo los límites inferior y superior. El "centro" (mediana del modelo) se usa
internamente para calcular el MAE del informe.

## 7. Cómo se mide la cobertura

Para cada fecha de emisión *t* con datos suficientes y con precio conocido en *t + h*:

1. Se calcula el intervalo `[L_t, U_t]` con datos hasta *t*.
2. Se observa `y = P_{t+h}`.
3. `dentro_t = 1` si `L_t ≤ y ≤ U_t`.

**Cobertura** = media de `dentro_t`. Todos los métodos se evalúan en el **mismo conjunto de fechas**
(la intersección de las fechas en que todos tienen intervalo).

**Tolerancia:** los intervalos a *h* días emitidos en días consecutivos se solapan y no son
independientes, así que se usa un tamaño efectivo `n_ef = n / h`:

`tolerancia = max(0,02; 2 · √(confianza·(1 − confianza) / n_ef))`

Un método está **calibrado** si `|cobertura − confianza| ≤ tolerancia`.

**Ancho medio** = media de `(U_t − L_t) / P_t`.

**Error (puntuación de intervalo de Winkler)**, con α = 1 − confianza:

`S_t = (U_t − L_t) + (2/α)·(L_t − y)·1{y < L_t} + (2/α)·(y − U_t)·1{y > U_t}`, dividido entre `P_t`.

Es una regla de puntuación propia para intervalos: no se puede mejorar haciendo el intervalo
artificialmente estrecho o ancho. Menor es mejor.

**Criterio de utilidad:** un método es útil solo si está calibrado **y** su puntuación media es
menor que la de la referencia ingenua en el mismo periodo. La tabla lo muestra explícitamente.

## 8. Reacción a resultados y noticias

- **Ventana:** publicación tras el cierre (≥ 16:00 Nueva York) o en día sin sesión → del cierre de
  ese día (o el último anterior) al cierre de la sesión siguiente; antes de la apertura o durante
  la sesión → del cierre anterior al cierre de ese día; hora desconocida → dos sesiones.
- **z** = `ln(P_después / P_antes) / (σ_diaria · √sesiones)`, con σ_diaria = desviación típica de
  los 60 retornos logarítmicos hasta el cierre previo al evento (sin datos posteriores).
- **Resumen:** % de publicaciones que superan la estimación, reacción media y absoluta media,
  percentiles 10 y 90, % de reacciones en la dirección de la sorpresa y correlación
  sorpresa-reacción. Son descriptivos y, con ~25 casos, muy inciertos.
- Las noticias solo se listan y se mide el movimiento de su sesión; no se analiza su contenido.

## 9. IA entrenable

- **Características** (fila *t*, solo pasado): volatilidades de 5, 20 y 60 días y su ratio; retornos
  de 1, 5, 20 y 60 días; RSI 14; distancia a las medias de 50 y 200; %B de Bollinger; caída desde el
  máximo de un año; rango diario medio; volumen anómalo; días desde la última publicación de
  resultados y su sorpresa (conocidas desde el cierre del día siguiente); volatilidad y retorno
  de 20 días del mercado (SPY).
- **Objetivo:** `z_t = ln(P_{t+h}/P_t) / σ_h,EWMA(t)`. La IA aprende los cuantiles α/2 y 1 − α/2 de z.
- **Modelo:** `GradientBoostingRegressor(loss="quantile")`, 100 árboles de una división,
  `learning_rate=0.03`, `min_samples_leaf=150`, `subsample=0.8`.
- **Entrenamiento sin look-ahead:** en la fecha *T* solo se usan muestras con `s + h ≤ T`. Se
  reentrena cada 126 sesiones (mínimo 300 muestras) y los intervalos de cada bloque se emiten con
  el último modelo disponible.
- **Calibración conformal (CQR):** del conjunto de entrenamiento, el último 25 % se reserva tras un
  hueco de *h* sesiones. Con `E = max(q_inf − z, z − q_sup)` en ese tramo, los cuantiles se desplazan
  en el cuantil `⌈(n+1)(1−α)⌉/n` de E.
- **Elección de la configuración:** se compararon tres configuraciones de árboles en AAPL, SPY y MSFT
  y se eligió la que mejor puntuación daba. Como esa comparación usa los mismos datos que luego se
  evalúan, los resultados de la IA tienen un pequeño sesgo optimista.
- **Importancias:** reducción media de la pérdida aportada por cada característica en los árboles;
  describen el modelo, no relaciones causales.

## 10. IA global (índices mundiales)

- **Cesta:** 37 índices de 30 países (ver `src/analizador/indices.py`) desde 1990. Las mismas
  características que la IA por ticker, sin las de resultados (los índices no publican BPA).
- **Mercado:** volatilidad y retorno de 20 días del S&P 500 tomados de su último cierre
  **anterior** a cada fecha, porque Asia y Europa cierran antes que EE. UU.
- **Modelo:** `HistGradientBoostingRegressor(loss="quantile")`, 150 iteraciones, 4 hojas,
  `min_samples_leaf=500`, para los cuantiles de las confianzas 50 %, 80 % y 95 %.
- **Sin look-ahead:** cada dos años (1996, 1998, …) se entrena un modelo con las muestras cuyo
  resultado se conocía antes del 1 de enero de ese año. Cada fecha usa el último modelo anterior a
  su año, en cualquier ticker. Un modelo "actual" con todos los datos da solo el rango de hoy.
- **Calibración conformal:** el 25 % más reciente de cada conjunto de entrenamiento, sin
  solapamiento con el de ajuste.
- **Importancias:** por permutación (aumento de la pérdida cuantílica al barajar cada dato).

## 11. Proyección: elección y validación del método

Para cada plazo se evalúan día a día los intervalos de EWMA, GARCH, volatilidad histórica,
histórico de movimientos e IA global. Sobre las fechas comunes:

- **Selección:** el 70 % más antiguo (solo fechas cuyo resultado se conocía antes del tramo de
  validación). Se elige el método calibrado con menor puntuación de intervalo; si ninguno está
  calibrado, el de cobertura más cercana al objetivo.
- **Validación:** el 30 % más reciente. La cobertura del método elegido en este tramo es la
  «fiabilidad comprobada» que muestra la web: *fiable* si está dentro de la tolerancia, *optimista*
  si acierta menos, *prudente* si acierta más y *poca evidencia* si hay menos de 8 periodos
  independientes.

## 12. Laboratorio de señales

- **Pregunta:** ¿predice la señal la dirección del precio a *h* sesiones mejor que el azar?
- **Azar:** acierto esperado de alguien que dice «sube» en la misma proporción w₊ que la señal pero
  al azar: `w₊·p + (1 − w₊)·(1 − p)`, con *p* la frecuencia de subidas de todos los índices en el periodo.
- **Periodos:** descubrimiento antes de 2010; comprobación desde 2010.
- **Tamaño efectivo:** `fechas_con_opinión / h × índices_efectivos`, con
  `índices_efectivos = N / (1 + (N − 1)·ρ)` y ρ la correlación media de los retornos semanales.
- **Significación:** prueba z de la ventaja en la comprobación; p-valores ajustados por Holm sobre
  todas las pruebas. «Supera al azar» exige ventaja positiva en ambos periodos y p ajustado < 0,05.

## 13. Probabilidad de estar más arriba

- **Frecuencia histórica** (en cada t): proporción de periodos de h sesiones ya terminados en t que acabaron
  al alza (mínimo 250 comprobaciones).
- **IA de dirección:** `HistGradientBoostingClassifier` (40 iteraciones, 3 hojas, `min_samples_leaf=3000`)
  con las características de la IA de rangos más la frecuencia histórica; entrenada con los índices mundiales,
  un modelo cada dos años solo con resultados conocidos. Se probaron tres configuraciones en la misma cesta
  (las más complejas empeoraban a la frecuencia histórica); elegir mirando esos datos introduce un pequeño
  sesgo optimista.
- **Elección por activo:** con el 70 % más antiguo de fechas se elige la fuente con menor puntuación de Brier;
  en el 30 % más reciente se informa del acierto del lado (> 50 % = «más arriba») frente a decir siempre
  «más arriba», y de la tabla de calibración («cuando dijo X %, subió Y %»).

## 13 bis. Correcciones por sesgo de supervivencia

- **Frecuencia histórica ajustada:** las empresas del catálogo son las que hoy son grandes, así que su
  historia de subidas es demasiado optimista (las que se hundieron no están). La frecuencia de cada activo se
  acerca a la de los índices mundiales según su evidencia: `(n·f + 20·f_mercado) / (n + 20)`, con
  `n = sesiones de historia / horizonte` y `f_mercado` = 55,3 % / 56,7 % / 58,2 % / 62,7 % / 65,4 % a
  5 días / 10 días / 1 mes / 6 meses / 1 año. Ejemplo: Broadcom a 1 año pasa de 91 % a 77 %.
- **La referencia ingenua no se elige como método del rango:** con activos que han subido mucho, los
  cuantiles de sus movimientos pasados dan rangos irreales (p. ej. un límite inferior por encima del precio
  de hoy). Se sigue calculando solo como referencia de comparación.

## 14. Miedo y tipos de interés (variante «ia_macro»)

- Datos: VIX (`^VIX`), tipo a 10 años (`^TNX`) y a 3 meses (`^IRX`) de EE. UU. desde 1990, siempre del
  cierre **anterior** a cada fecha.
- Características: nivel del VIX, VIX frente a su media de 1 año, cambio del VIX en 20 días, curva de tipos
  (10 años − 3 meses), su cambio en 60 días y el cambio del tipo a 10 años en 1 año.
- Resultado fuera de muestra (37 índices): acierto de dirección 57,9 % / 63,3 % / 67,8 % a 1 mes / 6 meses /
  1 año, frente a 58,8 % / 65,2 % / 67,5 % sin estos datos. No se elige a mano: compite por activo.
- Comprobado y descartado (octubre de 2026): **más información macro**. Variante «ia_amplia» con cambio a
  60 días y 1 año del dólar (`DX-Y.NYB`), petróleo (`CL=F`) y oro (`GC=F`), y la amplitud de las bolsas
  mundiales (% de los 37 índices sobre su media de 200 sesiones y su rentabilidad media a 60 sesiones).
  Fuera de muestra empeoró en todos los plazos (mejora Brier −0,1 % a −2,5 % frente a la frecuencia
  histórica, peor que «ia» e «ia_macro»), así que no se usa. Se puede repetir con
  `python -m analizador.entrenamiento --experimento-amplio`.
- Añadido: tendencia de 5 años (rentabilidad a 1.260 sesiones y caída desde el máximo de 5 años). Efecto
  neutro a corto plazo y ligera mejora a 1 año (mejora Brier de la «ia» +1,15 % → +1,89 %).
- Comprobado y descartado: «opinar solo cuando la IA está segura». Con probabilidades ≥ 75 % a 1 año, el
  mercado subió el 63 % de las veces (menos que la media del 68 %): la IA es demasiado optimista tras rachas buenas.

## 15. Situaciones especiales (por episodios)

- **Reglas fijadas de antemano:** pánico = VIX ≥ 30 (cierre anterior); caída fuerte = precio ≤ −20 % del
  máximo de 252 sesiones.
- **Episodios:** días que cumplen la regla separados por menos de 60 sesiones forman un episodio.
- **Resultado de un episodio:** proporción de sus días en que el precio estaba más arriba al cumplirse el
  plazo; el episodio cuenta «más arriba» si esa proporción supera el 50 %. Solo se resumen episodios completos.
- **Por qué no por días:** contados por días, los pánicos parecen acertar un 83-88 % a 1 año, pero esa cifra
  la dominan dos episodios largos (2008-09 y 2020). Por episodios, el S&P 500 acabó más arriba en 15 de 21
  (6 de 11 antes de 2010; 9 de 10 desde 2010).

## 15 bis. Precio objetivo de los analistas y su historial de aciertos

Yahoo Finance guarda cada cambio de precio objetivo de cada firma (sobre todo en empresas de EE. UU.,
desde 2012 aprox.). Cada fin de mes se calcula el **consenso**: la mediana del último objetivo de
cada firma publicado en los 12 meses anteriores (mínimo 3 firmas), usando solo información
disponible ese día. Los objetivos anteriores a un split se ajustan (eligiendo la versión, ajustada
o no, más coherente con el precio de ese día) y se descartan los incoherentes (fuera de 1/5–5× el
precio).

Cada consenso con un año ya cumplido se compara con el precio un año después (cierre ajustado por
splits, no por dividendos, como los objetivos): acierto de dirección, % de veces que llegó al
objetivo al año o lo tocó durante el año, error absoluto medio y diferencia media entre lo
anunciado y lo ocurrido. Las muestras mensuales se solapan y no son independientes. En empresas
que han subido muchísimo los analistas parecen «prudentes» por un efecto de selección.

## 16. Limitaciones metodológicas

- La normalidad subestima las colas; con niveles de confianza altos (95 %) la cobertura suele quedar
  por debajo del objetivo en activos con saltos.
- La cobertura agregada puede ocultar periodos de mala calibración; por eso se muestra la cobertura móvil.
- Un resultado de calibración histórico no garantiza el comportamiento futuro.

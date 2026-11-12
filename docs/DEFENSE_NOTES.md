# Notas de defensa - UrbanMart Customer Intelligence (Milestone 3)

Lenguaje simple, para repasar antes de la presentación. Los números exactos están en `REPORT_FACTS.md`.

## 1. Qué hace cada archivo (dos líneas cada uno)

- **app.py**: punto de entrada de Streamlit. Configura la página, revisa que existan los modelos,
  muestra las 6 tarjetas KPI y crea las 4 pestañas.
- **src/features.py**: construye la tabla por cliente (RFM + atributos) usando solo compras antes
  del corte 2024-11-01. Crea el target `churn_risk_flag` mirando los 60 días siguientes.
- **src/train_models.py**: divide 80/20, ajusta el preprocesador solo con train, entrena LR, árbol,
  ANN y K-Means. Guarda todo en `models/*.joblib` y las métricas de test en `reports/`.
- **src/app_helpers.py**: rutas con pathlib, cargadores con `@st.cache_data` / `@st.cache_resource`,
  revisión de versiones, predicción y la regla de vouchers (`decide_policy`).
- **src/tab_eda.py**: pestaña 1. Muestra datos crudos, `describe()`, histogramas por churn y
  tasa de churn por región y categoría.
- **src/tab_preprocessing.py**: pestaña 2. Nulos/duplicados, efecto del log, efecto del
  StandardScaler guardado, one-hot y mapa de correlación de Pearson.
- **src/tab_benchmark.py**: pestaña 3. Lee las métricas de test ya guardadas: tabla, matrices de
  confusión, curvas ROC y curva de pérdida de la ANN, con interpretación.
- **src/tab_whatif.py**: pestaña 4. Sliders y selectores, carga el modelo elegido, calcula la
  probabilidad en vivo, predice el segmento K-Means y muestra la alerta de voucher.
- **tests/smoke_test.py**: prueba automática con `AppTest`: la app arranca, cambia sliders y
  verifica la alerta de bajo y alto riesgo.
- **.streamlit/config.toml**: tema oscuro con color de acento verde.
- **models/metadata.json**: versiones de librerías, columnas, corte, balance de clases, resultados.

## 2. Flujo de datos (del CSV a la predicción)

1. `UrbanMart_Transactions.csv` (50,000 filas) ->
2. `features.py`: agrupa por cliente con compras **antes** del 2024-11-01 -> Recency, Frequency,
   Monetary, logs y categorías; target = ¿no compró en los 60 días siguientes? ->
3. `churn_modeling_table.csv` (2,000 clientes) ->
4. `train_models.py`: split estratificado 80/20 -> preprocesador (imputer + scaler + one-hot)
   ajustado con train -> modelos -> `models/*.joblib` + métricas de test en `reports/` ->
5. `app.py`: carga los `.joblib` (sin reentrenar) -> en What-If, los valores de los sliders pasan
   por el **mismo** preprocesador guardado -> `predict_proba` -> regla de voucher.

## 3. ¿Por qué el split temporal?

El dataset no trae etiqueta de churn. Si dijera "churn = Recency > 90 días" y luego usara Recency
como variable, el modelo solo copiaría la regla: ROC-AUC casi 1.0 pero sin valor (fuga de
información o *leakage*). Con el split temporal, las variables solo ven el pasado (antes del corte)
y el target se mide en el futuro (60 días después). Es igual a la vida real: hoy predigo quién
no va a comprar en los próximos 2 meses.

Elegí el corte 2024-11-01 porque con 2024-10-01 solo 4.3% eran churners (muy desbalanceado);
con 2024-11-01 son 13.1% (261 de 2,000).

## 4. ¿Por qué estos modelos?

- **K-Means (k=3)**: segmentación RFM del Milestone 2 (Champions, Developing, At Risk). No usa el
  target; se mide con Silhouette (0.383).
- **Regresión Logística**: modelo clásico, simple e interpretable (coeficientes). Línea base.
- **Árbol de decisión (profundidad 4)**: segundo modelo clásico, reglas fáciles de explicar.
- **ANN (MLPClassifier 32-16)**: red neuronal pedida por la rúbrica, capta relaciones no lineales.
  Usé scikit-learn porque TensorFlow/PyTorch pueden fallar en Python 3.14.
- Desbalance: `class_weight="balanced"` en LR y árbol; la MLP no tiene esa opción, así que la
  entrené con sobremuestreo aleatorio de churners (solo en train).

## 5. Resultado honesto (tenerlo muy claro)

Todos los ROC-AUC están cerca de 0.5 (LR 0.538, árbol 0.523, ANN 0.530). El dataset es sintético:
las compras están repartidas casi uniformes en el tiempo, así que el pasado casi no predice quién
deja de comprar. **No es un error del código**: es lo que pasa cuando se evita la fuga de
información. Mejor un 0.54 honesto que un 0.99 falso. Para el negocio: los vouchers deben ser
baratos, porque muchos clientes señalados no se iban a ir.

## 6. Demo de 8 minutos, pestaña por pestaña

- **0:00-1:00 Intro y KPIs**: problema (retención, gerente de CRM), target churn_risk_flag,
  leer las 6 tarjetas: 2,000 clientes, 50,000 transacciones, churn 13.1%, At Risk 13.8%, mejor modelo 0.538.
- **1:00-2:30 Pestaña 1 EDA**: muestra de datos, describe, cambiar el histograma (Recency, Monetary)
  coloreado por churn: las distribuciones casi se enciman -> poca señal. Barras por región/categoría.
- **2:30-4:00 Pestaña 2 Preprocesamiento**: 0 nulos y 0 duplicados (pero hay imputer). Log de
  Monetary (baja el sesgo). Scaler antes/después (media 0, desviación 1). One-hot: 4 categorías ->
  20 columnas. Heatmap: correlaciones con el target cerca de 0. Leer la nota del split temporal.
- **4:00-5:45 Pestaña 3 Benchmark**: remarcar "test set". Tabla: LR gana en ROC-AUC y F1, árbol en
  recall, ANN en accuracy. Matrices de confusión. ROC pegadas a la diagonal. Curva de pérdida:
  converge en 317 iteraciones, pero train 0.914 vs test 0.530 = sobreajuste.
- **5:45-7:30 Pestaña 4 What-If**: preset "At Risk" -> alerta roja. Escenario VIP (Recency 5,
  Freq 35, Monetary 90M, Yogyakarta, 36-50, Male, Sport) -> verde con VIP voucher. Escenario alto
  (Recency 0, Freq 40, Monetary 100M, Medan, 18-25, Female, Fashion) -> rojo con voucher fuerte.
  Mover el threshold y cambiar de modelo para mostrar que reacciona. Mostrar el historial.
- **7:30-8:00 Cierre**: pipeline serializado, GitHub con PR revisado, limitación (datos sin señal)
  y siguiente paso (datos reales con más variables de comportamiento).

## 7. Preguntas probables y respuestas cortas

1. **¿Cómo definieron churn_risk_flag?** 1 si el cliente no compra en los 60 días después del
   2024-11-01; las variables solo usan compras anteriores al corte.
2. **¿Por qué no usaron Recency > X días como churn?** Porque Recency también es variable: el modelo
   copiaría la regla (leakage) y el resultado sería falso.
3. **¿Por qué el ROC-AUC es tan bajo?** Los datos son sintéticos y uniformes; ninguna variable sola
   pasa de 0.54 de AUC. Es un resultado honesto, no un bug.
4. **¿Cómo sabes que las métricas no son de entrenamiento?** `train_models.py` evalúa con `x_test`
   (20% estratificado que nunca se usó para ajustar ni el scaler ni los modelos). La app solo lee esos CSV.
5. **¿La ANN sobreajusta?** Sí: ROC-AUC 0.914 en train vs 0.530 en test. Puse alpha=1.0 para
   reducirlo; aun así memoriza ruido.
6. **¿Cómo manejaron el desbalance?** Split estratificado, `class_weight="balanced"` en LR y árbol,
   sobremuestreo de churners solo en train para la MLP, y un threshold ajustable en la app.
7. **¿Por qué el R² es negativo?** Se calcula sobre la probabilidad contra la etiqueta 0/1; los
   modelos balanceados dan probabilidades cerca de 0.5 y el 87% son ceros. No es buena métrica para clasificación.
8. **¿Cómo cargan los modelos en la app?** `joblib.load` dentro de `@st.cache_resource`; no se
   reentrena nada. Si falta un archivo o cambia la versión de scikit-learn, sale un `st.error`.
9. **¿Para qué sirve st.session_state?** Guarda los valores de los sliders, los presets y el
   historial de escenarios entre cada rerun de Streamlit.
10. **¿Cómo evitaron errores de rutas?** Todas las rutas usan `Path(__file__).resolve()`; probamos
    lanzar la app desde otra carpeta.
11. **¿Por qué en What-If subir Recency baja un poco la probabilidad?** Con datos sin señal, la LR
    aprendió un coeficiente casi cero y negativo (-0.03); pesan más la categoría y la edad. Lo
    reconocemos como limitación.
12. **¿Reprodujeron el K-Means del Milestone 2?** Sí: mismo escalado y semilla, Silhouette 0.3827 y
    tamaños 996 / 729 / 275 idénticos.

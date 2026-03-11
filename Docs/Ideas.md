Ruido, Analisis Estadistico

Dashboard

Self Supervised Learning


**Adiós al Accuracy como métrica reina:** El Accuracy ya no nos sirve de nada. Un 75% de Accuracy significará que tu modelo es un fracaso (solo predice Sanos). Tendremos que evaluar tu modelo basándonos estrictamente en el **Macro F1-Score** y observar muy de cerca el *Recall* (Sensibilidad) de la clase 2 (AD).

**Resurrección de los Pesos de Clase (Class Weights):** La función de pérdida (CrossEntropy) va a necesitar pesos brutales. Un error al clasificar a un paciente con AD (clase minoritaria) va a tener que "dolerle" a la red 10 veces más que un error en un paciente Sano.

**Muestreo Inteligente (Opcional pero recomendado):** Como tenemos tantísimos Sanos (6479), nos podemos plantear hacer un *Downsampling* (descartar aleatoriamente a la mitad de los sanos) para no saturar a la red viendo siempre el mismo tipo de cerebro, dejando un dataset más equilibrado (ej. 2000 Sanos, 1444 MCI, 702 AD).

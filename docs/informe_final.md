# 📈 Informe Final de Proyecto: Ingeniería de Machine Learning y Deep Learning
# 🏨 Motor de Predicción de Cancelaciones Hoteleras (Pipeline End-to-End)

## 👥 1. Definición de Roles y Distribución del Trabajo

El desarrollo de este proyecto se realizó bajo una modalidad de **trabajo en equipo equitativo y coordinado**. En lugar de dividir las tareas de forma aislada, ambos integrantes participamos activamente en el diseño de la estructura del código, la corrección de errores y la validación de todo el flujo de datos.

* **Autor 1:** Said El Khababi Balamkadam
  * **Rol:** Desarrollador del proyecto.
  * **Contribuciones:** Análisis exploratorio de datos (EDA), limpieza y preprocesamiento, entrenamiento y evaluación de los modelos, redacción del informe y revisión final del código.
* **Autor 2:** Marvin Bernal
  * **Rol:** Desarrollador del proyecto.
  * **Contribuciones:** Análisis exploratorio de datos (EDA), limpieza y preprocesamiento, entrenamiento y evaluación de los modelos, redacción del informe y revisión final del código.

**Declaración de Colaboración:** Ambos autores compartimos la responsabilidad y autoría de todos los módulos del software al 100%. Todo el código fue unificado, probado y validado en conjunto para asegurar el correcto funcionamiento del sistema.

📦 **Repositorio del Proyecto:** [GitHub - ML/DL Final Project](https://github.com/MarvinBernal90/proyecto-final-ml-dl)

---

## 🎯 2. Justificación del Problema

En el sector hotelero, las cancelaciones imprevistas de reservas son uno de los problemas económicos y operativos más difíciles de gestionar. Cuando un cliente cancela a última hora, el hotel pierde ingresos por habitaciones vacías que ya no puede volver a vender, se complica la planificación del personal y se desorganiza el control de los suministros.

Este proyecto aborda este desafío mediante un enfoque de **Analítica Predictiva**, modelándolo como un problema de **Clasificación Binaria**. Al analizar las características históricas y los detalles de cada reserva (como los días de anticipación o el tipo de cliente), el sistema calcula la probabilidad de que una reserva sea cancelada (`is_canceled`).

Contar con este modelo predictivo le da al hotel una gran ventaja para tomar mejores decisiones: permite gestionar de forma segura estrategias como la sobreventa (*overbooking*) controlada, solicitar depósitos anticipados a las reservas que tengan mayor probabilidad de cancelación y optimizar la organización del personal en el día a día.

---

## 💾 3. Descripción y Auditoría del Dataset

Para este proyecto utilizamos un conjunto de datos que contiene **119,390 registros** y **32 variables** iniciales con toda la información sobre los flujos de las reservas. Estas variables incluyen datos de tiempo (`lead_time`, fechas de llegada), cantidad de huéspedes (adultos, niños), información financiera (`adr`), detalles del mercado (`market_segment`, `deposit_type`) e historial de comportamiento del cliente.

* **Variable Objetivo:** `is_canceled` (Variable Binaria)
  * `0` $\rightarrow$ **Reserva No Cancelada (Safe Booking):** El huésped se presentó en el hotel y completó su estancia.
  * `1` $\rightarrow$ **Reserva Cancelada (Cancellation Risk):** El cliente canceló la reserva o no se presentó (*No-Show*).

🛡️ **Control de Fuga de Datos (Data Leakage):** Durante el análisis inicial, descubrimos que las variables `reservation_status` y `reservation_status_date` ya contenían la respuesta final de lo que pasó con la reserva (es decir, revelaban si el cliente había cancelado o no antes de tiempo). Dejar estas variables en el entrenamiento haría que los modelos hicieran "trampa", dando resultados falsamente perfectos que no funcionarían en la realidad. Por lo tanto, el código las elimina por completo antes de entrenar cualquier algoritmo.

---

## 🔍 4. Análisis Exploratorio de Datos (EDA)

Durante la fase de análisis exploratorio (que se encuentra detallada paso a paso en nuestro Jupyter Notebook), encontramos los siguientes puntos clave sobre la estructura de los datos:

* **Tamaño del Dataset:** El conjunto de datos inicial cuenta con 119,390 filas y 32 columnas.
* **Análisis de Valores Nulos:** * La variable del país (`country`) tenía algunos pocos datos faltantes.
  * Las variables de `agent` y `company` tenían una gran cantidad de datos vacíos. En particular, **`company` tenía más del 94% de valores nulos**. Por esta razón, decidimos sacarla del proyecto, ya que intentar inventar o rellenar tantos datos vacíos (imputación) terminaría confundiendo a los modelos.
* **Errores en los Precios:** Encontramos algunos registros con valores negativos en la tarifa diaria (`adr`). Al ser imposible cobrar un precio negativo, los tratamos como errores de dedo o de captura del hotel y los eliminamos.
* **Estudio de Datos Duplicados:** Detectamos 32,252 filas idénticas usando código de Python. Tras analizar el problema a fondo, concluimos que no eran errores del sistema, sino **reservas legítimas hechas en bloque por agencias de viajes** para grupos grandes o convenciones. Eliminar estas filas solo por estar repetidas habría sido un error, ya que estaríamos borrando precisamente uno de los segmentos de clientes que más cancelan en la realidad.

---

## ⚙️ 5. Flujo de Limpieza y Preprocesamiento de Datos

Para asegurarnos de que el proyecto funcione de forma automática, ordenada y sin errores, organizamos todas las transformaciones de los datos en un orden secuencial dentro de nuestro script `preprocessing.py`:

1. **Lectura Segura de Datos:** Cargamos el conjunto de datos desde la carpeta `data/raw/` sin modificar ni alterar el archivo original.
2. **Filtrado y Limpieza:** Eliminamos los registros con precios negativos (`adr >= 0`) y quitamos de forma definitiva las columnas que causaban fuga de datos (`reservation_status`, `reservation_status_date`) y la columna `company` por exceso de nulos.
3. **Estrategia para Datos Faltantes (Imputación):** * Los valores nulos en `children` y `agent` se rellenaron con la **mediana** para que los valores extremos no afecten los cálculos.
   * Los nulos en la columna de país se marcaron con la etiqueta **`'Unknown'`** para no perder esos registros.
4. **Separación de Datos (Train-Test Split):** Dividimos los datos en un 80% para entrenamiento y un 20% para pruebas. Este paso se hace **antes** de cualquier codificación o escalado para garantizar que el set de prueba se mantenga totalmente oculto y simule datos nuevos del futuro.
5. **Codificación y Alineación de Columnas (`X_train_encoded`):** Aplicamos *One-Hot Encoding* para convertir los textos en números (unos y ceros). Para evitar que el set de entrenamiento y el de prueba terminen con tamaños diferentes, usamos el truco de **`.align(join='left')`**. Esto generó las matrices **`X_train_encoded`** y **`X_test_encoded`**, asegurando que ambas tengan exactamente las mismas **243 columnas** y que el modelo no se rompa al hacer predicciones.
6. **Escalado de los Datos:** Aplicamos un `StandardScaler` a las variables numéricas. El escalador calcula la media y la varianza utilizando únicamente los datos de entrenamiento (`.fit_transform`) y luego aplica esa misma transformación al set de prueba (`.transform`), protegiendo el pipeline contra la fuga de datos.

---

## 📐 6. Estructura del Sistema y Organización del Código

Para este proyecto dejamos atrás la estructura tradicional de trabajar únicamente en un archivo de Jupyter Notebook. En su lugar, organizamos el código en un paquete de scripts de Python (`.py`) separados por funciones para que el proyecto sea limpio, ordenado y fácil de mantener:

```text
src/
│   ├── __init__.py
│   ├── config.py         # Archivo central con las variables, rutas de carpetas y parámetros fijos.
│   ├── preprocessing.py  # Código encargado de la limpieza de datos, One-Hot Encoding y escalado.
│   ├── model_trainer.py  # Script para entrenar los modelos tradicionales (Logistic Regression, Random Forest, XGBoost).
│   ├── nn_trainer.py     # Código para armar y entrenar la Red Neuronal usando TensorFlow/Keras.
│   ├── evaluator.py      # Módulo para comparar las métricas, graficar curvas y elegir el mejor modelo (Champion).
│   └── predictor.py      # Script encargado de tomar datos nuevos y usar el mejor modelo para dar una predicción.
```
- Flujo Central (main.py): Creamos un script principal que funciona como el botón de encendido del proyecto. Al ejecutarlo, corre de forma secuencial todo el proceso (limpia los datos, entrena los modelos, los evalúa y guarda el mejor resultado) de manera automática y sin tener que mover nada manualmente.

- Aplicación Web (app.py): Diseñamos una interfaz gráfica sencilla e interactiva en Streamlit para que cualquier persona, pueda ingresar los datos de una nueva reserva y ver en tiempo real si el modelo predice que el cliente asistirá o cancelará.

## 🤖 7. Modelos Implementados y Métricas de Evaluación

Para resolver el problema, entrenamos y comparamos el rendimiento de cinco algoritmos diferentes:

### 1. Logistic Regression: 
Nuestro modelo base lineal, utilizado para establecer un punto de partida y comparar los resultados.

### 2. Decision Tree: 
Un clasificador no lineal que ayuda a entender las decisiones basadas en reglas simples.

### 3. Random Forest: 
Un modelo de ensamble que combina múltiples árboles de decisión (técnica de Bagging) para mejorar la precisión.

### 4. XGBoost: 
Un algoritmo avanzado de ensamble basado en árboles que optimiza los errores de forma secuencial (Gradient Boosting).

#### 5. Red Neuronal (DNN): 
Un Perceptrón Multicapa creado con TensorFlow/Keras, configurado para detenerse automáticamente si deja de mejorar (Early Stopping) dentro de un límite de 30 épocas.

### 🎯 Selección de la Métrica Principal
Aunque revisamos los resultados generales de precisión como el *Accuracy*, nos enfocamos principalmente en el **ROC-AUC** y el **F1-Score** para tomar la decisión final. 

Esto se debe a que las cancelaciones en el hotel no están perfectamente balanceadas (hay más personas que asisten de las que cancelan). Si usáramos solo el *Accuracy*, el modelo podría parecer perfecto simplemente diciendo que nadie va a cancelar. La métrica **ROC-AUC** nos asegura que el modelo sea realmente bueno identificando tanto a los clientes que van a asistir como a los que van a cancelar, evaluando las probabilidades de forma justa y sin engañarnos con resultados falsamente altos.

## 📊 8. Resultados, Evaluación y Elección del Mejor Modelo

Luego de ejecutar todo nuestro flujo de código en Python y evaluar el rendimiento de cada algoritmo con el set de datos de prueba (los datos que los modelos no conocían), obtuvimos los siguientes resultados finales:

| Modelo de Machine Learning | Librería Utilizada | Exactitud (Accuracy) | ROC-AUC | Estado del Modelo |
| :--- | :--- | :---: | :---: | :--- |
| **Random Forest** | Scikit-Learn | **0.8935** | **0.9589** | 🏆 **Ganador / Modelo Campeón** |
| **Red Neuronal (DNN)** | TensorFlow / Keras | 0.8711 | 0.9444 | Excelente rendimiento (Deep Learning) |
| **XGBoost** | XGBoost Framework | 0.8642 | 0.9420 | Alto rendimiento basado en árboles |
| **Decision Tree** | Scikit-Learn | 0.8450 | 0.9231 | Modelo base de árbol único |
| **Logistic Regression** | Scikit-Learn | 0.8168 | 0.8949 | Modelo base lineal |

### 📈 Análisis y Conclusiones Técnicas

* **El éxito de Random Forest:** Se consolidó como el ganador absoluto del proyecto con un excelente **0.9589 de ROC-AUC** (y 89.35% de Accuracy). Al utilizar la combinación de muchos árboles de decisión (*Bagging*), demostró una gran capacidad para entender los patrones repetitivos de las reservas grupales y masivas (las cuales decidimos dejar tras el EDA), logrando aprender muy bien sin caer en problemas de sobreajuste (*overfitting*).
* **Rendimiento de la Red Neuronal:** El modelo de aprendizaje profundo creado con Keras funcionó bastante bien, alcanzando un **0.9444 de ROC-AUC**. Su error de validación se estabilizó por completo en la época 27, demostrando que las capas neuronales son capaces de procesar bases de datos grandes y complejas, aunque requiere más recursos y tiempo de cómputo que los modelos de árboles.
* **Factores clave para predecir:** Al revisar el peso que el modelo le da a cada variable, confirmamos que el historial del cliente (como las cancelaciones previas `previous_cancellations`) junto con los detalles de riesgo de la reserva (el tipo de depósito `deposit_type` y los días de anticipación `lead_time`) son las características más importantes que usan los algoritmos para descubrir si una reserva se va a cancelar.

---

## 🛠️ 9. Modelo Final y Uso en la Aplicación Web

Nuestro código guarda de forma automática el modelo que obtuvo el mejor rendimiento en la carpeta `models/champion_model.pkl`. 

Para poder usar este modelo con datos nuevos, creamos el script independiente `src/predictor.py`. Este archivo está diseñado para tomar los datos de una nueva reserva, aplicarles la misma limpieza y el escalado guardado en `fitted_scaler.pkl`, y entregar un resultado muy fácil de entender con tres puntos clave:

1. El estado actual o real de la reserva.
2. La predicción del modelo (si cancelará o no).
3. El porcentaje de probabilidad de que esa reserva sea cancelada.

Este script se conecta directamente con nuestra aplicación web **`app.py` (Streamlit)**. De esta manera, se puede interactuar con el modelo final utilizando un formulario web sencillo, y controles deslizantes, viendo los resultados y alertas en tiempo real.

---

## 🛑 10. Reflexión Crítica: Limitaciones y Mejoras Futuras

A pesar del excelente rendimiento obtenido y de haber logrado una estructura limpia en nuestro código, identificamos algunas limitaciones técnicas que podrían mejorarse en futuras versiones del proyecto:

* **Falta de Ajuste Fino de Parámetros (Hyperparameter Tuning):** Entrenamos los modelos utilizando configuraciones estándar que sabíamos que funcionaban bien. Sin embargo, no incluimos una etapa de búsqueda automática avanzada (como *GridSearchCV* o *RandomizedSearchCV*) que ayudara a probar combinaciones automáticas para exprimir el máximo potencial de algoritmos como XGBoost o la Red Neuronal.
* **Dificultad para Explicar las Predicciones (Efecto Caja Negra):** Los modelos basados en muchos árboles (como Random Forest) y las redes neuronales son excelentes prediciendo, pero es difícil entender su lógica interna. Al no incluir librerías de explicabilidad como **SHAP** o **LIME**, solo puede ver el resultado final (si cancela o no), pero no puede saber exactamente *por qué* el modelo tomó esa decisión para una reserva en específico.
* **Impacto de las Temporadas del Año:** Aunque la métrica ROC-AUC nos ayudó a controlar el desbalance general de los datos, el comportamiento de los clientes puede cambiar drásticamente dependiendo de la temporada (por ejemplo, verano vs. invierno). En el futuro, se podrían aplicar técnicas de remuestreo avanzado (como **SMOTE**) o incluir variables climáticas y festivas para ajustar mejor el modelo a estos cambios de temporada.

---

## 🏁 11. Conclusiones Finales del Proyecto

Este proyecto nos permitió demostrar cómo un problema real del sector turístico, como lo son las cancelaciones de reservas, puede analizarse y resolverse de manera eficiente utilizando herramientas de Inteligencia Artificial. Logramos avanzar con éxito desde una fase inicial de exploración libre en cuadernos de Jupyter, hasta construir un **sistema ordenado, modular y automático en Python que funciona bajo las buenas prácticas del desarrollo de software**.

El modelo que seleccionamos (**Random Forest**, con un 0.9589 de ROC-AUC) nos ofreció el mejor equilibrio entre precisión matemática y estabilidad. Al conectar este modelo de forma directa con una aplicación web interactiva en **Streamlit**, logramos que un desarrollo basado en algoritmos complejos se convierta en una herramienta visual y fácil de usar.
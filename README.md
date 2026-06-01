# 🏨 Motor de Predicción de Cancelaciones Hoteleras (Pipeline End-to-End)

## 👤 Autores de la Práctica

* **Said El Khababi Balamkadam**
* **Marvin Bernal**

---

## 🎯 Descripción del Problema y Datos

En el sector hotelero, cuando un cliente cancela su reserva a último momento, el hotel pierde dinero y se queda con habitaciones vacías que ya no puede vender. Esto afecta tanto a las ganancias como a la organización del personal. El objetivo de este proyecto es aplicar la ciencia de datos para resolver este problema. Para ello, desarrollaremos un modelo de Machine Learning que trate este caso como un problema de **Clasificación Binaria**, lo que nos permitirá predecir con anticipación si una reserva se mantendrá en pie o será cancelada.

El objetivo es analizar los datos de las reservas para predecir si un huésped va a asistir o a cancelar. A través de estas predicciones, se generarán alertas sencillas que ayuden al personal del hotel a saber qué reservas tienen un alto riesgo de cancelarse.

* **Variable Objetivo:** `is_canceled`
  * `0` → **Reserva No Cancelada (Safe Booking):** El huésped llegó o realizó el check-in en el hotel.
  * `1` → **Reserva Cancelada (Cancellation Risk):** El huésped canceló o no se presentó (*No-Show*).

---

## 📁 Estructura del Proyecto y Arquitectura de Directorios

El espacio de trabajo sigue patrones de diseño de software modular a nivel empresarial para garantizar la reproducibilidad en ingeniería de Machine Learning:

```text
proyecto-final-ml/                  # Raíz principal del proyecto
│
├── data/                           # Capa de datos (Excluida de Git via .gitignore)
│   ├── raw/                        # Datos crudos originales e inmutables
│   │   └── hotel_bookings.csv      # Dataset fuente original 
│   └── processed/                  # Particiones aisladas, codificadas y escaladas
│       ├── X_train.csv / X_test.csv
│       └── y_train.csv / y_test.csv
│
├── docs/                           
│   └── informe_final.md            # Informe final del proyecto
│
├── models/                         # Registro de Modelos Serializados y Artefactos
│   ├── fitted_scaler.pkl           # Parámetros entrenados del StandardScaler
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── nn_keras_model.keras        # Checkpoint nativo de la Red Neuronal (.keras)
│   └── champion_model.pkl          # Copia automatizada del mejor modelo obtenido
│
├── notebooks/                      # Notebooks de Jupyter para investigación interactiva
│   └── 1.0-exploration-and-training.ipynb
│
├── src/                            # Capa de paquetes funcionales
│   ├── __init__.py
│   ├── config.py                   # Rutas centralizadas, esquemas e hiperparámetros
│   ├── preprocessing.py            # Pipeline de datos: Aislamiento, Codificación y Escalado
│   ├── model_trainer.py            # Bucle automatizado para entrenamiento ML tradicional
│   ├── nn_trainer.py               # Configuración de la Red Neuronal con TensorFlow
│   ├── evaluator.py                # Auditoría de métricas y serialización del campeón
│   └── predictor.py                # Motor de inferencia para datos de producción en tiempo real
│
├── app.py                          # Aplicación web interactiva con Streamlit
├── main.py                         # Orquestador central para la ejecución completa end-to-end
├── requirements.txt                # Dependencias fijadas del entorno de software
└── README.md                       # Documentación técnica del sistema
```

## ⚙️ Preprocesamiento Automatizado y Mitigación de Fuga de Datos

Para proteger la integridad en producción y aislar los estimadores de cualquier contaminación de información, los datos fluyen a través de un orden estricto en el pipeline:

1. **Eliminación de Data Leakage:** Las variables de estado posteriores (`reservation_status`, `reservation_status_date`) se descartan estrictamente durante la fase de lectura de datos crudos.

2. **Preservación de Duplicados por Lógica de Negocio:** El dataset contiene 32,252 filas idénticas. Estos registros se conservan intencionalmente basándose en la lógica del dominio: en operaciones hoteleras que carecen de un ID único de transacción, las filas idénticas representan reservas legítimas en bloque de operadores turísticos o eventos grupales. Eliminarlas introduciría un sesgo severo en contra del segmento de mercado de Grupos.

3. **Aislamiento de Particiones:** El dataset se divide en conjuntos de entrenamiento (80%) y prueba (20%) **antes** de cualquier transformación de los datos.

4. **Codificación Categórica Aislada:** Se aplica One-Hot Encoding por separado. Una alineación de esquema posterior a la codificación (`.reindex(fill_value=0)`) obliga a ambas matrices a adoptar una estructura determinista de **247 variables ordenadas alfabéticamente**.

5. **Escalamiento Estadístico:** Las columnas numéricas se transforman mediante un `StandardScaler`. Los parámetros se aprenden con `.fit_transform()` *exclusivamente* de `X_train_encoded`, dejando la partición `X_test_encoded` como un conjunto de validación completamente ciego.

---

## 🛠️ Instrucciones para Ejecutar el Proyecto

Esta arquitectura utiliza un marco de orquestación automatizado de un solo comando, eliminando por completo las ejecuciones secuenciales manuales.

### 1. Inicialización del Entorno e Instalación

Para garantizar la reproducibilidad del proyecto, cree un entorno virtual aislado e instale las dependencias del proyecto siguiendo estos pasos:

#### 1. Crear el entorno virtual
Se recomienda el uso de **Python 3.10 o superior**. Ejecute el siguiente comando en su terminal:
```bash
python3 -m venv .venv
```

(Nota: Si está en Windows y el comando anterior falla, intente utilizando únicamente python -m venv .venv)

#### 2. Activar entorno virtual
Dependiendo de su sistema operativo, ejecute el comando correspondiente:

##### En Linux/macOS:
```bash
source .venv/bin/activate
```

##### En Windows (Git Bash / PowerShell):
```bash
.venv\Scripts\activate
```

#### En Windows (CMD - Símbolo del sistema):
```bash
.venv\Scripts\activate
```

#### 3. Instalar dependencias del proyecto
Una vez activado el entorno virtual, actualice el gestor de paquetes e instale las librerías necesarias:
```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 2. Ejecutar el Pipeline de Entrenamiento Completo (End-to-End)

Para ejecutar la limpieza de datos, entrenar los 4 algoritmos tradicionales, entrenar la Red Neuronal de TensorFlow, auditar los resultados y extraer el modelo campeón de forma totalmente automática, ejecuta el orquestador centralizado:

```bash
python3 main.py
```

### 3. Lanzar la Interfaz Web de Producción
Para acceder al panel interactivo de riesgo operativo utilizado por los administradores del hotel, ejecuta la aplicación con Streamlit:

```bash
streamlit run app.py
```

Una vez inicializado, abre tu navegador web e ingresa a http://localhost:8501 para utilizar el formulario de inferencia y la pestaña de métricas del modelo.

## 📊 Métricas de Evaluación, Resultados y Conclusiones

Todos los modelos fueron evaluados utilizando **ROC-AUC** como métrica principal de selección, junto con **Accuracy**, asegurando una validación equilibrada de la confianza de clasificación.

### Tabla de Resumen de Resultados (Leaderboard Final)

| Arquitectura del Modelo | Toolkit de Backend | Exactitud (Accuracy) | ROC-AUC | Estado |
| :--- | :--- | :---: | :---: | :--- |
| **Random Forest** | Scikit-Learn | **0.8935** | **0.9589** | 🏆 **Modelo Campeón Absoluto** |
| **Red Neuronal (DNN)** | TensorFlow / Keras | 0.8711 | 0.9444 | Mejor Alternativa de Deep Learning |
| **XGBoost** | Framework XGBoost | 0.8642 | 0.9420 | Alto Rendimiento |
| **Decision Tree** | Scikit-Learn | 0.8450 | 0.9231 | Baseline de Árbol |
| **Logistic Regression** | Scikit-Learn | 0.8168 | 0.8949 | Baseline Estadístico |

## 📈 Análisis de Resultados e Insights

* **El desempeño de Random Forest:** Este modelo fue el que obtuvo el mejor resultado, superando a XGBoost y a la Red Neuronal con un **ROC-AUC de 0.9589**. Gracias a que utiliza la técnica de Bagging (combinar múltiples árboles entrenados con distintas muestras de datos), logró manejar muy bien las variables categóricas y los patrones repetitivos sin caer en problemas de sobreajuste (*overfitting*).

* **Resultados de la Red Neuronal (Deep Learning):** El modelo de aprendizaje profundo (un Perceptrón Multicapa creado con Keras) se entrenó durante 30 épocas, logrando que el error de validación se estabilizara alrededor de la época 27. Esto nos muestra que, aunque estas arquitecturas profundas son muy buenas para entender datos complejos, tardan bastante más tiempo en entrenarse que los modelos basados en árboles de decisión.

* **Variables más importantes:** Al revisar qué características influyeron más en las predicciones, confirmamos que el historial del cliente (`previous_cancellations`) junto con factores de riesgo de la reserva (como el tipo de depósito `deposit_type` y los días de anticipación `lead_time`) representan la mayor parte del peso que usan los modelos para decidir si una reserva se cancelará o no.

---

## 🏁 Conclusiones Finales

* **Estructura limpia del proyecto:** Logramos organizar con éxito el código base, pasando de pruebas en celdas de Jupyter Notebooks a scripts independientes de Python (`.py`). Ahora, el proyecto está estructurado de forma ordenada, permitiendo entrenar, evaluar y guardar los modelos desde un flujo centralizado sin tener que modificar el código manualmente.

* **Aplicación práctica para el hotel (Streamlit):** Gracias a la interfaz creada en Streamlit, el proyecto no se queda solo en métricas matemáticas o códigos difíciles de entender. La aplicación web convierte las predicciones del modelo en sugerencias prácticas para el personal del hotel (como "Exigir depósito" o "Pedir garantía de tarjeta"), facilitando la toma de decisiones para proteger los ingresos del negocio.

* **Resultados confiables y reproducibles:** Al fijar los estados de aleatoriedad (`random_state=42`) y guardar correctamente los objetos entrenados (como el escalador `fitted_scaler.pkl` y el mejor modelo `champion_model.pkl`), aseguramos que el sistema funcione exactamente igual cada vez que se ejecute. Esto garantiza la estabilidad y confiabilidad de nuestro pipeline de Machine Learning.
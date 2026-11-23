# Palmer-Archipelago-Penguin

Dataset utilizado: [https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data?select=penguins_size.csv](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data?select=penguins_size.csv)  
  
Con el dataset de pingüinos, dependiendo de las diferentes propiedades del pico, sexo y masa corporal, identificar la especie entre 3 tipos(Adelie, Chinstrap y Gentoo).

---

## Guion para explicar el código (~5 minutos)

**0:00 – 0:30 · Qué hace el proyecto (2 ficheros)**

> El proyecto tiene dos partes: `app.py`, que es el programa que se ejecuta en directo y predice la especie de un pingüino que introduces por teclado, y `generar_informe.py`, que hace exactamente lo mismo pero además calcula los resultados y monta el informe con los tres informes de clasificación.

**0:30 – 1:30 · Carga y limpieza (apartado a)**

> Empezamos leyendo el CSV real de Kaggle. Fijaos en estas dos líneas: `df['sex'] = df['sex'].replace('.', pd.NA)` porque el propio dataset trae un punto en el campo de sexo, y después `dropna()` para quitar las 11 filas incompletas. Nos quedamos en 333 registros. Después hacemos one-hot encoding de `island` y `sex` con `pd.get_dummies`: las categorías se convierten en columnas 0/1, porque los modelos necesitan números. La variable `species` es nuestro `y` y tiene 3 clases, por eso es una clasificación multiclase.

**1:30 – 2:15 · Split y escalado (apartados b y c)**

> Aquí está el punto más importante del código:
>
> ```python
> X_train, X_test, y_train, y_test = train_test_split(
>     X, y, test_size=0.3, random_state=42, stratify=y)
> ```
>
> `test_size=0.3` deja el 30% para pruebas, `stratify=y` mantiene la proporción de las tres especies en los dos conjuntos, y `random_state=42` fija la semilla para que sea reproducible. Y luego el escalador: **ojo, `fit_transform` solo con train y al test solo `transform`**. Si escaláramos con todo el dataset estaríamos filtrando información del test al entrenamiento, eso se llama data leakage. Hace falta sobre todo para KNN y la regresión logística, porque trabajan con distancias y con coeficientes: sin escalar, `body_mass_g` (miles de gramos) aplastaría a las medidas del pico (decenas de milímetros). El árbol no lo necesita, pero usa la misma matriz.

**2:15 – 3:30 · Entrenamiento y evaluación (apartado d)**

> Luego se crean los tres modelos y se entrena a los tres con los mismos datos de entrenamiento:
>
> ```python
> modelos = {
>     "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
>     "Árbol de Decisión": DecisionTreeClassifier(max_depth=4, random_state=42),
>     "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
> }
> ```
>
> La idea de cada uno en una frase: **KNN** vota a los 5 vecinos más próximos en el espacio escalado; el **árbol** parte el espacio con reglas de tipo `variable ≤ umbral` (criterio Gini, `max_depth=4` para que no se memorice); y la **logística** dibuja una frontera lineal, un hiperplano entre cada par de clases. Se evalúan con `predict` sobre el test y se imprime `classification_report` de cada uno: accuracy, precision, recall y f1 por especie. Resultado: KNN 1.000, logística 0.990, árbol 0.950.

**3:30 – 4:30 · El elemento único donde disienten (apartado e, lo clave)**

> Esta función es el corazón del trabajo: recorre el test y se queda con el primer pingüino donde los tres modelos **no** coinciden. En nuestro caso, un Adelie de la isla Dream con aleta de 185 mm:
>
> - **KNN** → Adelie (probabilidad 1.0)
> - **Logística** → Adelie (0.95)
> - **Árbol** → **Chinstrap** ← disiente
>
> ¿Por qué? Porque cada modelo representa la frontera de forma distinta. El árbol no mira distancias: recorre estas reglas —`flipper ≤ 207.5`, `culmen_length ≤ 42.35`, `culmen_depth ≤ 16.65`— y la última es `island_Dream > 0.50`, y ese subconjunto quedó etiquetado como Chinstrap. Se equivoca por sus "cajas" rectangulares, no porque sus vecinos estén mal.

**4:30 – 5:00 · Cierre**

> En resumen: el mismo pingüino recibe tres explicaciones distintas porque los tres modelos ven la realidad de forma distinta —distancias locales, frontera lineal y reglas rectangulares—. Por eso no nos fijamos solo en el accuracy, sino en el informe completo. Para ejecutarlo en directo: `python app.py`, te pide la isla, las medidas y el sexo, y os saca la predicción de los tres modelos con sus probabilidades.
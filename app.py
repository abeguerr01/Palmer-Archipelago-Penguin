"""
Clasificación multiclase de pingüinos del Archipiélago Palmer (Antártida)
=========================================================================

Objetivo: a partir de las medidas corporales (pico, aleta, masa) y de
variables categóricas (isla, sexo), predecir la ESPECIE del pinguino
(Adelie, Chinstrap o Gentoo): problema de clasificación MULTICLASE.

Apartados del ejercicio:
  a) Datos reales -> penguins_size.csv (Kaggle: Palmer Archipelago Penguins)
  b) Separación entrenamiento / pruebas
  c) Estandarización con un escalador (StandardScaler)
  d) Entrenamiento y classification_report de 3 modelos:
        - K-Nearest Neighbors (KNN)
        - Árbol de Decisión
        - Regresión Logística
     (se guarda además la gráfica del árbol en arbol_decision.png)
  e) Predicción de un único elemento: se localiza una observación en la que
     los modelos DISIEREN y se razona el porqué.
  f) Conclusiones (para exponer en clase).

Ejecutar:  python app.py
"""

import pandas as pd
import matplotlib

# "Agg" guarda la figura en PNG sin necesidad de pantalla (entornos sin
# display / en servidores). Si quieres abrir la ventana interactiva,
# cambia a "TkAgg" y ejecuta plt.show().
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

SEED = 42  # semilla fija -> resultados reproducibles


# ---------------------------------------------------------------------------
# a) Búsqueda de datos reales (clasificación multiclase)
# ---------------------------------------------------------------------------
def cargar_datos(ruta: str = "penguins_size.csv") -> pd.DataFrame:
    """Carga el CSV y lo limpia (valores ausentes y sexos no válidos)."""
    df = pd.read_csv(ruta)

    # En el dataset aparecen valores de sexo como '.' y NaN: los tratamos como ausentes
    df["sex"] = df["sex"].replace(".", pd.NA)

    antes = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"[a] Dataset cargado: {len(df)} registros válidos de {antes} "
          f"(se eliminaron {antes - len(df)} con datos ausentes).")
    print(f"[a] Clases objetivo (multiclase): {df['species'].unique().tolist()}")
    print(df["species"].value_counts().to_string(), "\n")
    return df


# ---------------------------------------------------------------------------
# Preparación: variables predictoras (X) y objetivo (y) + codificación
# ---------------------------------------------------------------------------
def preparar(df: pd.DataFrame):
    # Las variables numéricas se estandarizan; las categóricas se codifican
    # con one-hot encoding (isla y sexo no son ordinales).
    X = pd.get_dummies(
        df.drop(columns="species"),
        columns=["island", "sex"],
        dtype=int,
    )
    y = df["species"]
    return X, y


# ---------------------------------------------------------------------------
# b) Separación de la muestra  +  c) Estandarización
# ---------------------------------------------------------------------------
def separar_y_estandarizar(X, y):
    # Estratificación: mantiene la proporción de cada especie en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    print(f"[b] Muestra de entrenamiento: {len(X_train)} registros | "
          f"de pruebas: {len(X_test)} registros")

    # Escalador ajustado SOLO con el conjunto de entrenamiento (evita data leakage)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print("[c] Datos estandarizados con StandardScaler (media=0, desviación=1)\n")
    return X_train_s, X_test_s, y_train, y_test, scaler


# ---------------------------------------------------------------------------
# d) Entrenamiento y classification_report de los 3 modelos
# ---------------------------------------------------------------------------
def entrenar_y_evaluar(X_train, X_test, y_train, y_test):
    modelos = {
        # Basado en distancias: vecinos más cercanos en el espacio estandarizado
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        # Reglas de decisión: divisiones recursivas sobre las variables
        # (criterio Gini, profundidad limitada para evitar sobreajuste)
        "Árbol de Decisión": DecisionTreeClassifier(
            max_depth=4, random_state=SEED
        ),
        # Frontera lineal entre clases (multinomial) con regularización L2
        "Regresión Logística": LogisticRegression(
            max_iter=1000, random_state=SEED
        ),
    }

    entrenados = {}
    print("=" * 70)
    print("d) INFORMES DE CLASIFICACIÓN (sobre el conjunto de PRUEBAS)")
    print("=" * 70)
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        entrenados[nombre] = modelo

        print(f"\n--- {nombre} ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    return entrenados


# ---------------------------------------------------------------------------
# Gráfica del árbol de decisión con matplotlib
# ---------------------------------------------------------------------------
def graficar_arbol(modelo, feature_names, clases, ruta: str = "arbol_decision.png"):
    """Dibuja el árbol entrenado y lo guarda en un PNG."""
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        modelo,
        feature_names=list(feature_names),
        class_names=list(clases),
        filled=True,        # nodos coloreados según la clase mayoritaria
        rounded=True,       # nodos con esquinas redondeadas
        fontsize=9,
        impurity=True,      # muestra el criterio Gini de cada nodo
        proportion=False,   # muestra los nº de muestras reales
        ax=ax,
    )
    ax.set_title(
        "Árbol de decisión - Clasificación de especies de pingüino\n"
        "(Palmer Archipelago, Antarctica)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"[d] Gráfica del árbol de decisión guardada en '{ruta}'")
    return ruta


# ---------------------------------------------------------------------------
# e) Prueba de clasificación con un único elemento
#    -> Buscamos una observación real en la que los 3 modelos NO coincidan.
# ---------------------------------------------------------------------------
def predecir_un_elemento(X_test, y_test, scaler, modelos,
                         X_encoded: pd.DataFrame, df_crudo: pd.DataFrame):
    print("=" * 70)
    print("e) PRUEBA DE CLASIFICACIÓN CON UN ÚNICO ELEMENTO")
    print("=" * 70)

    # 1) Localizamos un punto (empezando por el conjunto de pruebas) donde
    #    al menos un modelo disienta del resto.
    candidato_idx, predicciones = None, None
    for i in range(len(X_test)):
        preds = {n: m.predict(X_test[i:i + 1])[0] for n, m in modelos.items()}
        if len(set(preds.values())) > 1:
            candidato_idx, predicciones = i, preds
            break

    # Si en el conjunto de pruebas no hubiera discordancia, miramos todo el dataset
    if candidato_idx is None:
        X_todo_s = scaler.transform(X_encoded)
        for i in range(len(X_todo_s)):
            preds = {n: m.predict(X_todo_s[i:i + 1])[0] for n, m in modelos.items()}
            if len(set(preds.values())) > 1:
                candidato_idx, predicciones = i, preds
                X_test = X_todo_s
                y_test = None
                break

    if candidato_idx is None:
        print("Los tres modelos coinciden en todas las observaciones.\n")
        return

    x = X_test[candidato_idx:candidato_idx + 1]
    if y_test is not None:
        # El conjunto de pruebas conserva el índice original del DataFrame
        real = y_test.iloc[candidato_idx]
        fila = df_crudo.iloc[y_test.index[candidato_idx]]
    else:
        real = None
        fila = df_crudo.iloc[candidato_idx]
    medidas = {
        k: (v.item() if hasattr(v, "item") else v) for k, v in fila.to_dict().items()
    }

    print("\nObservación elegida (medidas originales, sin estandarizar):")
    print(f"  {medidas}")
    print(f"  Especie REAL: {real if real is not None else '¿desconocida?'}\n")
    print("Predicciones de cada modelo:")
    for nombre, pred in predicciones.items():
        modelo = modelos[nombre]
        probas = {k: round(float(v), 3)
                  for k, v in zip(modelo.classes_, modelo.predict_proba(x)[0])}
        marca = " <-- DISIENTE" if len(set(predicciones.values())) > 1 and \
            list(predicciones.values()).count(pred) == 1 else ""
        print(f"  {nombre:22s} -> {pred:10s} (probabilidades: {probas}){marca}")

    print(
        """
Por qué ocurren predicciones distintas en el mismo punto
--------------------------------------------------------
* KNN: clasifica por VOTOS de los k vecinos más próximos en el espacio
  estandarizado. Si el punto cae en una zona fronteriza donde conviven
  vecinos de dos especies, el voto puede inclinarse hacia la especie con
  más densidad local (no mira la frontera global, mira sólo el entorno).

* Árbol de decisión: divide el espacio con rectas PARALELAS a los ejes
  (si culmen_length > X y body_mass <= Y ...). Es un modelo discreto y
  codicioso: una observación que cae en una "caja" creada para separar
  una especie puede quedar mal clasificada aunque esté visualmente cerca
  de otra, porque el árbol ignora la distancia real a los centros de clase.

* Regresión logística: dibuja una FRONTERA LINEAL (hiperplano) entre cada
  par de clases. Es el modelo más "suave": asigna probabilidades basadas
  en la distancia al hiperplano, por lo que en puntos muy próximos a la
  frontera puede elegir la clase con menor margen si la señal de una
  variable lineal lo justifica.

Conclusión: los tres modelos representan la realidad de forma distinta
(distancias locales / reglas rectangulares / frontera lineal), por eso
un mismo pinguino borderline puede recibir tres etiquetas diferentes.
"""
    )


# ---------------------------------------------------------------------------
# f) Conclusiones
# ---------------------------------------------------------------------------
def conclusiones(resultados: dict):
    print("=" * 70)
    print("f) CONCLUSIONES")
    print("=" * 70)
    orden = sorted(resultados.items(), key=lambda kv: kv[1], reverse=True)
    for nombre, acc in orden:
        print(f"  - {nombre:22s} accuracy = {acc:.3f}")
    print(
        """
  1. Los tres modelos obtienen precisiones distintas: cada uno representa
     la frontera entre especies de forma diferente (distancias, reglas y
     hiperplanos), por lo que cometen errores en puntos distintos.
  2. La estandarización es imprescindible para KNN y para la regresión
     logística, ya que ambos trabajan con distancias / tamaños de
     coeficientes; si no se escalasen, las variables con mayor rango
     (body_mass_g) dominarían el resultado frente a las de menor rango.
  3. El árbol de decisión no necesita escalado, pero es muy sensible a su
     profundidad: con poca profundidad simplifica demasiado (underfitting)
     y con mucha, memoriza el entrenamiento (overfitting).
  4. Un único pinguino puede ser clasificado de forma distinta por los
     modelos: en la frontera entre especies (pingüinos inmaduros o de
     medidas intermedias) las estrategias de cada algoritmo divergen.
  5. Para elegir el mejor modelo no basta con el accuracy global: conviene
     mirar precision, recall y f1-score por clase (classification_report),
     ya que la especie minoritaria (Chinstrap) es la más difícil de acertar.
"""
    )


def main():
    df = cargar_datos()
    X, y = preparar(df)
    X_train_s, X_test_s, y_train, y_test, scaler = separar_y_estandarizar(X, y)
    modelos = entrenar_y_evaluar(X_train_s, X_test_s, y_train, y_test)

    # Gráfica del árbol de decisión (matplotlib -> arbol_decision.png)
    graficar_arbol(
        modelos["Árbol de Decisión"],
        feature_names=X.columns,
        clases=sorted(y.unique()),
    )

    # Accuracies para las conclusiones
    resultados = {
        n: accuracy_score(y_test, m.predict(X_test_s)) for n, m in modelos.items()
    }

    # Apartado e): un único elemento
    # (pasamos el DataFrame sin codificar para mostrar las medidas originales)
    predecir_un_elemento(X_test_s, y_test, scaler, modelos, X,
                         df.drop(columns="species"))

    conclusiones(resultados)


if __name__ == "__main__":
    main()

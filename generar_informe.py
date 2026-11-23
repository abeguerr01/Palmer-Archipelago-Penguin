"""
Genera el documento de entrega (DOCX) de la práctica "Modelar el mundo real"
a partir de la plantilla que hay en el escritorio.

El documento incluye:
  1. Portada (la de la plantilla: título, fecha y autores)
  2. Introducción
  3. Apartados a) a f) del enunciado, con:
       - tablas de datos y de resultados
       - los classification_report literales de los 3 modelos
       - el elemento único donde los modelos DISIENTEN + porqué
       - la imagen del árbol de decisión (arbol_decision.png)
  4. Guion de exposición en clase (5-10 minutos)
  5. Anexo con el código y cómo ejecutarlo

Uso:
    python generar_informe.py

La plantilla original se respeta: se copia antes a
    ~/Escritorio/Modelar_el_mundo_real_plantilla.docx
y el resultado se guarda encima de la plantilla (y en PDF con LibreOffice).
"""

import shutil
from pathlib import Path

import pandas as pd
import matplotlib

# Backend sin pantalla: guarda el PNG en cualquier entorno
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Inches, Pt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

SEED = 42

RAIZ = Path(__file__).resolve().parent
PLANTILLA = Path.home() / "Escritorio" / "Modelar_el_mundo_real.docx"
COPIA_PLANTILLA = Path.home() / "Escritorio" / "Modelar_el_mundo_real_plantilla.docx"
IMAGEN_ARBOL = RAIZ / "arbol_decision.png"

URL_DATOS = ("https://www.kaggle.com/datasets/parulpandey/"
             "palmer-archipelago-antarctica-penguin-data?select=penguins_size.csv")


# ===========================================================================
# 1. PIPELINE DE PRÁCTICA (los mismos pasos que en app.py)
# ===========================================================================
def preparar_datos():
    """Apartado a): carga, limpieza y codificación de los datos reales."""
    df = pd.read_csv(RAIZ / "penguins_size.csv")
    df["sex"] = df["sex"].replace(".", pd.NA)
    total = len(df)
    df = df.dropna().reset_index(drop=True)

    X = pd.get_dummies(df.drop(columns="species"),
                       columns=["island", "sex"], dtype=int)
    y = df["species"]
    return df, X, y, total


def dividir_y_escalar(X, y):
    """Apartados b) y c): split estratificado + estandarización."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()          # media 0, desviación típica 1
    X_train_s = scaler.fit_transform(X_train)   # ajusta SOLO con train
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler


def entrenar_modelos(X_train_s, y_train_s):
    """Apartado d): los 3 modelos vistos en clase."""
    modelos = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=4,
                                                    random_state=SEED),
        "Regresión Logística": LogisticRegression(max_iter=1000,
                                                  random_state=SEED),
    }
    for m in modelos.values():
        m.fit(X_train_s, y_train_s)
    return modelos


def elemento_discordante(modelos, X_test_s, y_test, X, df):
    """Apartado e): primer elemento del test donde los modelos DISIENTEN."""
    for i in range(len(X_test_s)):
        preds = {n: m.predict(X_test_s[i:i + 1])[0]
                 for n, m in modelos.items()}
        if len(set(preds.values())) > 1:
            idx_real = y_test.index[i]
            return {
                "i": i,
                "medidas": df.loc[idx_real].to_dict(),
                "real": y_test.iloc[i],
                "preds": preds,
                "probas": {
                    n: {c: round(float(p), 3)
                        for c, p in zip(m.classes_,
                                        m.predict_proba(X_test_s[i:i + 1])[0])}
                    for n, m in modelos.items()
                },
                "x": X_test_s[i:i + 1],           # escalado (para predecir)
                "x_orig": X.loc[idx_real].values,  # unidades originales (mm, g)
            }
    return None


def reglas_del_arbol(arbol, x_scal, x_orig, columnas, scaler):
    """Devuelve las reglas que sigue el árbol hasta la hoja, en unidades
    originales (mm y g), para poder explicar la predicción."""
    tree, node, reglas = arbol.tree_, 0, []
    while tree.children_left[node] != -1:
        f = tree.feature[node]
        umbral_orig = tree.threshold[node] * scaler.scale_[f] + scaler.mean_[f]
        valor = x_orig[f]                      # valor REAL del pinguino
        cumple = x_scal[0][f] <= tree.threshold[node]
        reglas.append(
            f"{columnas[f]} = {valor:g} "
            f"{'<=' if cumple else '>'} {umbral_orig:.2f}"
            f"   -> {'SÍ' if cumple else 'NO'}"
        )
        node = tree.children_left[node] if cumple else tree.children_right[node]
    hoja = arbol.classes_[tree.value[node].argmax()]
    return reglas, hoja


# ===========================================================================
# 2. GRÁFICA DEL ÁRBOL (matplotlib -> arbol_decision.png)
# ===========================================================================
def graficar_arbol(modelo, columnas, clases, ruta=IMAGEN_ARBOL):
    """Dibuja el árbol de decisión y lo guarda en un PNG.

    Se ejecuta en CADA generación del informe, así que la imagen se
    regenera siempre (se sobrescribe la anterior).
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        modelo,
        feature_names=list(columnas),
        class_names=list(clases),
        filled=True,     # nodos coloreados según la clase mayoritaria
        rounded=True,    # nodos con esquinas redondeadas
        fontsize=9,
        impurity=True,   # criterio Gini de cada nodo
        ax=ax,
    )
    ax.set_title(
        "Árbol de decisión - Clasificación de especies de pingüino\n"
        "(Palmer Archipelago, Antarctica)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(str(ruta), dpi=150)   # se sobrescribe en cada ejecución
    plt.close(fig)
    print(f"Gráfica del árbol generada: {ruta}")


# ===========================================================================
# 3. UTILIDADES PARA ESCRIBIR EL DOCUMENTO
# ===========================================================================
def titulo(doc, texto, nivel=1):
    doc.add_heading(texto, level=nivel)


def parrafo(doc, texto, negrita_inicial=None):
    p = doc.add_paragraph()
    if negrita_inicial:
        p.add_run(negrita_inicial).bold = True
    p.add_run(texto)
    return p


def estilo_disponible(doc, nombre):
    """Devuelve el nombre del estilo si existe en la plantilla, si no None."""
    try:
        doc.styles[nombre]
        return nombre
    except KeyError:
        return None


def viñetas(doc, items):
    # La plantilla no trae el estilo "List Bullet": si no existe, usamos
    # viñeta manual para que el documento sea robusto con cualquier plantilla.
    estilo = estilo_disponible(doc, "List Bullet")

    for it in items:
        p = doc.add_paragraph(style=estilo) if estilo else doc.add_paragraph()
        if not estilo:
            p.paragraph_format.left_indent = Inches(0.25)
            p.add_run("•  ")
        if isinstance(it, tuple):
            p.add_run(it[0]).bold = True
            p.add_run(it[1])
        else:
            p.add_run(it)


def tabla(doc, cabeceras, filas, ancho_primera_col=None):
    t = doc.add_table(rows=1, cols=len(cabeceras))
    t.style = "Table Grid"
    for c, h in zip(t.rows[0].cells, cabeceras):
        c.text = str(h)
        for p in c.paragraphs:
            for r in p.runs:
                r.font.bold = True
    for fila in filas:
        cells = t.add_row().cells
        for c, v in zip(cells, fila):
            c.text = str(v)
    doc.add_paragraph()  # separación
    return t


def bloque_codigo(doc, texto, tamano=8.5):
    """Texto monoespaciado tal y como sale por consola."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(texto)
    run.font.name = "Courier New"
    run.font.size = Pt(tamano)
    return p


def salto_pagina(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def figura(doc, ruta_imagen, pie, ancho_max=6.2):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(ruta_imagen), width=Inches(ancho_max))
    caption = doc.add_paragraph(pie, style="Caption")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER


# ===========================================================================
# 4. CONTENIDO DEL INFORME
# ===========================================================================
def construir_documento(resultados):
    """resultados: dict con todos los datos calculados."""
    # --- Respetamos la plantilla: copia de seguridad la primera vez y
    #     SIEMPRE partimos de la copia limpia para que regenerar el
    #     informe no duplique el contenido ---
    if PLANTILLA.exists() and not COPIA_PLANTILLA.exists():
        shutil.copy(PLANTILLA, COPIA_PLANTILLA)
        print(f"Plantilla original copiada a: {COPIA_PLANTILLA}")
    base = COPIA_PLANTILLA if COPIA_PLANTILLA.exists() else PLANTILLA

    doc = Document(str(base))

    # Portada de la plantilla + salto de página
    salto_pagina(doc)

    # ---------------------------------------------------------------
    titulo(doc, "1. Introducción", 1)
    # ---------------------------------------------------------------
    parrafo(
        doc,
        "En este trabajo modelamos una situación del mundo real: clasificar "
        "pingüinos del Archipiélago Palmer (Antártida) en una de sus tres "
        "especies —Adelie, Chinstrap y Gentoo— a partir de medidas "
        "corporales reales. A diferencia de ejemplos de manual como el "
        "dataset de Iris, los datos proceden de una expedición científica "
        "publicada en Kaggle, así que contienen valores ausentes, clases "
        "desbalanceadas y casos frontera que obligan a investigar por qué "
        "los modelos no siempre coinciden.",
    )
    parrafo(
        doc,
        "El objetivo final no es sólo obtener un porcentaje de acierto alto, "
        "sino entender por qué los tres modelos estudiados (KNN, árbol de "
        "decisión y regresión logística) toman decisiones distintas ante el "
        "mismo individuo.",
    )

    # ---------------------------------------------------------------
    titulo(doc, "a) Búsqueda de datos reales (clasificación multiclase)", 1)
    # ---------------------------------------------------------------
    parrafo(
        doc,
        f"Fuente: fichero penguins_size.csv del dataset “Palmer "
        f"Archipelago Antarctica Penguin Data” de Kaggle: {URL_DATOS}",
    )
    parrafo(
        doc,
        f"El fichero original contiene {resultados['total']} registros y 7 "
        f"columnas. Tras eliminar las filas con valores ausentes quedan "
        f"{resultados['n_valid']} registros válidos. Es un problema de "
        f"clasificación MULTICLASE porque la variable objetivo tiene 3 "
        f"clases:",
    )
    tabla(doc,
          ["Especies (objetivo)", "Registros", "%"],
          resultados["distribucion"])
    parrafo(doc, "Variables del dataset:", negrita_inicial="")
    tabla(doc,
          ["Variable", "Tipo", "Descripción"],
          [
              ["species", "Categórica (objetivo)", "Especie: Adelie, Chinstrap o Gentoo"],
              ["island", "Categórica", "Isla: Torgersen, Biscoe o Dream"],
              ["culmen_length_mm", "Numérica", "Largo del pico (mm)"],
              ["culmen_depth_mm", "Numérica", "Grosor del pico (mm)"],
              ["flipper_length_mm", "Numérica", "Largo de la aleta (mm)"],
              ["body_mass_g", "Numérica", "Masa corporal (g)"],
              ["sex", "Categórica", "Sexo: MALE o FEMALE"],
          ])
    parrafo(
        doc,
        "Limpieza y codificación: se sustituye el valor “.” de sex por "
        "ausente y se eliminan las 11 filas con datos incompletos. Las dos "
        "variables categóricas predictoras (island y sex) se codifican con "
        "one-hot encoding (pd.get_dummies), obteniendo 9 columnas "
        "predictoras: 4 numéricas + 5 binarias.",
    )

    # ---------------------------------------------------------------
    titulo(doc, "b) Separación de la muestra para entrenamiento y pruebas", 1)
    # ---------------------------------------------------------------
    bloque_codigo(
        doc,
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.3, random_state=42, stratify=y\n"
        ")",
    )
    viñetas(doc, [
        ("70 % entrenamiento / 30 % pruebas: ",
         f"{resultados['n_train']} registros para ajustar los modelos y "
         f"{resultados['n_test']} reservados para evaluarlos con datos que "
         "nunca han visto."),
        ("stratify=y: ",
         "mantiene la proporción de cada especie en ambos conjuntos; sin "
         "ello, la clase minoritaria (Chinstrap) podría quedar "
         "infrarrepresentada en el test y engañar al informe."),
        ("random_state=42: ",
         "fija la semilla para que la división (y todo el experimento) sea "
         "reproducible en cualquier máquina."),
        ("Por qué separamos antes de escalar: ",
         "los parámetros del escalador se calculan con el entrenamiento "
         "para no “ver” el test."),
    ])
    parrafo(
        doc,
        f"Reparto resultante en el conjunto de pruebas: "
        f"{resultados['dist_test']}.",
    )

    # ---------------------------------------------------------------
    titulo(doc, "c) Estandarización de los datos (con un escalador)", 1)
    # ---------------------------------------------------------------
    bloque_codigo(
        doc,
        "scaler = StandardScaler()\n"
        "X_train_s = scaler.fit_transform(X_train)   # ajusta con TRAIN\n"
        "X_test_s  = scaler.transform(X_test)        # sólo transforma",
    )
    parrafo(
        doc,
        "StandardScaler aplica x' = (x − μ) / σ, de modo que cada variable "
        "predictora queda con media 0 y desviación típica 1. Es necesario "
        "porque:",
    )
    viñetas(doc, [
        ("KNN: ",
         "usa distancia euclídea; sin escalar, body_mass_g (2.700–6.300) "
         "dominaría flipper_length_mm (172–231) y las medidas del pico "
         "prácticamente no contarían."),
        ("Regresión logística: ",
         "la escala de las variables afecta a la convergencia del "
         "gradiente y al tamaño de los coeficientes; estandarizar la "
         "regularización L2 es comparable entre variables."),
        ("Árbol de decisión: ",
         "no lo necesita porque compara cada variable con su propio "
         "umbral, pero estandarizar no le perjudica y nos permite usar la "
         "misma matriz para los tres modelos."),
    ])
    parrafo(
        doc,
        "Ojo con el data leakage: el escalador se ajusta sólo con "
        "X_train (fit_transform) y al test sólo se le aplica transform. "
        "Si se ajustara con todo el conjunto, estaríamos usando "
        "información del test durante el entrenamiento.",
    )

    # ---------------------------------------------------------------
    titulo(doc, "d) Entrenamiento y pruebas de los 3 modelos", 1)
    # ---------------------------------------------------------------
    tabla(doc,
          ["Modelo", "Idea del algoritmo", "Parámetros"],
          [
              ["KNN (k=5)",
               "Clasifica por votación de los 5 vecinos más próximos en el "
               "espacio estandarizado",
               "n_neighbors=5"],
              ["Árbol de Decisión",
               "Divide el espacio con reglas “variable ≤ umbral” "
               "(criterio Gini), interpretable",
               "max_depth=4, random_state=42"],
              ["Regresión Logística",
               "Frontera lineal (hiperplano) entre cada par de clases, con "
               "probabilidades",
               "max_iter=1000 (multinomial, L2)"],
          ])
    parrafo(doc, "Resumen de resultados sobre el conjunto de pruebas:")
    tabla(doc,
          ["Modelo", "Accuracy", "Comentario"],
          resultados["resumen"])

    for nombre in resultados["modelos"]:
        titulo(doc, f"Informe de clasificación: {nombre}", 2)
        bloque_codigo(doc, resultados["informes_texto"][nombre])
        parrafo(
            doc,
            f"Lectura rápida: accuracy = {resultados['accuracies'][nombre]:.3f}. "
            f"Fíjate en precision (¿cuántos de los predichos como X son "
            f"realmente X?), recall (¿cuántos de los reales X se han "
            f"acertado?) y f1 (media armónica de ambos) de cada especie.",
        )

    # ---------------------------------------------------------------
    titulo(doc, "e) Prueba de clasificación con un único elemento", 1)
    # ---------------------------------------------------------------
    disc = resultados["discordante"]
    if disc is None:
        parrafo(doc, "Los tres modelos coinciden en todas las observaciones.")
    else:
        m = disc["medidas"]
        parrafo(
            doc,
            "Elegimos una observación real del conjunto de pruebas en la "
            "que los tres modelos NO coinciden:",
        )
        tabla(doc,
              ["Dato", "Valor"],
              [
                  ["Isla", m["island"]],
                  ["Largo del pico (culmen_length_mm)", f"{m['culmen_length_mm']} mm"],
                  ["Grosor del pico (culmen_depth_mm)", f"{m['culmen_depth_mm']} mm"],
                  ["Largo de la aleta (flipper_length_mm)", f"{m['flipper_length_mm']} mm"],
                  ["Masa corporal (body_mass_g)", f"{m['body_mass_g']:.0f} g"],
                  ["Sexo", m["sex"]],
                  ["ESPECIE REAL", disc["real"]],
              ])
        parrafo(doc, "Predicción de cada modelo (con sus probabilidades):")
        filas = []
        for nombre, pred in disc["preds"].items():
            probas = ", ".join(f"{c}: {p}" for c, p in disc["probas"][nombre].items())
            distintos = list(disc["preds"].values()).count(pred) == 1
            filas.append([nombre, pred, probas,
                          "DISIENTE" if distintos else "coincide"])
        tabla(doc, ["Modelo", "Predicción", "Probabilidades", "¿Coincide?"],
              filas)

        titulo(doc, "¿Por qué sucede esto?", 2)
        viñetas(doc, [
            ("KNN (predice Adelie): ",
             "vota entre los 5 vecinos más próximos del pinguino en el "
             "espacio estandarizado; en esa zona hay varios Adelie "
             "cercanos, así que gana la mayoría local."),
            ("Regresión logística (predice Adelie): ",
             "calcula la distancia al hiperplano que separa las clases; "
             "este punto cae del lado de Adelie con una probabilidad de "
             "0.95, muy por encima de Chinstrap (0.049)."),
            ("Árbol de decisión (predice Chinstrap): ",
             "NO mira distancias: recorre reglas rectangulares "
             "preestablecidas. Ésta es la ruta que sigue este pinguino "
             "(en unidades originales):"),
        ])
        bloque_codigo(doc, "\n".join(
            f"  {i+1}. {r}" for i, r in enumerate(disc["reglas"])
        ) + f"\n  -> Hoja alcanzada: {disc['hoja']} (especie real: {disc['real']})")
        parrafo(
            doc,
            f"Las tres primeras reglas son correctas (aleta corta, pico "
            f"corto y poco grueso: características de Adelie), pero la "
            f"última regla usa la isla como desempate: el individuo "
            f"pertenece a Dream (island_Dream = 1 > 0.50 = NO), y ese "
            f"subconjunto del árbol quedó etiquetado como Chinstrap. El "
            f"árbol se equivoca porque parte el espacio con rectas "
            f"paralelas a los ejes e ignora la distancia real a los "
            f"centros de clase: es un error de sus “cajas”, no de sus "
            f"vecinos.",
        )
        parrafo(
            doc,
            "En resumen: el mismo pinguino recibe tres explicaciones "
            "distintas porque los tres modelos representan la realidad de "
            "forma distinta (distancias locales / frontera lineal / reglas "
            "rectangulares). Ése es el objetivo de este apartado.",
        )
        if IMAGEN_ARBOL.exists():
            figura(doc, IMAGEN_ARBOL,
                   "Figura 1. Árbol de decisión entrenado. Los nodos "
                   "muestran el umbral, el Gini, las muestras y la clase "
                   "mayoritaria (naranja: Adelie, verde: Chinstrap, "
                   "violeta: Gentoo).")

    # ---------------------------------------------------------------
    titulo(doc, "f) Conclusiones", 1)
    # ---------------------------------------------------------------
    viñetas(doc, resultados["conclusiones"])

    # ---------------------------------------------------------------
    titulo(doc, "Guion de exposición en clase (5-10 minutos)", 1)
    # ---------------------------------------------------------------
    parrafo(
        doc,
        "Guion orientativo para la presentación; en el aula se puede "
        "proyectar la imagen del árbol y la salida por consola de app.py.",
    )
    tabla(doc,
          ["Tiempo", "Bloque", "Qué contar"],
          [
              ["0:00 – 1:00", "Introducción y datos",
               "Qué modelamos (especies de pingüinos de la Antártida), "
               "fuente real de Kaggle y por qué NO usamos Iris; 344 "
               "registros, 3 clases multiclase."],
              ["1:00 – 2:30", "Preparación",
               "Limpieza (344 → 333), one-hot de isla y sexo, reparto "
               "70/30 estratificado con semilla 42 y por qué hace falta "
               "el StandardScaler (KNN y logística)."],
              ["2:30 – 4:30", "Los 3 modelos",
               "Idea de cada algoritmo en una frase y sus classification_"
               "report: KNN 1.000, Logística 0.990, Árbol 0.950; "
               "comentar precision/recall de Chinstrap."],
              ["4:30 – 7:30", "El elemento único (clave)",
               "Mostrar el pinguino de prueba, la tabla con las 3 "
               "predicciones (el árbol disiente) y la figura del árbol: "
               "señalar la ruta flipper → culmen_length → culmen_depth → "
               "island_Dream, que lo manda a Chinstrap. Explicar por qué: "
               "cajas rectangulares vs. distancias vs. hiperplano."],
              ["7:30 – 9:00", "Conclusiones",
               "Las predicciones difieren porque cada modelo representa "
               "la frontera de forma distinta; elegir modelo según el "
               "informe completo y no sólo el accuracy; errores en la "
               "frontera entre especies."],
              ["9:00 – 10:00", "Cierre y dudas",
               "Enseñar cómo se ejecuta app.py (predicción interactiva) y "
               "responder preguntas."],
          ])

    # ---------------------------------------------------------------
    titulo(doc, "Anexo: código y ejecución", 1)
    # ---------------------------------------------------------------
    parrafo(doc, "Ficheros del proyecto:", negrita_inicial="")
    viñetas(doc, [
        ("app.py — ", "entrena los 3 modelos y predice la especie de un "
                      "pingüino introducido por consola (predictor "
                      "interactivo)."),
        ("generar_informe.py — ", "este documento: regenera la imagen del "
                                  "árbol (arbol_decision.png) y el informe "
                                  "DOCX con los resultados actualizados."),
        ("penguins_size.csv — ", "datos reales (Kaggle)."),
        ("arbol_decision.png — ", "gráfica del árbol de decisión: se "
                                  "regenera en cada ejecución de "
                                  "generar_informe.py."),
        ("requirements.txt — ", "pandas, scikit-learn, matplotlib y "
                                "python-docx."),
    ])
    parrafo(doc, "Ejecución:", negrita_inicial="")
    bloque_codigo(
        doc,
        "pip install -r requirements.txt\n"
        "python app.py               # predictor interactivo por consola\n"
        "python generar_informe.py   # regenera árbol PNG + informe DOCX",
    )
    parrafo(doc, "Núcleo del pipeline (idéntico en app.py y aquí):",
            negrita_inicial="")
    bloque_codigo(
        doc,
        "df = pd.read_csv('penguins_size.csv').dropna()\n"
        "X = pd.get_dummies(df.drop(columns='species'),\n"
        "                   columns=['island', 'sex'])\n"
        "y = df['species']                       # 3 clases: multiclase\n"
        "\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.3, random_state=42, stratify=y)\n"
        "\n"
        "scaler = StandardScaler()\n"
        "X_train_s = scaler.fit_transform(X_train)   # ajusta con TRAIN\n"
        "X_test_s = scaler.transform(X_test)        # sólo transforma\n"
        "\n"
        "for modelo in [KNeighborsClassifier(n_neighbors=5),\n"
        "               DecisionTreeClassifier(max_depth=4),\n"
        "               LogisticRegression(max_iter=1000)]:\n"
        "    modelo.fit(X_train_s, y_train)\n"
        "    print(classification_report(y_test,\n"
        "                                modelo.predict(X_test_s)))",
    )

    doc.save(str(PLANTILLA))
    print(f"Documento guardado en: {PLANTILLA}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    # --- a) datos reales ---
    df, X, y, total = preparar_datos()
    n_valid = len(df)
    dist = y.value_counts()
    distribucion = [
        [esp, int(n), f"{100*n/n_valid:.1f} %"] for esp, n in dist.items()
    ]

    # --- b) y c) split + escalado ---
    X_train_s, X_test_s, y_train, y_test, scaler = dividir_y_escalar(X, y)

    # --- d) modelos e informes ---
    modelos = entrenar_modelos(X_train_s, y_train)
    informes_texto, accuracies, resumen = {}, {}, []
    for nombre, m in modelos.items():
        y_pred = m.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        accuracies[nombre] = acc
        informes_texto[nombre] = classification_report(
            y_test, y_pred, digits=3, zero_division=0)
        resumen.append([nombre, f"{acc:.3f}", ""])

    # Comentario de la tabla resumen
    resumen[0][2] = "Mejor precisión: vecinos directos en el espacio escalado"
    resumen[1][2] = "Muy cerca del KNN, frontera lineal suave"
    resumen[2][2] = "Peor: sus “cajas” se equivocan en la frontera"

    # --- e) elemento discordante + ruta del árbol ---
    disc = elemento_discordante(modelos, X_test_s, y_test, X, df)
    if disc is not None:
        columnas = list(X.columns)
        reglas, hoja = reglas_del_arbol(modelos["Árbol de Decisión"],
                                        disc["x"], disc["x_orig"],
                                        columnas, scaler)
        disc["reglas"], disc["hoja"] = reglas, hoja

    # --- imagen del árbol (se regenera siempre) ---
    graficar_arbol(modelos["Árbol de Decisión"], X.columns,
                   sorted(y.unique()), ruta=IMAGEN_ARBOL)

    # --- conclusiones ---
    conclusiones = [
        ("Los tres modelos predicen distinto ",
         f"(KNN {accuracies['KNN (k=5)']:.3f}, regresión logística "
         f"{accuracies['Regresión Logística']:.3f}, árbol de decisión "
         f"{accuracies['Árbol de Decisión']:.3f}) porque representan la "
         "frontera entre especies de forma diferente: distancias locales, "
         "hiperplano lineal y reglas rectangulares. Por eso cometen "
         "errores en puntos distintos."),
        ("La estandarización es imprescindible ",
         "para KNN y regresión logística; sin ella, la variable "
         "body_mass_g (miles de gramos) pesaría muchísimo más que las "
         "medidas del pico (decenas de milímetros). El árbol no la "
         "necesita, porque compara cada variable con su propio umbral."),
        ("Un modelo interpretable puede ser peor: ",
         "el árbol (0.950) es el más fácil de explicar pero el que peor "
         "clasifica, porque sus divisiones estrictamente horizontales y "
         "verticales no capturan la frontera real entre Adelie y "
         "Chinstrap."),
        ("Hay que mirar el informe completo, ",
         "no sólo el accuracy: en el árbol la clase minoritaria "
         "(Chinstrap) es la que más arrastra el error, mientras que "
         "Gentoo se separa casi perfectamente por su tamaño."),
        ("En la frontera entre especies ",
         "—pingüinos jóvenes o con medidas intermedias— cualquier modelo "
         "puede fallar; el apartado e) demuestra que un mismo individuo "
         "puede recibir tres etiquetas distintas y que la explicación "
         "depende del algoritmo."),
        ("Aplicación real: ",
         "un sistema así podría apoyar a investigadores en la "
         "identificación automática de especies a partir de fotos o "
         "medidas de campo, pero conviene combinar modelos o revisar los "
         "casos donde discrepan antes de fiarse de uno solo."),
    ]

    construir_documento({
        "total": total,
        "n_valid": n_valid,
        "distribucion": distribucion,
        "n_train": len(X_train_s),
        "n_test": len(X_test_s),
        "dist_test": ", ".join(f"{k}: {v}" for k, v in
                               y_test.value_counts().items()),
        "modelos": list(modelos),
        "accuracies": accuracies,
        "informes_texto": informes_texto,
        "resumen": resumen,
        "discordante": disc,
        "conclusiones": conclusiones,
    })


if __name__ == "__main__":
    main()

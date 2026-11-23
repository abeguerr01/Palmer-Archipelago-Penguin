"""
Predice la especie de un pingüino (Adelie, Chinstrap o Gentoo) a partir de
sus medidas introducidas por consola.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

SEED = 42


def cargar_datos(ruta: str = "penguins_size.csv") -> pd.DataFrame:
    """Carga el CSV y elimina las filas con datos ausentes."""
    df = pd.read_csv(ruta)
    df["sex"] = df["sex"].replace(".", pd.NA)  # sexo desconocido en el CSV
    return df.dropna().reset_index(drop=True)


def guardar_arbol(
    modelo_arbol: DecisionTreeClassifier,
    columnas,
    clases,
    ruta_salida: str = "arbol_decision.png",
):
    """Guarda una imagen PNG con el árbol de decisión entrenado."""
    plt.figure(figsize=(20, 10))
    plot_tree(
        modelo_arbol,
        feature_names=list(columnas),
        class_names=list(clases),
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=300)
    plt.close()
    print(f"Árbol de decisión guardado en: {ruta_salida}")


def mostrar_comparativa(modelos: dict, X_train_s, y_train, X_test_s, y_test):
    """Muestra la precisión (accuracy) de cada modelo en train y en test."""
    print("\n--- Comparativa de precisión (accuracy) ---")
    print(f"{'Modelo':22s} | {'Train':>8s} | {'Test':>8s}")
    print("-" * 46)
    for nombre, modelo in modelos.items():
        acc_train = accuracy_score(y_train, modelo.predict(X_train_s))
        acc_test = accuracy_score(y_test, modelo.predict(X_test_s))
        print(f"{nombre:22s} | {acc_train:7.1%} | {acc_test:7.1%}")


def entrenar(df: pd.DataFrame):
    """Entrena los 3 modelos, muestra precisión, guarda el PNG y devuelve artefactos."""
    X = pd.get_dummies(
        df.drop(columns="species"),
        columns=["island", "sex"],
        dtype=int,
    )
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )

    # Escalador ajustado solo con el entrenamiento (imprescindible para KNN
    # y regresión logística, que trabajan con distancias / coeficientes)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    modelos = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=4, random_state=SEED),
        "Regresión Logística": LogisticRegression(max_iter=1000, random_state=SEED),
    }
    for modelo in modelos.values():
        modelo.fit(X_train_s, y_train)

    mostrar_comparativa(modelos, X_train_s, y_train, X_test_s, y_test)
    guardar_arbol(
        modelos["Árbol de Decisión"],
        X.columns,
        modelos["Árbol de Decisión"].classes_,
    )
    print()

    return modelos, scaler, X.columns


def pedir_texto(mensaje: str, opciones: list) -> str:
    while True:
        valor = input(f"{mensaje} ({'/'.join(opciones)}): ").strip().capitalize()
        if valor in opciones:
            return valor
        print(f"  Valor no válido. Elige uno de: {', '.join(opciones)}")


def pedir_numero(mensaje: str) -> float:
    while True:
        try:
            return float(input(f"{mensaje}: ").strip())
        except ValueError:
            print("  Eso no es un número, inténtalo de nuevo.")


def pedir_datos() -> pd.DataFrame:
    print("--- Datos del pingüino ---")
    datos = {
        "island": pedir_texto("Isla", ["Torgersen", "Biscoe", "Dream"]),
        "culmen_length_mm": pedir_numero("Largo del pico (mm)"),
        "culmen_depth_mm": pedir_numero("Grosor del pico (mm)"),
        "flipper_length_mm": pedir_numero("Largo de la aleta (mm)"),
        "body_mass_g": pedir_numero("Masa corporal (g)"),
        "sex": pedir_texto("Sexo", ["Male", "Female"]).upper(),
    }
    return pd.DataFrame([datos])


def predecir(modelos, scaler, columnas):
    fila = pedir_datos()

    # Misma codificación que en el entrenamiento, alineada por columnas
    fila = pd.get_dummies(fila, columns=["island", "sex"], dtype=int)
    fila = fila.reindex(columns=columnas, fill_value=0)
    fila_s = scaler.transform(fila)

    print("\n--- Resultado ---")
    votos = {}
    for nombre, modelo in modelos.items():
        especie = modelo.predict(fila_s)[0]
        probas = {
            clase: round(float(proba), 3)
            for clase, proba in zip(modelo.classes_, modelo.predict_proba(fila_s)[0])
        }
        votos[especie] = votos.get(especie, 0) + 1
        print(f"  {nombre:22s} -> {especie:10s} {probas}")

    # Especie más votada por los 3 modelos
    mas_probable = max(votos.items(), key=lambda kv: kv[1])[0]
    print(f"\n  El pingüino más probable es: {mas_probable}")


def main():
    df = cargar_datos()
    print(f"Dataset: {len(df)} registros | Especies: {df['species'].unique().tolist()}\n")
    modelos, scaler, columnas = entrenar(df)
    predecir(modelos, scaler, columnas)


if __name__ == "__main__":
    main()
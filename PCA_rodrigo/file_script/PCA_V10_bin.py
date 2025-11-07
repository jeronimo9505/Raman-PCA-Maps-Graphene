import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')  # Cambiar el backend
import matplotlib.pyplot as plt
import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Importar herramientas para gráficos 3D
from matplotlib.widgets import Button  # Importar botón interactivo
from matplotlib.widgets import TextBox
from sklearn.decomposition import PCA
import pandas as pd
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt





# Definir el rango de interés para Raman Shifts
raman_shift_range = [215, 247]  # Convertir en lista para permitir asignaciones
  # Modificable según necesidad

# Directorios de los grupos y sus etiquetas asociadas
group_directories = [
    "/Users/rodrigojeronimocruz/Desktop/pilar/r41_w",
    "/Users/rodrigojeronimocruz/Desktop/pilar/r41_g"
    
]

group_labels = [
    "Blanco",
    "R41"
]

# Definir colores para los grupos
color_order = [
    "red"
]






def load_group_data_with_matrices(directory, group_label, raman_shift_range):
    """
    Carga datos de un grupo desde archivos .bin, consolidando datos en un formato único.
    """
    print(f"Cargando datos del grupo '{group_label}' en la región {raman_shift_range}.")
    all_spectra = []
    valid_files = []
    raman_shifts = None

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        if file_name.startswith("orig") and file_name.endswith(".bin"):
            try:
                with open(file_path, "rb") as bin_file:
                    data = pickle.load(bin_file)

                file_raman_shifts = data.get("Raman Shifts", [])
                intensity_data = data.get("Intensity Data", {})

                if not file_raman_shifts or not intensity_data:
                    print(f"Datos incompletos en el archivo {file_path}.")
                    continue

                # Filtrar Raman Shifts
                filtered_indices = [
                    i for i, shift in enumerate(file_raman_shifts)
                    if raman_shift_range[0] <= shift <= raman_shift_range[1]
                ]

                if not filtered_indices:
                    print(f"No datos en rango en {file_path}.")
                    continue

                filtered_shifts = [file_raman_shifts[i] for i in filtered_indices]
                filtered_spectra = {
                    key: [value[i] for i in filtered_indices]
                    for key, value in intensity_data.items()
                }

                # Verificar Raman Shifts consistentes
                if raman_shifts is None:
                    raman_shifts = filtered_shifts
                elif raman_shifts != filtered_shifts:
                    raise ValueError(f"Raman Shifts inconsistentes en {file_path}.")

                # Crear DataFrame organizado
                for key, intensities in filtered_spectra.items():
                    matrix_name, point_name = key.split("_", 1)
                    all_spectra.append({
                        "Group": group_label,
                        "Subgroup": matrix_name,
                        "Point": point_name,
                        **dict(zip(raman_shifts, intensities))
                    })

                valid_files.append(file_path)

            except Exception as e:
                print(f"Error procesando archivo {file_path}: {e}")

    if not all_spectra:
        raise ValueError(f"No se encontraron datos válidos en: {directory}")

    # Consolidar en un único DataFrame
    consolidated_data = pd.DataFrame(all_spectra)
    print(f"Procesados {len(valid_files)} archivos del grupo '{group_label}'.")
    return consolidated_data, raman_shifts     


def apply_global_pca(data, n_components=2):
    """
    Aplica PCA global sobre los datos Raman y retorna las coordenadas proyectadas.
    
    Parámetros:
    - data: DataFrame que incluye columnas de etiquetas ('Group', 'Subgroup') 
            y columnas de Raman Shifts.
    - n_components: Número de componentes principales a calcular.
    
    Retorna:
    - pca_df: DataFrame con las coordenadas proyectadas y etiquetas originales.
    - explained_variance: Lista con la varianza explicada por cada componente principal.
    """
    # Extraer intensidades Raman (columnas numéricas)
    feature_columns = [col for col in data.columns if col not in ['Group', 'Subgroup', 'Point']]
    X = data[feature_columns].values

    # Aplicar PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Crear DataFrame con las proyecciones y etiquetas originales
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_df['Group'] = data['Group'].values
    pca_df['Subgroup'] = data['Subgroup'].values
    pca_df['Point'] = data['Point'].values

    # Obtener la varianza explicada por cada componente
    explained_variance = pca.explained_variance_ratio_

    return pca_df, explained_variance


def plot_pca_with_matplotlib(pca_df):
    """
    Visualiza los resultados del PCA usando Matplotlib, diferenciando grupos y subgrupos.
    
    Parámetros:
    - pca_df: DataFrame con columnas 'PC1', 'PC2', 'Group', y 'Subgroup'.
    """
    # Obtener los grupos y subgrupos únicos
    unique_groups = pca_df['Group'].unique()
    markers = ['o', 's', 'D', '^', 'v', 'P', '*']  # Diferentes marcadores para subgrupos

    plt.figure(figsize=(12, 8))

    for group in unique_groups:
        # Filtrar datos por grupo
        group_data = pca_df[pca_df['Group'] == group]

        # Obtener los subgrupos dentro de este grupo
        unique_subgroups = group_data['Subgroup'].unique()

        for i, subgroup in enumerate(unique_subgroups):
            # Filtrar datos por subgrupo
            subgroup_data = group_data[group_data['Subgroup'] == subgroup]

            # Asignar un marcador único por subgrupo
            marker = markers[i % len(markers)]

            # Graficar los datos del subgrupo
            plt.scatter(
                subgroup_data['PC1'],
                subgroup_data['PC2'],
                label=f"{group} - {subgroup}",
                marker=marker,
                alpha=0.8,
                s=100
            )

    # Configurar etiquetas y leyenda
    plt.title("PCA de Grupos y Subgrupos (Matplotlib)", fontsize=16)
    plt.xlabel("Componente Principal 1 (PC1)", fontsize=12)
    plt.ylabel("Componente Principal 2 (PC2)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title="Leyenda")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_pca_groups_and_subgroups(pca_df):
    """
    Analiza la varianza y dispersión por grupo y subgrupo en el espacio PCA.
    """
    results = {}
    groups = pca_df['Group'].unique()

    for group in groups:
        group_data = pca_df[pca_df['Group'] == group]
        print(f"\nGrupo: {group}")
        print(f"Varianza en PC1: {group_data['PC1'].var():.4f}")
        print(f"Varianza en PC2: {group_data['PC2'].var():.4f}")

        subgroups = group_data['Subgroup'].unique()
        results[group] = {}

        for subgroup in subgroups:
            subgroup_data = group_data[group_data['Subgroup'] == subgroup]
            print(f"  Subgrupo: {subgroup}")
            print(f"    Varianza en PC1: {subgroup_data['PC1'].var():.4f}")
            print(f"    Varianza en PC2: {subgroup_data['PC2'].var():.4f}")

            # Guardar resultados en un diccionario
            results[group][subgroup] = {
                'PC1_variance': subgroup_data['PC1'].var(),
                'PC2_variance': subgroup_data['PC2'].var()
            }
    return results

























    

if __name__ == "__main__":
    # Parámetros de ejecución
    manual_offset = 1.8  # Ajusta según sea necesario
    
    # Cargar y procesar los datos
    all_data = []
    for directory, label in zip(group_directories, group_labels):
        try:
            # Cargar datos del grupo
            data, raman_shifts = load_group_data_with_matrices(directory, label, raman_shift_range)
            all_data.append(data)
        except ValueError as e:
            print(f"Error al procesar el grupo '{label}': {e}")
    
    # Consolidar todos los datos en un único DataFrame
    if all_data:
        combined_data = pd.concat(all_data, axis=0)
        print("Todos los datos se han cargado y consolidado correctamente.")
        
        # Aplicar PCA
        print("Aplicando PCA...")
        pca_results, explained_variance = apply_global_pca(combined_data)
        
        # Mostrar varianza explicada
        print("Varianza explicada por los componentes principales:", explained_variance)
        
        # Visualizar PCA
        print("Generando visualización...")
        plot_pca_with_matplotlib(pca_results)
    else:
        print("No se cargaron datos válidos. Revisa los directorios y archivos.")

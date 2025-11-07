import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')  # Cambiar el backend
import matplotlib.pyplot as plt

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Importar herramientas para gráficos 3D
from matplotlib.widgets import Button  # Importar botón interactivo
from matplotlib.widgets import TextBox

# Definir el rango de interés para Raman Shifts
raman_shift_range = [588, 664]  # Convertir en lista para permitir asignaciones
  # Modificable según necesidad

# Directorios de los grupos y sus etiquetas asociadas
group_directories = [
    "C:\\Users\\Rodrigo\\Desktop\\Isabel\\Analizar_snv\\Blanco policarbonato\\Processed",
    "C:\\Users\\Rodrigo\\Desktop\\Isabel\\Analizar_snv\\15nmAu_35s\\15nmAu_19pm_35s\\Processed",
    "C:\\Users\\Rodrigo\\Desktop\\Isabel\\Analizar_snv\\15nmAu_35s\\15nmAu_110pm_35s\\Processed",
    "C:\\Users\\Rodrigo\\Desktop\\Isabel\\Analizar_snv\\15nmAu_35s\\15nmAu_1000pm_35s\\Processed",
    "C:\\Users\\Rodrigo\\Desktop\\Isabel\\Analizar_snv\\30nmAu_35s\\30nmAu_19pm_35\\Processed",
    "C:\\Users\\Rodrigo\\Desktop\\Isabel\\Analizar_snv\\30nmAu_35s\\30nmAu_110pm_35s\\Processed",
    "C:\\Users\\Rodrigo\\Desktop\\Isabel\\Analizar_snv\\30nmAu_35s\\30nmAu_1000pm_35s\\Processed",
    "C:\\Users\\Rodrigo\\Desktop\\Isabel\\Analizar_snv\\pilares\\pilares_110ppm\\Processed",
    "C:\\Users\\Rodrigo\\Desktop\\Isabel\\Analizar_snv\\pilares\\pilares_1000pm\\Processed"
]

group_labels = [
    "Ref",
    "19 ppm- 15nmAu",
    "110 ppm- 15nmAu",
    "1558 ppm- 15nmAu",
    "19 ppm- 30nmAu",
    "110 ppm- 30nmAu",
    "1558 ppm- 30nmAu",
    "110 ppm- pilares",
    "1558 ppm- pilares"
]

# Definir colores para los grupos
color_order = [
    "red", "green", "blue", "purple",  # 15nm
    "orange", "yellow", "magenta",    # 30nm
    "cyan", "brown"
]
color_map = {label: color for label, color in zip(group_labels, color_order)}

# Validar sincronización entre directorios y etiquetas
assert len(group_directories) == len(group_labels), "El número de rutas y etiquetas no coincide."


def load_group_data(directory, group_label, raman_shift_range):
    """Carga datos de un grupo, filtrando por la región de interés en Raman Shifts."""
    print(f"Cargando datos del grupo '{group_label}' en la región {raman_shift_range}.")
    all_spectra = []
    valid_files = []
    raman_shifts = None

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith(".csv"):
            try:
                data = pd.read_csv(file_path)

                if 'Raman Shifts' in data.columns:
                    filtered_data = data[(data['Raman Shifts'] >= raman_shift_range[0]) & 
                                         (data['Raman Shifts'] <= raman_shift_range[1])]
                    if raman_shifts is None:
                        raman_shifts = filtered_data['Raman Shifts'].values

                    spectra = filtered_data.drop(columns=['Raman Shifts'])
                    all_spectra.append(spectra)
                    valid_files.append(file_path)
            except Exception as e:
                print(f"Error procesando el archivo {file_path}: {e}")

    if not all_spectra:
        raise ValueError(f"No se encontraron datos válidos en: {directory}")

    consolidated_data = pd.concat(all_spectra, axis=1).T
    labels = [group_label] * len(consolidated_data)

    print(f"Procesados {len(valid_files)} archivos del grupo '{group_label}'.")
    return consolidated_data, labels, raman_shifts


def apply_snv(spectrum):
    """Aplica Standard Normal Variate (SNV) a un espectro."""
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    return (spectrum - mean) / std if std != 0 else spectrum


def get_representative_spectrum(group_data, pca_result_group):
    """Encuentra el espectro representativo de un grupo usando SNV."""
    cluster_center = np.mean(pca_result_group, axis=0)
    distances = np.linalg.norm(pca_result_group - cluster_center, axis=1)
    representative_index = np.argmin(distances)
    return apply_snv(group_data.iloc[representative_index])


""""


"""






###



def plot_full_spectra_with_region(group_directories, group_labels, raman_shift_range, pca_result_groups, manual_offset):
    """
    Grafica espectros representativos completos destacando la región de interés,
    respetando los colores asignados en color_map.
    """
    plt.figure(figsize=(12, 8))

    for i, (directory, label) in enumerate(zip(group_directories, group_labels)):
        # Cargar los datos del grupo
        group_data, _, raman_shifts = load_group_data(directory, label, (0, float('inf')))  # Cargar todo el rango
        
        # Obtener el espectro representativo
        representative_spectrum = get_representative_spectrum(group_data, pca_result_groups[i])
        
        # Usar el color asociado al grupo
        color = color_map[label]
        
        # Graficar con el offset manual
        plt.plot(raman_shifts, representative_spectrum + i * manual_offset, color=color_map[label])

    # Resaltar la región evaluada
    plt.axvspan(raman_shift_range[0], raman_shift_range[1], color='yellow', alpha=0.2)

    # Configuración del gráfico
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Intensidad (con offset)')
    plt.title('Espectros Representativos por Grupo - Región Evaluada Destacada')
    
    # Forzar la eliminación de la leyenda
    if plt.gca().get_legend():
        plt.gca().get_legend().remove()

    plt.grid()
    plt.show()





def perform_pca_analysis_and_plot_full(group_directories, group_labels, raman_shift_range, manual_offset):
    """
    Realiza el análisis PCA y luego genera un gráfico de espectros representativos completos con offset manual.
    """
    all_data = []
    all_labels = []
    pca_result_groups = []

    # Cargar y consolidar datos para todos los grupos
    for directory, label in zip(group_directories, group_labels):
        group_data, group_label, _ = load_group_data(directory, label, raman_shift_range)
        all_data.append(group_data)
        all_labels.extend(group_label)

    all_data_combined = pd.concat(all_data, axis=0).fillna(0)

    # Realizar PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_data_combined)

    # Dividir resultados por grupo
    start = 0
    for group_data in all_data:
        end = start + len(group_data)
        pca_result_groups.append(pca_result[start:end])
        start = end

    # Gráfico de varianza explicada
    explained_variance = pca.explained_variance_ratio_
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
    plt.xlabel('Componente Principal')
    plt.ylabel('Porcentaje de Varianza Explicada')
    plt.title(f'Varianza Explicada por PCA - RS Range ({raman_shift_range[0]}, {raman_shift_range[1]})')
    plt.grid()
    plt.show()

    # Gráfico PCA con etiquetas de grupos
    plt.figure(figsize=(10, 8))

    colors = [color_map[label] for label in all_labels]
    for i, (group_result, label) in enumerate(zip(pca_result_groups, group_labels)):
        plt.scatter(
            group_result[:, 0], group_result[:, 1],
            c=color_map[label], alpha=0.7, label=label, edgecolor='k'
        )

    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title(f'Análisis PCA - Comparación de Grupos - RS Range ({raman_shift_range[0]}, {raman_shift_range[1]})')

    # Añadir leyenda
    plt.legend(title="Grupos", loc="upper right")
    plt.grid()
    plt.show()

    # Graficar espectros representativos
    plot_full_spectra_with_region(group_directories, group_labels, raman_shift_range, pca_result_groups, manual_offset)


def plot_combined_layout(group_directories, group_labels, raman_shift_range, manual_offset):
    """
    Genera un lienzo con tres gráficos:
    - Análisis PCA (lado izquierdo, con opción de alternar entre 2D y 3D)
    - Espectros representativos (lado derecho superior)
    - Varianza explicada (lado derecho inferior)
    """
    # Cargar y preparar datos
    all_data = []
    all_labels = []
    pca_result_groups = []

    for directory, label in zip(group_directories, group_labels):
        group_data, group_label, _ = load_group_data(directory, label, raman_shift_range)
        all_data.append(group_data)
        all_labels.extend(group_label)

    all_data_combined = pd.concat(all_data, axis=0).fillna(0)

    # Realizar PCA
    pca = PCA(n_components=3)  # Cambiado a 3 componentes para permitir 3D
    pca_result = pca.fit_transform(all_data_combined)

    # Dividir resultados por grupo
    start = 0
    for group_data in all_data:
        end = start + len(group_data)
        pca_result_groups.append(pca_result[start:end])
        start = end

    # Crear subplots
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])

    # Crear el subplot de PCA (2D inicialmente)
# Crear el subplot de PCA (2D inicialmente) y graficar al inicio
    pca_ax = fig.add_subplot(grid[:, 0])  # Ocupa las filas completas del lado izquierdo
    is_3d = [False]  # Estado inicial: 2D

    # Gráfico PCA inicial (2D)
    for i, (group_result, label) in enumerate(zip(pca_result_groups, group_labels)):
        pca_ax.scatter(
            group_result[:, 0], group_result[:, 1],
            c=color_map[label], alpha=0.7, label=label, edgecolor='k'
        )
    pca_ax.set_xlabel('Componente Principal 1')
    pca_ax.set_ylabel('Componente Principal 2')
    pca_ax.set_title('Análisis PCA')
    pca_ax.legend(title="Grupos", loc="upper right")
    pca_ax.grid()


    # Función para alternar entre 2D y 3D
    def toggle_pca(event):
        nonlocal pca_ax  # Usar nonlocal para modificar pca_ax dentro de la función
        pca_ax.clear()  # Limpiar el eje actual

        if not is_3d[0]:
            # Cambiar a 3D
            pca_ax.remove()  # Eliminar el eje actual
            pca_ax = fig.add_subplot(grid[:, 0], projection='3d')  # Crear un eje 3D
            for i, (group_result, label) in enumerate(zip(pca_result_groups, group_labels)):
                pca_ax.scatter(
                    group_result[:, 0], group_result[:, 1], group_result[:, 2],
                    c=color_map[label], alpha=0.7, label=label
                )
            pca_ax.set_zlabel('Componente Principal 3')
        else:
            # Cambiar a 2D
            pca_ax.remove()  # Eliminar el eje actual
            pca_ax = fig.add_subplot(grid[:, 0])  # Crear un eje 2D
            for i, (group_result, label) in enumerate(zip(pca_result_groups, group_labels)):
                pca_ax.scatter(
                    group_result[:, 0], group_result[:, 1],
                    c=color_map[label], alpha=0.7, label=label, edgecolor='k'
                )

        # Configuración común
        pca_ax.set_xlabel('Componente Principal 1')
        pca_ax.set_ylabel('Componente Principal 2')
        pca_ax.set_title('Análisis PCA')
        pca_ax.legend(title="Grupos", loc="upper right")
        pca_ax.grid()
        fig.canvas.draw_idle()  # Actualizar el gráfico
        is_3d[0] = not is_3d[0]  # Alternar el estado

    # Crear un botón para alternar entre 2D y 3D
    button_ax = fig.add_axes([0.05, 0.05, 0.1, 0.05])  # [left, bottom, width, height]
    toggle_button = Button(button_ax, 'Toggle 3D')
    toggle_button.on_clicked(toggle_pca)

    # Gráfico de espectros representativos
    spectra_ax = fig.add_subplot(grid[0, 1])  # Ocupa el cuadrante superior derecho
    for i, (directory, label) in enumerate(zip(group_directories, group_labels)):
        group_data, _, raman_shifts = load_group_data(directory, label, (0, float('inf')))
        representative_spectrum = get_representative_spectrum(group_data, pca_result_groups[i])
        color = color_map[label]
        spectra_ax.plot(raman_shifts, representative_spectrum + i * manual_offset, color=color)
    spectra_ax.axvspan(raman_shift_range[0], raman_shift_range[1], color='yellow', alpha=0.2, label='Región Evaluada')
    spectra_ax.set_xlabel('Raman Shift (cm⁻¹)')
    spectra_ax.set_ylabel('Intensidad (con offset)')
    spectra_ax.set_title('Espectros Representativos por Grupo')
    spectra_ax.grid()

    # Gráfico de varianza explicada
    variance_ax = fig.add_subplot(grid[1, 1])  # Ocupa el cuadrante inferior derecho
    explained_variance = pca.explained_variance_ratio_
    variance_ax.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
    variance_ax.set_xlabel('Componente Principal')
    variance_ax.set_ylabel('Porcentaje de Varianza Explicada')
    variance_ax.set_title('Varianza Explicada por PCA')
    variance_ax.grid()

    # Ajustar espacio y mostrar
    plt.subplots_adjust(left=0.05, bottom=0.15, right=0.98, top=0.95, wspace=0.1, hspace=0.25)
    plt.show()


if __name__ == "__main__":
    manual_offset = 1.8
    plot_combined_layout(group_directories, group_labels, raman_shift_range, manual_offset)

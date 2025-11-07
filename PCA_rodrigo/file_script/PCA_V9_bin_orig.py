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
import io
from PIL import Image, ImageGrab
import tempfile
from PIL import Image
from tkinter import Tk
from PIL import Image, ImageTk
import platform
import subprocess
from matplotlib import patheffects
from datetime import datetime



# Definir el rango de interés para Raman Shifts
raman_shift_range = [700, 900]  # Convertir en lista para permitir asignaciones
  # Modificable según necesidad

# Global font scale for all plots
FONT_SCALE = 2

# Peak window tolerance around centroid peak for robust RSD (in cm^-1)
PEAK_WINDOW_CM = 5

# Directorios de los grupos y sus etiquetas asociadas
group_directories = [
    "C:\\Users\\Rodrigo\\OneDrive - UNIVERSIDAD NACIONAL AUTÓNOMA DE MÉXICO\\Raman Process\\Ejemplo_Isabel\\Analizar_snv\\Blanco policarbonato\\Processed"
]

group_labels = [
    "hola"
]

# Definir colores para los grupos
color_order = [
    "red"
]





import os
import pickle
import pandas as pd
def scale_fonts(factor=1.4):
    """Scale common Matplotlib font sizes by a factor (axes, ticks, legend, titles)."""
    keys = ['font.size', 'axes.titlesize', 'axes.labelsize', 'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize']
    for k in keys:
        try:
            plt.rcParams[k] = plt.rcParams.get(k) * factor
        except Exception:
            pass

# Apply font scaling globally
scale_fonts(FONT_SCALE)

# Make fonts bold globally
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'


def apply_bold_to_axes(ax):
    """Ensure bold font on axis titles, labels, and tick labels (2D/3D)."""
    try:
        ax.title.set_fontweight('bold')
        ax.xaxis.label.set_fontweight('bold')
        ax.yaxis.label.set_fontweight('bold')
        for t in ax.get_xticklabels():
            t.set_fontweight('bold')
        for t in ax.get_yticklabels():
            t.set_fontweight('bold')
        if hasattr(ax, 'zaxis'):
            ax.zaxis.label.set_fontweight('bold')
            for t in ax.get_zticklabels():
                t.set_fontweight('bold')
    except Exception:
        pass


def build_color_map(labels):
    """Crea un mapa de colores determinista para las etiquetas dadas."""
    unique_labels = list(dict.fromkeys(labels))
    cmap = plt.get_cmap('tab10')
    return {label: cmap(i % 10) for i, label in enumerate(unique_labels)}

# Label layout tuning (fraction of axes span)
LABEL_MIN_SEP_FRAC = 0.01   # before: 0.035 (reduce horizontal separation)
LABEL_PAD_FRAC = 0.005      # before: 0.02  (edge padding)
LABEL_Y_BASE_FRAC = 0.01    # base vertical offset above baseline
LABEL_Y_STEP_FRAC = 0.005   # step between stagger levels


def shift_axes_right(ax, shift=0.4, max_right=2):
    """Shift an axes to the right by a fraction of figure width, staying within max_right."""
    try:
        pos = ax.get_position()
        new_x0 = min(pos.x0 + shift, max_right - pos.width)
        ax.set_position([new_x0, pos.y0, pos.width, pos.height])
    except Exception:
        pass


def shrink_axes_width(ax, factor=0.9):
    """Shrink an axes' width by the given factor, anchored at the left edge."""
    try:
        pos = ax.get_position()
        new_w = pos.width * factor
        ax.set_position([pos.x0, pos.y0, new_w, pos.height])
    except Exception:
        pass


def load_group_data(directory, group_label, raman_shift_range):
    """Carga datos de un grupo desde archivos .bin, filtrando por la región de interés en Raman Shifts."""
    print(f"Cargando datos del grupo '{group_label}' en la región {raman_shift_range}.")
    all_spectra = []
    valid_files = []
    raman_shifts = None

    # Iterar sobre los archivos en el directorio
    for file_name in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, file_name)

        # Filtrar por archivos que comiencen con "orig" y terminen en ".bin"
        if file_name.startswith("orig") and file_name.endswith(".bin"):
            try:
                # Cargar el archivo binario
                with open(file_path, "rb") as bin_file:
                    data = pickle.load(bin_file)

                # Obtener los Raman Shifts y los datos de intensidad
                file_raman_shifts = data.get("Raman Shifts", [])
                intensity_data = data.get("Intensity Data", {})

                if not file_raman_shifts or not intensity_data:
                    print(f"Datos incompletos en el archivo {file_path}.")
                    continue

                # Filtrar los Raman Shifts y las intensidades correspondientes
                filtered_indices = [i for i, shift in enumerate(file_raman_shifts)
                                    if raman_shift_range[0] <= shift <= raman_shift_range[1]]

                if not filtered_indices:
                    print(f"No se encontraron datos en el rango especificado en {file_path}.")
                    continue

                # Extraer las intensidades filtradas
                filtered_shifts = [file_raman_shifts[i] for i in filtered_indices]
                filtered_spectra = {
                    point: [intensity[i] for i in filtered_indices]
                    for point, intensity in intensity_data.items()
                }

                # Convertir a DataFrame
                spectra_df = pd.DataFrame.from_dict(filtered_spectra, orient='index')

                # Establecer los Raman Shifts filtrados si aún no se han establecido
                if raman_shifts is None:
                    raman_shifts = filtered_shifts

                all_spectra.append(spectra_df)
                valid_files.append(file_path)

            except Exception as e:
                print(f"Error procesando el archivo {file_path}: {e}")

    if not all_spectra:
        raise ValueError(f"No se encontraron datos válidos en: {directory}")

    # Consolidar todos los espectros en un único DataFrame
    consolidated_data = pd.concat(all_spectra, axis=0)
    labels = [group_label] * len(consolidated_data)

    print(f"Procesados {len(valid_files)} archivos del grupo '{group_label}'.")
    return consolidated_data, labels, raman_shifts


# Removed ellipse/outlier helpers per user request



def apply_snv(spectrum):
    """Aplica Standard Normal Variate (SNV) a un espectro."""
    mean = np.mean(spectrum)
    std = np.std(spectrum)
    return (spectrum - mean) / std if std != 0 else spectrum


def get_representative_spectrum(group_data, pca_result_group):
    """Encuentra el espectro representativo de un grupo (sin aplicar SNV)."""
    cluster_center = np.mean(pca_result_group, axis=0)
    distances = np.linalg.norm(pca_result_group - cluster_center, axis=1)
    representative_index = np.argmin(distances)
    # Devolver el espectro crudo (sin normalización SNV)
    return group_data.iloc[representative_index]


""""


"""






###



def plot_full_spectra_with_region(group_directories, group_labels, raman_shift_range, pca_result_groups, manual_offset):
    """
    Grafica espectros representativos completos destacando la región de interés,
    respetando los colores asignados en color_map.
    """
    plt.figure(figsize=(12, 8))

    # Asignar colores deterministas por etiqueta
    color_map = build_color_map(group_labels)

    for i, (directory, label) in enumerate(zip(group_directories, group_labels)):
        # Cargar los datos del grupo
        group_data, _, raman_shifts = load_group_data(directory, label, (0, float('inf')))  # Cargar todo el rango
        
        # Obtener el espectro representativo
        representative_spectrum = get_representative_spectrum(group_data, pca_result_groups[i])
        
        # Usar el color asociado al grupo
        color = color_map[label]
        
        # Graficar con el offset manual
        plt.plot(raman_shifts, representative_spectrum + i * manual_offset, color=color, linewidth=2)

    # Resaltar la región evaluada
    plt.axvspan(raman_shift_range[0], raman_shift_range[1], color='yellow', alpha=0.2)

    # Configuración del gráfico
    plt.xlabel('Raman Shift (cm⁻¹)', fontweight='bold')
    plt.ylabel('Intensity (with offset)', fontweight='bold')
    plt.title('Representative Spectra by Group - Highlighted Evaluated Region', fontweight='bold')
    apply_bold_to_axes(plt.gca())
    
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
    raman_shifts_common = None
    for directory, label in zip(group_directories, group_labels):
        group_data, group_label, rshifts = load_group_data(directory, label, raman_shift_range)
        all_data.append(group_data)
        all_labels.extend(group_label)
        if raman_shifts_common is None:
            raman_shifts_common = rshifts

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
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center', color='blue')
    plt.xlabel('Principal Component', fontweight='bold')
    plt.ylabel('Explained Variance (%)', fontweight='bold')
    plt.title(f'Explained Variance by PCA - RS Range ({raman_shift_range[0]}, {raman_shift_range[1]})', fontweight='bold')
    apply_bold_to_axes(plt.gca())
    plt.grid()
    plt.show()

    # Gráfico PCA con etiquetas de grupos
    plt.figure(figsize=(10, 8))
    color_map = build_color_map(group_labels)
    colors = [color_map[label] for label in all_labels]
    for i, (group_result, label) in enumerate(zip(pca_result_groups, group_labels)):
        plt.scatter(
            group_result[:, 0], group_result[:, 1],
            c=color_map[label], alpha=0.7, label=label, edgecolor='k'
        )

    plt.xlabel('Principal Component 1', fontweight='bold')
    plt.ylabel('Principal Component 2', fontweight='bold')
    plt.title(f'PCA Analysis - Group Comparison - RS Range ({raman_shift_range[0]}, {raman_shift_range[1]})', fontweight='bold')

    # Legend removed per request
    apply_bold_to_axes(plt.gca())
    plt.grid()
    plt.show()

    # Graficar espectros representativos
    plot_full_spectra_with_region(group_directories, group_labels, raman_shift_range, pca_result_groups, manual_offset)


def plot_combined_layout(group_directories, group_labels, raman_shift_range, manual_offset, color_order, title):
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

    color_map = {label: color for label, color in zip(group_labels, color_order)}
    manual_offset = 1

    # Validar sincronización entre directorios y etiquetas
    assert len(group_directories) == len(group_labels), "El número de rutas y etiquetas no coincide."

    raman_shifts_common = None
    for directory, label in zip(group_directories, group_labels):
        group_data, group_label, rshifts = load_group_data(directory, label, raman_shift_range)
        all_data.append(group_data)
        all_labels.extend(group_label)
        if raman_shifts_common is None:
            raman_shifts_common = rshifts

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
    pca_ax = fig.add_subplot(grid[:, 0])  # Ocupa las filas completas del lado izquierdo
    is_3d = [False]  # Estado inicial: 2D

    # Gráfico PCA inicial (2D)
    for i, (group_result, label) in enumerate(zip(pca_result_groups, group_labels)):
        pca_ax.scatter(
            group_result[:, 0], group_result[:, 1],
            c=color_map[label], alpha=0.7, label=label, edgecolor='k'
        )
    # Ellipse/outlier overlays removed per request
    # Shrink PCA axes width to create extra spacing to the right
    shrink_axes_width(pca_ax, factor=0.9)
    pca_ax.set_xlabel('Principal Component 1', fontweight='bold')
    pca_ax.set_ylabel('Principal Component 2', fontweight='bold')
    pca_ax.set_title(
        f"{title} - Raman Shift {raman_shift_range[0]}-{raman_shift_range[1]}",
        fontsize=int(18 * FONT_SCALE)  # scaled title size
    )
    # Legend removed per request
    apply_bold_to_axes(pca_ax)
    # Signed Mahalanobis (1D PC1) baseline and markers with vertical labels and shading
    pc1_all = pca_result[:, 0]
    mu_global = float(np.mean(pc1_all))
    std_global = float(np.std(pc1_all, ddof=1)) if len(pc1_all) > 1 else 0.0
    std_global = std_global if std_global > 0 else 1.0
    group_mu_pc1 = [float(np.mean(gr[:, 0])) for gr in pca_result_groups]
    signed_dm = [(mu - mu_global) / std_global for mu in group_mu_pc1]
    xmin, xmax = pca_ax.get_xlim()
    ymin, ymax = pca_ax.get_ylim()
    # Ensure the baseline aligns with PC2 = 0 and is visible
    if not (ymin <= 0 <= ymax):
        pca_ax.set_ylim(min(ymin, 0), max(ymax, 0))
        ymin, ymax = pca_ax.get_ylim()
    yr = ymax - ymin
    y_base = 0.0
    band_h = 0.04 * yr
    # Shaded band behind the baseline for readability
    pca_ax.axhspan(y_base - band_h, y_base + band_h, color='gray', alpha=0.12, zorder=1)
    # Baseline
    hline = pca_ax.hlines(y_base, xmin, xmax, colors='gray', linestyles='-', alpha=0.6, linewidth=2.5, zorder=2)
    # Global centroid marker and label
    pca_ax.plot(mu_global, y_base, marker='D', color='black', markersize=9, zorder=6,
                markeredgecolor='white', markeredgewidth=1.5)
    txt0 = pca_ax.text(
        mu_global, y_base + 0.02 * yr, "0.00σ", ha='center', va='bottom', color='black',
        fontweight='bold', rotation=90, zorder=7
    )
    try:
        txt0.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])
    except Exception:
        pass
    # Compute adjusted x positions to reduce overlap among labels
    xr = xmax - xmin
    dx_min = LABEL_MIN_SEP_FRAC * xr
    pad = LABEL_PAD_FRAC * xr
    order = np.argsort(group_mu_pc1)
    x_adjusted = [None] * len(group_mu_pc1)
    placed = []
    for idx in order:
        mu = group_mu_pc1[idx]
        candidate = mu
        attempt = 0
        direction = 1
        while any(abs(candidate - px) < dx_min for px in placed):
            attempt += 1
            direction *= -1
            candidate = mu + direction * attempt * dx_min
        candidate = max(min(candidate, xmax - pad), xmin + pad)
        x_adjusted[idx] = candidate
        placed.append(candidate)

    # Group markers and vertical labels (staggered y; shifted x with leader lines)
    for i, (lbl, mu, dm) in enumerate(zip(group_labels, group_mu_pc1, signed_dm)):
        pca_ax.plot(mu, y_base, marker='o', color=color_map[lbl], markersize=9, zorder=6,
                    markeredgecolor='white', markeredgewidth=1.2)
        y_text = y_base + (LABEL_Y_BASE_FRAC + LABEL_Y_STEP_FRAC * (i % 3)) * yr
        x_text = x_adjusted[i]
        if abs(x_text - mu) > 1e-9:
            pca_ax.plot([mu, x_text], [y_base + 0.005 * yr, y_text - 0.01 * yr],
                        color=color_map[lbl], alpha=0.6, linewidth=1, zorder=5)
        label_text = f"{dm:+.2f}σ - {lbl}"
        txt = pca_ax.text(
            x_text, y_text, label_text, ha='center', va='bottom', color=color_map[lbl], fontweight='bold',
            rotation=90, zorder=7
        )
        try:
            txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])
        except Exception:
            pass
    pca_ax.margins(y=0.25)
    pca_ax.grid()
    # Biplot overlay removed per request

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
            pca_ax.set_zlabel('Principal Component 3', fontweight='bold')
        else:
            # Cambiar a 2D
            pca_ax.remove()  # Eliminar el eje actual
            pca_ax = fig.add_subplot(grid[:, 0])  # Crear un eje 2D
            for i, (group_result, label) in enumerate(zip(pca_result_groups, group_labels)):
                pca_ax.scatter(
                    group_result[:, 0], group_result[:, 1],
                    c=color_map[label], alpha=0.7, label=label, edgecolor='k'
                )
                # Ellipse/outlier overlays removed per request

        # Configuración común
        pca_ax.set_xlabel('Principal Component 1', fontweight='bold')
        pca_ax.set_ylabel('Principal Component 2', fontweight='bold')
        # Shrink PCA axes width also after toggling 2D/3D
        shrink_axes_width(pca_ax, factor=0.9)
        pca_ax.set_title(
            f"{title} - Raman Shift {raman_shift_range[0]}-{raman_shift_range[1]}",
            fontsize=int(18 * FONT_SCALE)
        )
        # Legend removed per request
        apply_bold_to_axes(pca_ax)
        # Re-draw signed Mahalanobis (1D PC1) baseline and markers with vertical labels and shading
        pc1_all = pca_result[:, 0]
        mu_global = float(np.mean(pc1_all))
        std_global = float(np.std(pc1_all, ddof=1)) if len(pc1_all) > 1 else 0.0
        std_global = std_global if std_global > 0 else 1.0
        group_mu_pc1 = [float(np.mean(gr[:, 0])) for gr in pca_result_groups]
        signed_dm = [(mu - mu_global) / std_global for mu in group_mu_pc1]
        xmin, xmax = pca_ax.get_xlim()
        ymin, ymax = pca_ax.get_ylim()
        # Ensure the baseline aligns with PC2 = 0 and is visible
        if not (ymin <= 0 <= ymax):
            pca_ax.set_ylim(min(ymin, 0), max(ymax, 0))
            ymin, ymax = pca_ax.get_ylim()
        yr = ymax - ymin
        y_base = 0.0
        band_h = 0.04 * yr
        pca_ax.axhspan(y_base - band_h, y_base + band_h, color='gray', alpha=0.12, zorder=1)
        pca_ax.hlines(y_base, xmin, xmax, colors='gray', linestyles='-', alpha=0.6, linewidth=2.5, zorder=2)
        pca_ax.plot(mu_global, y_base, marker='D', color='black', markersize=9, zorder=6,
                    markeredgecolor='white', markeredgewidth=1.5)
        txt0 = pca_ax.text(
            mu_global, y_base + 0.02 * yr, "0.00σ", ha='center', va='bottom', color='black',
            fontweight='bold', rotation=90, zorder=7
        )
        try:
            txt0.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])
        except Exception:
            pass
        # Adjust x positions to reduce overlap
        xr = xmax - xmin
        dx_min = LABEL_MIN_SEP_FRAC * xr
        pad = LABEL_PAD_FRAC * xr
        order = np.argsort(group_mu_pc1)
        x_adjusted = [None] * len(group_mu_pc1)
        placed = []
        for idx in order:
            mu = group_mu_pc1[idx]
            candidate = mu
            attempt = 0
            direction = 1
            while any(abs(candidate - px) < dx_min for px in placed):
                attempt += 1
                direction *= -1
                candidate = mu + direction * attempt * dx_min
            candidate = max(min(candidate, xmax - pad), xmin + pad)
            x_adjusted[idx] = candidate
            placed.append(candidate)

        for i, (lbl, mu, dm) in enumerate(zip(group_labels, group_mu_pc1, signed_dm)):
            pca_ax.plot(mu, y_base, marker='o', color=color_map[lbl], markersize=9, zorder=6,
                        markeredgecolor='white', markeredgewidth=1.2)
            y_text = y_base + (LABEL_Y_BASE_FRAC + LABEL_Y_STEP_FRAC * (i % 3)) * yr
            x_text = x_adjusted[i]
            if abs(x_text - mu) > 1e-9:
                pca_ax.plot([mu, x_text], [y_base + 0.005 * yr, y_text - 0.01 * yr],
                            color=color_map[lbl], alpha=0.6, linewidth=1, zorder=5)
            label_text = f"{dm:+.2f}σ - {lbl}"
            txt = pca_ax.text(
                x_text, y_text, label_text, ha='center', va='bottom', color=color_map[lbl], fontweight='bold',
                rotation=90, zorder=7
            )
            try:
                txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground='white')])
            except Exception:
                pass
        pca_ax.margins(y=0.25)
        pca_ax.grid()
        fig.canvas.draw_idle()  # Actualizar el gráfico
        is_3d[0] = not is_3d[0]  # Alternar el estado

    # Crear un botón para alternar entre 2D y 3D
    
    # Gráfico de espectros representativos
    spectra_ax = fig.add_subplot(grid[0, 1])  # Ocupa el cuadrante superior derecho
    for i, (directory, label) in enumerate(zip(group_directories, group_labels)):
        group_data, _, raman_shifts = load_group_data(directory, label, (0, float('inf')))
        representative_spectrum = get_representative_spectrum(group_data, pca_result_groups[i])
        color = color_map[label]
        spectra_ax.plot(raman_shifts, representative_spectrum + i * manual_offset, color=color)
    spectra_ax.axvspan(raman_shift_range[0], raman_shift_range[1], color='yellow', alpha=0.2, label='Evaluated Region')
    spectra_ax.set_xlabel('Raman Shift (cm⁻¹)', fontweight='bold')
    spectra_ax.set_ylabel('Intensity (with offset)', fontweight='bold')
    spectra_ax.set_title('Representative Spectra by Group', fontweight='bold')
    apply_bold_to_axes(spectra_ax)
    spectra_ax.grid()
    # Move the spectra plot slightly to the right
    shift_axes_right(spectra_ax, shift=0.06, max_right=0.985)

    # Gráfico de varianza explicada
    variance_ax = fig.add_subplot(grid[1, 1])  # Ocupa el cuadrante inferior derecho
    explained_variance = pca.explained_variance_ratio_
    variance_ax.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center')
    variance_ax.set_xlabel('Principal Component', fontweight='bold')
    variance_ax.set_ylabel('Explained Variance (%)', fontweight='bold')
    variance_ax.set_title('Explained Variance by PCA', fontweight='bold')
    apply_bold_to_axes(variance_ax)
    variance_ax.grid()
    # Move the variance plot slightly to the right
    shift_axes_right(variance_ax, shift=0.06, max_right=0.985)

    # =====================
    # Export metrics and spectra
    # =====================
    try:
        # Prepare output directory and file names
        script_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
        out_dir = os.path.join(script_dir, 'outputs')
        os.makedirs(out_dir, exist_ok=True)

        def sanitize_filename(s):
            keep = [c if c.isalnum() or c in (' ', '-', '_') else '_' for c in s]
            return ''.join(keep).strip().replace(' ', '_')

        # Summary sheet: title and explained variance of PC1 and PC2
        ev = pca.explained_variance_ratio_
        title_row = {
            'Title': [title],
            'Raman Shift Range': [f"{raman_shift_range[0]}-{raman_shift_range[1]}"]
        }
        ev_row = {
            'PC1 Variance Ratio': [float(ev[0]) if len(ev) > 0 else None],
            'PC2 Variance Ratio': [float(ev[1]) if len(ev) > 1 else None]
        }
        peak_cfg = {'Peak Window (cm^-1)': [f"±{PEAK_WINDOW_CM}"]}
        summary_df = pd.concat([pd.DataFrame(title_row), pd.DataFrame(ev_row), pd.DataFrame(peak_cfg)], axis=1)

        # Group metrics sheet
        pc1_all = pca_result[:, 0]
        mu_global = float(np.mean(pc1_all))
        std_global = float(np.std(pc1_all, ddof=1)) if len(pc1_all) > 1 else 0.0
        std_global = std_global if std_global > 0 else 1.0

        metrics_rows = []
        for i, (label, group_df, grp_scores) in enumerate(zip(group_labels, all_data, pca_result_groups)):
            grp_scores = np.asarray(grp_scores)
            n = grp_scores.shape[0]
            pc1_cent = float(np.mean(grp_scores[:, 0]))
            pc2_cent = float(np.mean(grp_scores[:, 1]))
            pc1_signed_sigma = (pc1_cent - mu_global) / std_global
            pc1_std = float(np.std(grp_scores[:, 0], ddof=1)) if n > 1 else 0.0
            pc2_std = float(np.std(grp_scores[:, 1], ddof=1)) if n > 1 else 0.0
            cov_det = None
            try:
                if n > 1:
                    cov = np.cov(grp_scores[:, :2].T)
                    cov_det = float(np.linalg.det(cov))
            except Exception:
                cov_det = None

            # Representative (centroid) spectrum and its peak
            rep_spec = get_representative_spectrum(group_df, grp_scores)
            rep_vals = rep_spec.values.astype(float)
            idx_peak = int(np.argmax(rep_vals))
            peak_shift = float(raman_shifts_common[idx_peak]) if raman_shifts_common is not None else idx_peak
            peak_intensity = float(rep_vals[idx_peak])

            # RSD at centroid peak across all raw spectra in this group (fixed index)
            try:
                intensities_at_peak = group_df.iloc[:, idx_peak].astype(float).values
                mean_peak = float(np.mean(intensities_at_peak))
                std_peak = float(np.std(intensities_at_peak, ddof=1)) if len(intensities_at_peak) > 1 else 0.0
                rsd_percent = float(std_peak / mean_peak * 100.0) if mean_peak != 0 else np.nan
            except Exception:
                mean_peak = np.nan
                std_peak = np.nan
                rsd_percent = np.nan

            # Robust RSD within ± window around centroid peak (per spectrum, take max in window)
            try:
                shifts_arr = np.asarray(raman_shifts_common, dtype=float)
                win_mask = np.abs(shifts_arr - peak_shift) <= float(PEAK_WINDOW_CM)
                win_idx = np.where(win_mask)[0]
                if win_idx.size == 0:
                    win_idx = np.array([idx_peak])
                # values shape: n_spectra x n_window
                vals_window = group_df.iloc[:, win_idx].astype(float).values
                # max per spectrum in window
                max_per_spec = np.max(vals_window, axis=1)
                mean_win = float(np.mean(max_per_spec))
                std_win = float(np.std(max_per_spec, ddof=1)) if len(max_per_spec) > 1 else 0.0
                rsd_win = float(std_win / mean_win * 100.0) if mean_win != 0 else np.nan
                window_span = float(shifts_arr[win_idx].max() - shifts_arr[win_idx].min()) if win_idx.size > 1 else 0.0
            except Exception:
                mean_win = np.nan
                std_win = np.nan
                rsd_win = np.nan
                window_span = np.nan

            metrics_rows.append({
                'Group': label,
                'N': n,
                'PC1 Centroid': pc1_cent,
                'PC2 Centroid': pc2_cent,
                'PC1 Signed Distance (σ)': pc1_signed_sigma,
                'PC1 Std': pc1_std,
                'PC2 Std': pc2_std,
                'Within-group CovDet (PC1-2)': cov_det,
                'Centroid Peak Shift (cm^-1)': peak_shift,
                'Centroid Peak Intensity': peak_intensity,
                'Mean Intensity at Peak': mean_peak,
                'RSD% at Peak (raw)': rsd_percent,
                'Mean Intensity in Window': mean_win,
                'RSD% in Window (raw)': rsd_win,
                'Window Span Used (cm^-1)': window_span
            })

            # Note: Per-group TXT export removed; we'll export a single combined TXT below.
            pass

        group_metrics_df = pd.DataFrame(metrics_rows)

        # Build wide centroid spectra DataFrame (full spectrum): first column = Raman_Shift_cm^-1,
        # then one column per group with the centroid representative spectrum (no cropping).
        centroid_series = {}
        all_shifts = set()
        for i, (directory, label) in enumerate(zip(group_directories, group_labels)):
            try:
                full_group_data, _, full_shifts = load_group_data(directory, label, (0, float('inf')))
                rep_spec_full = get_representative_spectrum(full_group_data, pca_result_groups[i])
                # Series indexed by shift
                ser = pd.Series(rep_spec_full.values.astype(float), index=pd.Index(full_shifts, name='Raman_Shift_cm^-1'))
                # If duplicated shifts exist, average them
                ser = ser.groupby(level=0).mean()
                centroid_series[label] = ser
                all_shifts.update(ser.index.tolist())
            except Exception:
                # Skip this group if anything fails during full-range export
                continue

        centroid_df = None
        if all_shifts:
            wide_index = sorted(all_shifts)
            centroid_df = pd.DataFrame({'Raman_Shift_cm^-1': wide_index})
            for label, ser in centroid_series.items():
                centroid_df[label] = ser.reindex(wide_index).values

        # Write Excel with two sheets
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_name = f"results_{sanitize_filename(title)}_{ts}.xlsx"
        excel_path = os.path.join(out_dir, excel_name)
        with pd.ExcelWriter(excel_path) as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            group_metrics_df.to_excel(writer, sheet_name='GroupMetrics', index=False)
            if centroid_df is not None:
                centroid_df.to_excel(writer, sheet_name='CentroidSpectra', index=False)
        print(f"Saved Excel metrics to: {excel_path}")

        # Build a single combined TXT with full-spectrum centroid for each group
        ts2 = datetime.now().strftime('%Y%m%d_%H%M%S')
        combined_txt = os.path.join(out_dir, f"centroid_spectra_all_{sanitize_filename(title)}_{ts2}.txt")
        with open(combined_txt, 'w', encoding='utf-8') as f:
            # Metadata header lines
            sep = ';'
            f.write(f"Title{sep}{title}\n")
            f.write(f"Generated{sep}{datetime.now().isoformat()}\n")
            f.write(f"Export_Type{sep}Combined centroid spectra (full spectrum)\n")
            f.write(f"Delimiter{sep}semicolon\n")
            f.write(f"Evaluated_Region_cm^-1{sep}{raman_shift_range[0]}-{raman_shift_range[1]}\n\n")

            # Table header
            f.write(f"Group{sep}Raman_Shift_cm^-1{sep}Intensity\n")

            # Long-format rows: one line per (group, shift)
            for i, (directory, label) in enumerate(zip(group_directories, group_labels)):
                try:
                    # Load full-range data for this group
                    full_group_data, _, full_shifts = load_group_data(directory, label, (0, float('inf')))
                    # Use the same representative index selection used for PCA group
                    rep_spec_full = get_representative_spectrum(full_group_data, pca_result_groups[i])
                    rep_vals_full = rep_spec_full.values.astype(float)
                    for rs, val in zip(full_shifts, rep_vals_full):
                        f.write(f"{label}{sep}{rs}{sep}{val}\n")
                except Exception as ex:
                    # Log error as a metadata-style line
                    f.write(f"Warning{sep}Failed to export group {label}: {ex}\n")

        print(f"Saved combined centroid spectra TXT to: {combined_txt}")
    except Exception as e:
        print(f"Warning: failed to export metrics/spectra: {e}")

    # Ajustar espacio y mostrar
    
    def copy_to_clipboard(event):
        """Save the figure to a temporary PNG file and open it for manual copy."""
        # Create an in-memory buffer for the figure
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        # Convert the buffer to a Pillow image
        image = Image.open(buf)

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            image.save(tmpfile.name, format="PNG")
            tmpfile_path = tmpfile.name

        # Inform and open the temporary file
        print(f"The image has been saved to {tmpfile_path}. You can copy it manually.")
        Image.open(tmpfile_path).show()


    # Botón para alternar entre 2D y 3D
    button_ax = fig.add_axes([0.05, 0.01, 0.08, 0.03])  # [left, bottom, width, height]
    toggle_button = Button(button_ax, '2D/3D')
    try:
        toggle_button.label.set_fontweight('bold')
    except Exception:
        pass
    toggle_button.on_clicked(toggle_pca)

    # Botón para copiar la imagen
    button_ax_copy = fig.add_axes([0.15, 0.01, 0.08, 0.03])  # [left, bottom, width, height]
    copy_button = Button(button_ax_copy, 'Copy')
    try:
        copy_button.label.set_fontweight('bold')
    except Exception:
        pass
    copy_button.on_clicked(copy_to_clipboard)


    plt.subplots_adjust(left=0.06, bottom=0.15, right=0.98, top=0.95, wspace=0.2, hspace=0.25)
    plt.show()




if __name__ == "__main__":
    manual_offset = 0
    title = "PCA Analysis of Samples"  # Default chart title in English
    plot_combined_layout(group_directories, group_labels, raman_shift_range, manual_offset, color_order, title)

import os
import pandas as pd
import pickle
import glob
import math
from sklearn.preprocessing import StandardScaler


def detect_csv_structure(file_path):
    """
    Detecta la estructura del archivo CSV ignorando filas vacías.
    """
    try:
        df = pd.read_csv(file_path, header=None, skip_blank_lines=False)
        if df.empty:
            print(f"El archivo {file_path} está vacío. Saltando...")
            return None, None
        first_row = df.iloc[0]
        if pd.to_numeric(first_row, errors="coerce").notna().all():
            print(f"Detectada estructura de una sola matriz (sin encabezados) en {file_path}.")
            return "single_matrix", df.reset_index(drop=True)
        else:
            print(f"Detectada estructura con múltiples matrices (con encabezados) en {file_path}.")
            return "multiple_matrices", df.reset_index(drop=True)
    except Exception as e:
        print(f"Error al detectar la estructura del archivo: {e}")
        return None, None


def snv_normalization(data):
    """
    Normaliza los datos de intensidad utilizando el método de Standard Normal Variate (SNV).
    """
    intensities = [point for point in data.values()]
    df_intensities = pd.DataFrame(intensities).T  # Transponer para columnas como puntos
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df_intensities)

    normalized_dict = {}
    for i, point in enumerate(data.keys()):
        normalized_dict[point] = normalized_data[:, i].tolist()

    return normalized_dict


def process_directory_and_save_combined(directory_path, output_file):
    """
    Procesa todos los archivos CSV en un directorio y guarda los datos combinados en un archivo binario.
    """
    try:
        csv_files = glob.glob(os.path.join(directory_path, '**', '*.csv'), recursive=True)

        if not csv_files:
            print(f"No se encontraron archivos CSV en el directorio {directory_path}.")
            return

        combined_data = {
            "Raman Shifts": None,  # Para guardar los Raman Shifts compartidos
            "Intensity Data": {}   # Para guardar todas las intensidades combinadas
        }

        for file_path in csv_files:
            print(f"Procesando archivo: {file_path}")
            structure, df = detect_csv_structure(file_path)

            if df is None:
                continue

            if structure == "single_matrix":
                process_single_matrix(df, file_path, combined_data)
            elif structure == "multiple_matrices":
                process_multiple_matrices(df, file_path, combined_data)
            else:
                print(f"Estructura no soportada o mal formateada para el archivo: {file_path}")

        # Guardar los datos combinados en un archivo binario
        with open(output_file, "wb") as bin_file:
            pickle.dump(combined_data, bin_file)

        print(f"Todos los datos combinados se guardaron en '{output_file}'.")

    except Exception as e:
        print(f"Error al procesar y guardar los datos: {e}")


def process_single_matrix(df, file_path, combined_data):
    """
    Procesa una única matriz y la agrega al conjunto combinado.
    """
    try:
        df = df.dropna(how='all').reset_index(drop=True)

        raman_shifts = df.iloc[0].dropna().values.tolist()
        intensity_data = {}

        num_rows = len(df) - 1
        matrix_size = int(num_rows ** 0.5)
        if matrix_size ** 2 != num_rows:
            raise ValueError(f"El archivo no corresponde a una matriz cuadrada válida: {file_path}")

        for i, row in df.iloc[1:].iterrows():
            m, n = divmod(i, matrix_size)
            point_name = f"Point_{m+1}_{n+1}"
            intensity_data[point_name] = row.dropna().values.tolist()

        normalized_data = snv_normalization(intensity_data)

        matrix_name = os.path.splitext(os.path.basename(file_path))[0]
        if combined_data["Raman Shifts"] is None:
            combined_data["Raman Shifts"] = raman_shifts
        elif combined_data["Raman Shifts"] != raman_shifts:
            raise ValueError(f"Los Raman Shifts en '{file_path}' no coinciden con los existentes.")

        for point, intensities in normalized_data.items():
            combined_key = f"{matrix_name}_{point}"
            combined_data["Intensity Data"][combined_key] = intensities

    except Exception as e:
        print(f"Error procesando el archivo '{file_path}': {e}")


def process_multiple_matrices(df, file_path, combined_data):
    """
    Procesa múltiples matrices y las agrega al conjunto combinado.
    """
    try:
        if df.iloc[0, 0] != 'axisX WL':
            print(f"El archivo {file_path} no contiene 'axisX WL' en A1. Saltando este archivo...")
            return

        raman_shifts = df.iloc[0, 1:].dropna().values.tolist()
        data = df.iloc[1:, :]
        grouped_data = data.groupby(data.iloc[:, 0])

        for matrix_name, group in grouped_data:
            matrix_name = matrix_name.strip()
            group = group.drop(columns=[group.columns[0]])
            matrix_size = len(group)
            matrix_size_sqrt = int(math.sqrt(matrix_size))

            if matrix_size_sqrt ** 2 != matrix_size:
                print(f"Matriz '{matrix_name}' no tiene un tamaño cuadrado válido. Saltando...")
                continue

            intensity_data = {}
            for i, (_, row) in enumerate(group.iterrows()):
                m, n = divmod(i, matrix_size_sqrt)
                point_name = f"Point_{m+1}_{n+1}"
                intensity_data[point_name] = row.dropna().values.tolist()

            normalized_data = snv_normalization(intensity_data)

            if combined_data["Raman Shifts"] is None:
                combined_data["Raman Shifts"] = raman_shifts
            elif combined_data["Raman Shifts"] != raman_shifts:
                raise ValueError(f"Los Raman Shifts en '{file_path}' no coinciden con los existentes.")

            for point, intensities in normalized_data.items():
                combined_key = f"{matrix_name}_{point}"
                combined_data["Intensity Data"][combined_key] = intensities

    except Exception as e:
        print(f"Error procesando el archivo '{file_path}': {e}")


def process_all_directories(group_directories):
    """
    Procesa una lista de directorios, generando un archivo binario combinado en cada uno.
    """
    for directory in group_directories:
        print(f"Procesando directorio: {directory}")
        output_file = os.path.join(directory, "snv_combined_data_spectra.bin")
        process_directory_and_save_combined(directory, output_file)
    print("Todos los directorios fueron procesados.")


if __name__ == "__main__":
    # Si deseas probarlo directamente desde este script
    group_directories = [
        "path/to/directory1",
        "path/to/directory2",
        "path/to/directory3"
    ]
    process_all_directories(group_directories)

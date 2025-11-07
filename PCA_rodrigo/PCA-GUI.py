import tkinter as tk
from tkinter import filedialog, Toplevel, ttk
import json 
import os

class PCAAnalyserApp:
    def __init__(self, root):
        self.root = root



        self.frame = tk.Frame(root)
        self.frame.pack(padx=10, pady=10)

       
        icon_path = os.path.join(os.path.dirname(__file__), "file_script", "icon.png")
        icon = tk.PhotoImage(file=icon_path)
        self.root.iconphoto(True, icon)


        self.root.title("PCA analyser raman spectrum")


         # Reducir el tamaño del ícono (ajusta el valor según lo necesites)
        icon = icon.subsample(8, 8)  # Reduce el tamaño a la mitad








        # Mostrar el ícono en la ventana, centrado
        self.icon_label = tk.Label(self.root, image=icon)

        
        self.icon_label.image = icon  # Mantener referencia para evitar que se elimine
        self.icon_label.pack(pady=20)
        
        # Variables para almacenar directorios, etiquetas y colores
        self.group_directories = []
        self.group_labels = [] 
        self.color_order = []

        # Lista de colores
        self.color_list = [
            "red", "green", "blue", "purple", "orange", "yellow", "magenta",
            "cyan", "brown", "pink", "lime", "teal", "olive", "navy", "maroon", "gray"
        ]

        # Frame principal
        

        # Título
        tk.Label(self.frame, text="PCA analyser raman spectrum", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=5, pady=10)

        # Primera línea: Botones
        self.add_control_buttons()

        # Segunda línea: Raman shift range
        self.add_raman_shift_line()

        # Agregar la primera línea de entrada para grupos
        self.add_group_line()

    def add_control_buttons(self):
        """Añade la primera línea con los botones 'Add Group', 'Check', 'Run', 'Save', y 'SNV'."""
        tk.Button(self.frame, text="Add Group", command=self.add_group_line).grid(row=1, column=0, padx=5, pady=5)
        tk.Button(self.frame, text="Check", command=self.show_variables).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.frame, text="Run", command=self.run_pca_analysis).grid(row=1, column=2, padx=5, pady=5)
        tk.Button(self.frame, text="SNV", command=self.run_snv_and_pca_analysis).grid(row=1, column=3, padx=5, pady=5)
        tk.Button(self.frame, text="Save", command=self.save_to_file).grid(row=1, column=4, padx=5, pady=5)  # Botón Save
        tk.Button(self.frame, text="Open", command=self.open_from_file).grid(row=1, column=5, padx=5, pady=5)  # Botón Open
        tk.Button(self.frame, text="?", command=self.show_help).grid(row=1, column=6, padx=5, pady=5)  # Botón Help



    def run_pca_analysis(self):
        """Ejecuta el análisis con datos originales seguido del análisis PCA."""
        try:
            # === Paso 1: Ejecución del script csv_to_bin_orig.py ===
            # Obtener las rutas de los directorios desde la interfaz
            group_directories = [var.get() for var in self.group_directories]

            # Validación: asegurar que al menos un directorio esté seleccionado
            if not all(group_directories):
                raise ValueError("Asegúrate de que todos los directorios estén seleccionados.")

            # Importar y ejecutar el script csv_to_bin_orig.py
            import sys
            import os
            script_dir = os.path.join(os.path.dirname(__file__), "file_script")
            sys.path.append(script_dir)

            import csv_to_bin_orig  # Importar el módulo

            # Llamar a la función para procesar los directorios
            csv_to_bin_orig.process_all_directories(group_directories)

            # Mostrar mensaje de éxito para el preprocesamiento
            print("Preprocesamiento con datos originales completado correctamente.")

            # === Paso 2: Ejecución del script PCA_V9_bin_orig.py ===
            # Obtener variables adicionales necesarias para el análisis PCA
            raman_shift_range = (
                float(self.min_raman_shift.get()),  # Mínimo
                float(self.max_raman_shift.get())   # Máximo
            )
            group_labels = [entry.get() for entry in self.group_labels]  # Obtener directamente de las entradas
            color_order = [var.get() for var in self.color_order]
            title = self.title.get()  # Obtener el título desde el cuadro de texto

            # Validación
            if not all(group_labels) or len(set(group_labels)) != len(group_labels):
                raise ValueError("Asegúrate de que todas las etiquetas sean únicas y no estén vacías.")
            if len(group_directories) != len(group_labels) or len(group_labels) != len(color_order):
                raise ValueError("Asegúrate de que cada grupo tenga una carpeta, una etiqueta y un color asociado.")

            # Importar el script PCA_V9_bin_orig
            import PCA_V9_bin_orig

            # Definir el valor de manual_offset
            manual_offset = 0 # Valor ajustable según necesidad

            # Llamar a la función con los argumentos correctos
            PCA_V9_bin_orig.plot_combined_layout(
                group_directories,
                group_labels,
                raman_shift_range,
                manual_offset,
                color_order,
                title  # Pasar el título como argumento
            )

            # Mostrar mensaje de éxito para PCA
            import tkinter as tk
            from tkinter import messagebox
            tk.messagebox.showinfo("Éxito", "El análisis PCA con datos originales se completó correctamente y las gráficas se generaron.")

        except Exception as e:
            # Mostrar mensaje de error si algo falla
            import tkinter as tk
            from tkinter import messagebox
            tk.messagebox.showerror("Error", f"Ocurrió un error al ejecutar el análisis:\n{e}")
            print(f"Error al ejecutar el análisis PCA con datos originales: {e}")






    def run_snv_and_pca_analysis(self):
        """Ejecuta el análisis SNV seguido del análisis PCA."""
        try:
            # === Paso 1: Ejecución del script csv_to_bin_snv.py ===
            group_directories = [var.get() for var in self.group_directories]

            if not all(group_directories):
                raise ValueError("Asegúrate de que todos los directorios estén seleccionados.")

            import sys
            import os
            script_dir = os.path.join(os.path.dirname(__file__), "file_script")
            sys.path.append(script_dir)

            import csv_to_bin_snv
            csv_to_bin_snv.process_all_directories(group_directories)
            print("Preprocesamiento con SNV completado correctamente.")

            # === Paso 2: Ejecución del script PCA_V9_bin_snv.py ===
            raman_shift_range = (
                float(self.min_raman_shift.get()),
                float(self.max_raman_shift.get())
            )
            group_labels = [entry.get() for entry in self.group_labels]
            color_order = [var.get() for var in self.color_order]
            title = self.title.get()  # Obtener el título desde el cuadro de texto

            if not all(group_labels) or len(set(group_labels)) != len(group_labels):
                raise ValueError("Asegúrate de que todas las etiquetas sean únicas y no estén vacías.")
            if len(group_directories) != len(group_labels) or len(group_labels) != len(color_order):
                raise ValueError("Asegúrate de que cada grupo tenga una carpeta, una etiqueta y un color asociado.")

            import PCA_V9_bin_snv
            manual_offset = 0
            PCA_V9_bin_snv.plot_combined_layout(
                group_directories,
                group_labels,
                raman_shift_range,
                manual_offset,
                color_order,
                title  # Pasar el título como argumento
            )

            import tkinter as tk
            from tkinter import messagebox
            tk.messagebox.showinfo("Éxito", "El análisis SNV y PCA se completaron correctamente y las gráficas se generaron.")

        except Exception as e:
            import tkinter as tk
            from tkinter import messagebox
            tk.messagebox.showerror("Error", f"Ocurrió un error al ejecutar el análisis:\n{e}")
            print(f"Error al ejecutar el análisis SNV y PCA: {e}")







    def show_help(self):
            """Abre una ventana para mostrar el contenido del archivo info.txt."""
            # Ruta del archivo info.txt
            info_path = os.path.join(os.path.dirname(__file__), "file_script", "info.txt")
            try:
                # Leer el contenido del archivo
                with open(info_path, "r") as file:
                    info_content = file.read()

                # Crear una nueva ventana para mostrar el contenido
                help_window = Toplevel(self.root)
                help_window.title("Information")
                help_window.geometry("400x300")

                # Mostrar el contenido en un widget de texto
                text_widget = tk.Text(help_window, wrap="word", font=("Arial", 12))
                text_widget.insert("1.0", info_content)
                text_widget.configure(state="disabled")  # Hacerlo solo lectura
                text_widget.pack(fill="both", expand=True, padx=10, pady=10)

            except FileNotFoundError:
                tk.messagebox.showerror("Error", f"File 'info.txt' not found in {info_path}.")
            except Exception as e:
                tk.messagebox.showerror("Error", f"An error occurred: {e}")



    def add_raman_shift_line(self):
        """Añade la segunda línea con el rango de Raman shift y un campo adicional para el título."""
        # Etiqueta para Raman shift range
        tk.Label(self.frame, text="Raman shift range:", font=("Arial", 12)).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        # Entradas para el rango mínimo y máximo de Raman shift
        self.min_raman_shift = tk.StringVar(value="0")
        tk.Entry(self.frame, textvariable=self.min_raman_shift, width=10).grid(row=2, column=1, padx=5, pady=5)
        self.max_raman_shift = tk.StringVar(value="2000")
        tk.Entry(self.frame, textvariable=self.max_raman_shift, width=10).grid(row=2, column=2, padx=5, pady=5)
        
        # Etiqueta y cuadro de texto para el título
        tk.Label(self.frame, text="Title:", font=("Arial", 12)).grid(row=2, column=3, padx=5, pady=5, sticky="w")
        self.title = tk.StringVar(value="Default Title")
        tk.Entry(self.frame, textvariable=self.title, width=20).grid(row=2, column=4, padx=5, pady=5)


    def add_group_line(self):
        """Añade una línea de entrada para seleccionar carpeta, grupo y color."""
        row = len(self.group_directories) + 3  # Fila dinámica basada en las entradas actuales

        # Botón "Select Folder"
        dir_var = tk.StringVar()
        select_folder_button = tk.Button(self.frame, text="Select Folder", command=lambda: self.select_folder(dir_var, status_label))
        select_folder_button.grid(row=row, column=0, padx=5, pady=5)

        # Estado de selección
        status_label = tk.Label(self.frame, text="Not Selected", fg="red")
        status_label.grid(row=row, column=1, padx=5, pady=5)

        # Entry para nombre del grupo
        group_entry = tk.Entry(self.frame, width=20)
        group_entry.grid(row=row, column=2, padx=5, pady=5)

        # Lista desplegable para seleccionar color
        color_var = tk.StringVar(value=self.color_list[row - 3])  # Color predeterminado según la fila
        color_dropdown = ttk.Combobox(self.frame, textvariable=color_var, values=self.color_list, state="readonly", width=10)
        color_dropdown.grid(row=row, column=3, padx=5, pady=5)

        # Botón "X" para eliminar la fila
        tk.Button(self.frame, text="X", command=lambda: self.delete_group_line(row)).grid(row=row, column=4, padx=5, pady=5)

        # Guardar referencias
        self.group_directories.append(dir_var)
        self.color_order.append(color_var)
        self.group_labels.append(group_entry)  # Guardar directamente la referencia al Entry+


    def get_group_labels(self):
        """Obtiene los valores actuales de las etiquetas de los grupos."""
        return [entry.get() for entry in self.group_labels]



    def delete_group_line(self, row):
        """Elimina una línea completa (carpeta, grupo y color) y actualiza las filas dinámicamente."""
        for widget in self.frame.grid_slaves():
            if widget.grid_info()["row"] == row:
                widget.destroy()

        # Remover referencias de las listas
        index = row - 3
        self.group_directories.pop(index)
        self.color_order.pop(index)
        self.group_labels.pop(index)

        # Reorganizar las filas restantes
        self.reorganize_rows()

    def reorganize_rows(self):
        """Reorganiza las filas después de eliminar una línea."""
        for idx, (dir_var, label_entry, color_var) in enumerate(zip(self.group_directories, self.group_labels, self.color_order)):
            row = idx + 3  # Recalcular la fila basada en el índice actual

            # Actualizar posiciones de los widgets
            button = self.frame.grid_slaves(row=row, column=0)
            if button:
                button[0].grid(row=row, column=0, padx=5, pady=5)

            entry = self.frame.grid_slaves(row=row, column=1)
            if entry:
                entry[0].grid(row=row, column=1, padx=5, pady=5)

            dropdown = self.frame.grid_slaves(row=row, column=2)
            if dropdown:
                dropdown[0].grid(row=row, column=2, padx=5, pady=5)

            delete_button = self.frame.grid_slaves(row=row, column=3)
            if delete_button:
                delete_button[0].grid(row=row, column=3, padx=5, pady=5)

            

    def select_folder(self, dir_var, status_label):
        """Abre un diálogo para seleccionar carpeta, actualiza el estado y guarda la ruta en un formato adecuado."""
        folder_path = filedialog.askdirectory()
        if folder_path:
            # Convertir la ruta seleccionada al formato adecuado para Python
            folder_path = os.path.normpath(folder_path)

            # Guardar la ruta en el widget asociado
            dir_var.set(folder_path)
            status_label.config(text="Selected", fg="green")
        else:
            # Si no se selecciona una carpeta, limpiar el valor
            dir_var.set("")
            status_label.config(text="Not Selected", fg="red")


    def show_variables(self):
        """Abre una nueva ventana y muestra las variables almacenadas."""
        new_window = Toplevel(self.root)
        new_window.title("Variables")

        # Obtener valores
        min_raman = self.min_raman_shift.get()
        max_raman = self.max_raman_shift.get()
        directories = [f"Group {i + 1}: {self.group_directories[i].get()}" for i in range(len(self.group_directories))]
        labels = [f"Group {i + 1}: {self.group_labels[i].get()}    -   {self.color_order[i].get()}" for i in range(len(self.group_labels))]

        # Mostrar las variables
        tk.Label(new_window, text="Raman Shift Range:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
        tk.Label(new_window, text=f"Min: {min_raman}\nMax: {max_raman}", font=("Arial", 12)).pack(anchor="w", padx=10)

        tk.Label(new_window, text="Directories:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
        tk.Label(new_window, text="\n".join(directories), font=("Arial", 12), anchor="w", justify="left").pack(fill="x", padx=10)

        tk.Label(new_window, text="Labels and Colors:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
        tk.Label(new_window, text="\n".join(labels), font=("Arial", 12), anchor="w", justify="left").pack(fill="x", padx=10)

    ####
####
    def save_to_file(self):
        """Guarda las variables en un archivo JSON seleccionando la ubicación y nombre."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            initialdir=".",
            title="Save Variables"
        )
        if file_path:
            try:
                # Crear un diccionario con las variables
                data = {
                    "raman_shift_range": [self.min_raman_shift.get(), self.max_raman_shift.get()],
                    
                    "directories": [var.get() for var in self.group_directories],
                    "labels": [var.get() for var in self.group_labels],
                    "colors": [var.get() for var in self.color_order],
                    "title": self.title.get(),  # Guardar el título
                }
                with open(file_path, "w") as file:
                    json.dump(data, file, indent=4)
                tk.messagebox.showinfo("Success", "Variables saved successfully as JSON!")
            except Exception as e:
                tk.messagebox.showerror("Error", f"An error occurred: {e}")



    def open_from_file(self):
        """Abre un archivo JSON, carga las variables y actualiza la interfaz."""
        file_path = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            initialdir=".",
            title="Open Variables"
        )
        if file_path:
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
                # Cargar Raman shift range y título
                raman_shift_range = data.get("raman_shift_range", ["0", "2000"])
                self.min_raman_shift.set(raman_shift_range[0])
                self.max_raman_shift.set(raman_shift_range[1])
                self.title.set(data.get("title", "Default Title"))  # Cargar el título
                
                # Cargar grupos, etiquetas y colores
                directories = data.get("directories", [])
                labels = data.get("labels", [])
                colors = data.get("colors", [])
                self.clear_group_lines()
                for directory, label, color in zip(directories, labels, colors):
                    self.add_group_line_with_values(directory, label, color)
                tk.messagebox.showinfo("Success", "Variables loaded successfully!")
            except Exception as e:
                tk.messagebox.showerror("Error", f"An error occurred while loading the file: {e}")



    def clear_group_lines(self):
        """Elimina todas las filas de grupos actuales."""
        for widget in self.frame.grid_slaves():
            if int(widget.grid_info()["row"]) >= 3:  # Fila 3 en adelante contiene los grupos
                widget.destroy()
        self.group_directories.clear()
        self.group_labels.clear()
        self.color_order.clear()


    def add_group_line_with_values(self, directory, label, color):
        """Añade una fila de grupo con valores predefinidos."""
        row = len(self.group_directories) + 3

        # Botón "Select Folder"
        dir_var = tk.StringVar(value=directory)
        select_folder_button = tk.Button(self.frame, text="Select Folder", command=lambda: self.select_folder(dir_var, status_label))
        select_folder_button.grid(row=row, column=0, padx=5, pady=5)

        # Estado de selección
        status_label = tk.Label(self.frame, text="Selected" if directory else "Not Selected", fg="green" if directory else "red")
        status_label.grid(row=row, column=1, padx=5, pady=5)

        # Entry para nombre del grupo
        group_var = tk.StringVar(value=label)
        tk.Entry(self.frame, textvariable=group_var, width=20).grid(row=row, column=2, padx=5, pady=5)

        # Lista desplegable para seleccionar color
        color_var = tk.StringVar(value=color)
        color_dropdown = ttk.Combobox(self.frame, textvariable=color_var, values=self.color_list, state="readonly", width=10)
        color_dropdown.grid(row=row, column=3, padx=5, pady=5)

        # Botón "X" para eliminar la fila
        tk.Button(self.frame, text="X", command=lambda: self.delete_group_line(row)).grid(row=row, column=4, padx=5, pady=5)

        # Guardar las referencias
        self.group_directories.append(dir_var)
        self.group_labels.append(group_var)
        self.color_order.append(color_var)



if __name__ == "__main__":
    root = tk.Tk()
    app = PCAAnalyserApp(root)
    root.mainloop()

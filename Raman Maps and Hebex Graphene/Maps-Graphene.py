import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import shutil
import subprocess

class FileEntry:
    def __init__(self, filename, size_x=50, size_y=50):
        self.filename = filename
        self.size_x = tk.StringVar(value=str(size_x))
        self.size_y = tk.StringVar(value=str(size_y))

class RamanGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Raman Data Manager")
        self.geometry("600x400")
        self.file_entries = []
        self.selected_index = tk.IntVar(value=-1)
        self.create_widgets()
        self.refresh_list()

    def create_widgets(self):
        # Top buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(btn_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Add Data", command=self.add_file).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Exit", command=self.quit).pack(side=tk.RIGHT, padx=5)

        # List of files
        self.list_frame = tk.Frame(self)
        self.list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Run button
        run_frame = tk.Frame(self)
        run_frame.pack(fill=tk.X, padx=5, pady=10)
        tk.Button(run_frame, text="Run", command=self.run_selected).pack()

    def refresh_list(self):
        # Store previous selection
        prev_selection = self.selected_index.get()
        for widget in self.list_frame.winfo_children():
            widget.destroy()

        if not self.file_entries:
            self.selected_index.set(-1)
            tk.Label(self.list_frame, text="No data loaded.").pack()
            return

        header = tk.Frame(self.list_frame)
        header.pack(fill=tk.X)
        tk.Label(header, text="Select", width=6).pack(side=tk.LEFT)
        tk.Label(header, text="Filename", width=25).pack(side=tk.LEFT)
        tk.Label(header, text="Size X (µm)", width=12).pack(side=tk.LEFT)
        tk.Label(header, text="Size Y (µm)", width=12).pack(side=tk.LEFT)
        tk.Label(header, text="Delete", width=8).pack(side=tk.LEFT)

        for idx, entry in enumerate(self.file_entries):
            row = tk.Frame(self.list_frame)
            row.pack(fill=tk.X, pady=2)
            rbtn = tk.Radiobutton(row, variable=self.selected_index, value=idx)
            rbtn.pack(side=tk.LEFT)
            tk.Label(row, text=os.path.basename(entry.filename), width=25, anchor='w').pack(side=tk.LEFT)

            x_entry = tk.Entry(row, textvariable=entry.size_x, width=8)
            x_entry.pack(side=tk.LEFT, padx=2)
            y_entry = tk.Entry(row, textvariable=entry.size_y, width=8)
            y_entry.pack(side=tk.LEFT, padx=2)
            tk.Label(row, text="µm", width=4).pack(side=tk.LEFT)
            tk.Label(row, text="µm", width=4).pack(side=tk.LEFT)
            del_btn = tk.Button(row, text="X", command=lambda i=idx: self.delete_file(i), fg="red", width=3)
            del_btn.pack(side=tk.LEFT, padx=4)
        
        # Restore selection if possible
        if 0 <= prev_selection < len(self.file_entries):
            self.selected_index.set(prev_selection)
        else:
            self.selected_index.set(-1)

    def add_file(self):
        files = filedialog.askopenfilenames(title="Select data files")
        for f in files:
            if not any(e.filename == f for e in self.file_entries):
                self.file_entries.append(FileEntry(f))
        self.refresh_list()

    def delete_file(self, idx):
        if idx < len(self.file_entries):
            del self.file_entries[idx]
            # Correct selection after deletion
            if len(self.file_entries) == 0:
                self.selected_index.set(-1)
            elif self.selected_index.get() >= len(self.file_entries):
                self.selected_index.set(len(self.file_entries)-1)
            self.refresh_list()

    def save_config(self):
        if not self.file_entries:
            messagebox.showinfo("Save Config", "No data to save.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON Files", "*.json")],
                                                 title="Save Config")
        if not file_path:
            return
        config = []
        for entry in self.file_entries:
            config.append({
                "filename": entry.filename,
                "size_x": entry.size_x.get(),
                "size_y": entry.size_y.get()
            })
        with open(file_path, "w") as f:
            json.dump(config, f, indent=2)
        messagebox.showinfo("Save Config", "Configuration saved.")

    def load_config(self):
        file_path = filedialog.askopenfilename(defaultextension=".json",
                                               filetypes=[("JSON Files", "*.json")],
                                               title="Load Config")
        if not file_path:
            return
        try:
            with open(file_path, "r") as f:
                config = json.load(f)
            self.file_entries = [FileEntry(item["filename"], item.get("size_x", 50), item.get("size_y", 50)) for item in config]
            self.selected_index.set(-1)
            self.refresh_list()
        except Exception as e:
            messagebox.showerror("Error", f"Could not load config:\n{e}")

    def run_selected(self):
        idx = self.selected_index.get()
        if idx == -1 or idx >= len(self.file_entries):
            messagebox.showinfo("Run", "Please select a data row to run.")
            return
        entry = self.file_entries[idx]
        filename = entry.filename
        try:
            size_x = float(entry.size_x.get())
            size_y = float(entry.size_y.get())
        except ValueError:
            messagebox.showerror("Error", "Size X and Size Y must be numeric values.")
            return

        # --- RUN g4.py IN THE SAME DIRECTORY ---
        g4_path = os.path.join(os.path.dirname(__file__), 'g4.py')
        # Detect python or python3
        python_exec = shutil.which('python') or shutil.which('python3') or 'python'
        if not os.path.isfile(g4_path):
            messagebox.showerror("Error", f"Could not find g4.py at {g4_path}")
            return
        try:
            # Launch g4.py with parameters
            subprocess.Popen([python_exec, g4_path, filename, str(size_x), str(size_y)])
            messagebox.showinfo("Run", f"Processing:\n{os.path.basename(filename)}\nSize X: {size_x} µm\nSize Y: {size_y} µm")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run g4.py:\n{e}")

if __name__ == "__main__":
    app = RamanGUI()
    app.mainloop()




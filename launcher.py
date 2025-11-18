# launcher.py
import subprocess
import webbrowser
import tkinter as tk
from tkinter import messagebox
import os
import sys
import threading

streamlit_process = None
SERVER_FLAG = "--server-mode"

# Para encontrar app.py tanto en desarrollo como dentro del .exe
def resource_path(relative_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller descomprime aquí los ficheros
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def _run_streamlit_inprocess(app_path: str, port: int = 8501):
    try:
        from streamlit.web import cli as stcli
    except Exception as e:
        messagebox.showerror("Error", f"No se puede importar Streamlit:\n{e}")
        return

    # Ejecuta streamlit en el proceso actual
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    try:
        # Esto no devuelve hasta que pare el servidor
        stcli.main()
    except SystemExit:
        pass


def start_streamlit():
    global streamlit_process

    if streamlit_process is not None and streamlit_process.poll() is None:
        messagebox.showinfo("Información", "La app ya está en ejecución.")
        return

    app_path = resource_path("mltutor/app.py")

    try:
        if hasattr(sys, "_MEIPASS"):
            # Modo congelado (PyInstaller): relanzamos este mismo ejecutable en modo servidor
            exe = sys.executable
            cmd = [exe, SERVER_FLAG]
        else:
            # Desarrollo: invocamos el intérprete actual con este script en modo servidor
            cmd = [sys.executable, os.path.abspath(__file__), SERVER_FLAG]

        streamlit_process = subprocess.Popen(cmd)
        webbrowser.open("http://localhost:8501")
    except Exception as e:
        messagebox.showerror("Error", f"No se ha podido lanzar Streamlit:\n{e}")


def stop_streamlit_and_exit():
    global streamlit_process

    if streamlit_process is not None and streamlit_process.poll() is None:
        try:
            streamlit_process.terminate()
        except Exception:
            pass
    root.destroy()


# --- Interfaz Tkinter ---
root = tk.Tk()
root.title("Launcher Streamlit")
root.resizable(False, False)

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

label = tk.Label(frame, text="Control de la app Streamlit")
label.pack(pady=(0, 10))

btn_start = tk.Button(frame, text="Start", width=15, command=start_streamlit)
btn_start.pack(pady=5)

btn_stop = tk.Button(frame, text="Stop", width=15, command=stop_streamlit_and_exit)
btn_stop.pack(pady=5)

# Si cierran la ventana con la X, paramos también streamlit
root.protocol("WM_DELETE_WINDOW", stop_streamlit_and_exit)

if __name__ == "__main__":
    # Si nos lanzan con el flag especial, actuamos como servidor Streamlit
    if SERVER_FLAG in sys.argv:
        app_path = resource_path("mltutor/app.py")
        # Ejecutamos Streamlit en este mismo proceso para funcionar en ejecutable congelado
        _run_streamlit_inprocess(app_path, port=8501)
        sys.exit(0)
    else:
        root.mainloop()
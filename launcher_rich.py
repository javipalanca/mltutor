#!/usr/bin/env python
"""
Launcher con interfaz rica en terminal usando Rich.
Requiere: pip install rich
"""
import subprocess
import webbrowser
import time
import sys
import os
import signal
import urllib.request
import threading

# Global process variable
streamlit_process = None
SERVER_FLAG = "--server-mode"


def resource_path(relative_path: str) -> str:
    """Obtiene la ruta correcta tanto en desarrollo como en ejecutable congelado."""
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def _run_streamlit_inprocess(app_path: str, port: int = 8501):
    """Ejecuta Streamlit en el proceso actual (para modo congelado)."""
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        # Fallback para versiones antiguas de streamlit
        try:
            from streamlit import cli as stcli
        except Exception as e:
            print(f"Error importando Streamlit: {e}")
            sys.exit(1)
            
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--global.developmentMode",
        "false",
    ]
    try:
        stcli.main()
    except SystemExit:
        pass


def run_gui(url: str):
    """Ejecuta la interfaz gráfica de control."""
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError:
        return

    root = tk.Tk()
    root.title("MLTutor Launcher")
    root.geometry("300x150")
    root.resizable(False, False)

    def on_open():
        webbrowser.open(url)

    def on_close():
        if streamlit_process:
            streamlit_process.terminate()
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)

    tk.Label(root, text="🧠 MLTutor", font=("Arial", 16, "bold")).pack(pady=(20, 5))
    tk.Label(root, text="El servidor está ejecutándose.", fg="green").pack(pady=5)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    tk.Button(btn_frame, text="Abrir Navegador", command=on_open).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Detener y Salir", command=on_close, bg="#ffcccc").pack(side=tk.LEFT, padx=5)

    # Check server health periodically
    def check_health():
        if streamlit_process and streamlit_process.poll() is not None:
            # Server died
            try:
                messagebox.showerror("Error", "El servidor de Streamlit se ha detenido inesperadamente.")
            except:
                pass
            on_close()
        root.after(2000, check_health)

    check_health()
    try:
        root.mainloop()
    except Exception as e:
        with open(os.path.expanduser("~/mltutor_crash.log"), "w") as f:
            f.write(str(e))
        sys.exit(1)


def wait_for_server(url: str, timeout: int = 60) -> bool:
    """Espera a que el servidor responda 200 OK."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def signal_handler(sig, frame):
    """Maneja la señal de interrupción."""
    print("\nDeteniendo MLTutor...")
    if streamlit_process:
        streamlit_process.terminate()
    print("✓ MLTutor detenido correctamente")
    sys.exit(0)


def main():
    """Inicia Streamlit con interfaz rica o GUI según el entorno."""
    global streamlit_process
    
    # Si nos lanzan con el flag especial, actuamos como servidor Streamlit
    if SERVER_FLAG in sys.argv:
        app_path = resource_path("mltutor/app.py")
        _run_streamlit_inprocess(app_path, port=8501)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    
    app_path = resource_path("mltutor/app.py")
    port = 8501
    url = f"http://localhost:{port}"
    
    # Detectar si debemos usar GUI (sin consola o .app)
    use_gui = False
    if sys.stdout is None or not sys.stdout.isatty():
        use_gui = True
    
    # Si estamos en modo GUI, no usamos Rich
    if use_gui:
        # Iniciar servidor en segundo plano
        env = os.environ.copy()
        if 'USE_GPU' not in env:
            env['USE_GPU'] = '0'
            
        if hasattr(sys, "_MEIPASS"):
            cmd = [sys.executable, SERVER_FLAG]
        else:
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                app_path,
                "--server.port", str(port),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ]

        streamlit_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Esperamos activamente antes de mostrar la GUI para asegurar que arranque
        wait_for_server(f"{url}/_stcore/health", timeout=30)
        webbrowser.open(url)
        
        run_gui(url)
        return

    # Modo Consola (Rich)
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, TextColumn
        
        console = Console()
        
        # Banner
        console.print(Panel.fit(
            "[bold blue]🧠 MLTutor[/bold blue]\n"
            "[dim]Aprende Machine Learning de forma interactiva[/dim]",
            border_style="blue"
        ))
        
        # Detectar configuración de GPU
        use_gpu = os.environ.get('USE_GPU', '0') == '1'
        backend = "GPU (Metal)" if use_gpu else "CPU"
        
        console.print(f"\n[cyan]📊 Servidor:[/cyan] {url}")
        console.print(f"[cyan]⚙️  Backend:[/cyan] {backend}")
        console.print("[dim]Presiona Ctrl+C para detener[/dim]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Iniciando servidor...", total=None)
            
            try:
                # Configurar entorno
                env = os.environ.copy()
                if 'USE_GPU' not in env:
                    env['USE_GPU'] = '0'
                
                # Determinar comando
                if hasattr(sys, "_MEIPASS"):
                    # Modo congelado: relanzamos este mismo ejecutable con flag
                    cmd = [sys.executable, SERVER_FLAG]
                else:
                    # Modo desarrollo: usamos python -m streamlit
                    cmd = [
                        sys.executable, "-m", "streamlit", "run",
                        app_path,
                        "--server.port", str(port),
                        "--server.headless", "true",
                        "--browser.gatherUsageStats", "false"
                    ]

                # Iniciar Streamlit
                streamlit_process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Esperar a que el servidor esté listo
                if wait_for_server(f"{url}/_stcore/health"):
                    progress.update(task, description="[green]Servidor iniciado ✓")
                else:
                    progress.update(task, description="[red]Tiempo de espera agotado")
                    console.print("[red]El servidor tardó demasiado en responder.[/red]")
                
                progress.stop()
                
                # Abrir navegador
                console.print("[green]✓[/green] Abriendo navegador...\n")
                webbrowser.open(url)
                
                console.print("[bold green]MLTutor está funcionando[/bold green]")
                console.print("[dim]El servidor seguirá corriendo. Presiona Ctrl+C cuando termines.[/dim]\n")
                
                # Mantener vivo
                streamlit_process.wait()
                
            except Exception as e:
                console.print(f"[red]❌ Error:[/red] {e}")
                if streamlit_process:
                    streamlit_process.terminate()
                sys.exit(1)
                
    except ImportError:
        print("Rich no está instalado. Ejecutando en modo básico.")
        # Fallback básico si rich falla
        pass


if __name__ == "__main__":
    main()

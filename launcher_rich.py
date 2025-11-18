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
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

console = Console()
streamlit_process = None


def resource_path(relative_path: str) -> str:
    """Obtiene la ruta correcta tanto en desarrollo como en ejecutable congelado."""
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def signal_handler(sig, frame):
    """Maneja la señal de interrupción."""
    console.print("\n[yellow]Deteniendo MLTutor...[/yellow]")
    if streamlit_process:
        streamlit_process.terminate()
        #streamlit_process.wait()
    console.print("[green]✓[/green] MLTutor detenido correctamente")
    sys.exit(0)


def main():
    """Inicia Streamlit con una interfaz rica en terminal."""
    global streamlit_process
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Banner
    console.print(Panel.fit(
        "[bold blue]🧠 MLTutor[/bold blue]\n"
        "[dim]Aprende Machine Learning de forma interactiva[/dim]",
        border_style="blue"
    ))
    
    app_path = resource_path("mltutor/app.py")
    port = 8501
    url = f"http://localhost:{port}"
    
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
            # Por defecto usa CPU (más estable)
            # Para usar GPU: USE_GPU=1 python launcher_rich.py
            if 'USE_GPU' not in env:
                env['USE_GPU'] = '0'
            
            # Iniciar Streamlit
            streamlit_process = subprocess.Popen(
                [
                    sys.executable, "-m", "streamlit", "run",
                    app_path,
                    "--server.port", str(port),
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false"
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Esperar inicio
            time.sleep(2)
            progress.update(task, description="[green]Servidor iniciado ✓")
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
            sys.exit(1)


if __name__ == "__main__":
    main()

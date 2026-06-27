#!/usr/bin/env python
"""
Launcher de MLTutor con interfaz rica en terminal usando Rich.

Funciona tanto en desarrollo (python launcher_rich.py) como dentro de un
ejecutable PyInstaller. En modo congelado no existe un intérprete Python
externo, así que el launcher se relanza a sí mismo con --server-mode y
ejecuta Streamlit en el propio proceso.
"""
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser

SERVER_FLAG = "--server-mode"
DEFAULT_PORT = 8501

streamlit_process = None


def resource_path(relative_path: str) -> str:
    """Obtiene la ruta correcta tanto en desarrollo como en ejecutable congelado."""
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)


def find_free_port(preferred: int = DEFAULT_PORT) -> int:
    """Devuelve el puerto preferido si está libre; si no, uno libre cualquiera."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port: int, timeout: float = 120.0) -> bool:
    """Espera a que el servidor acepte conexiones en el puerto dado."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if streamlit_process is not None and streamlit_process.poll() is not None:
            return False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(0.3)
    return False


def run_server(port: int) -> None:
    """Ejecuta Streamlit en este mismo proceso (necesario en modo congelado)."""
    from streamlit.web import cli as stcli

    app_path = resource_path(os.path.join("mltutor", "app.py"))
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false",
        "--global.developmentMode", "false",
    ]
    try:
        stcli.main()
    except SystemExit:
        pass


def spawn_server(port: int, env: dict) -> subprocess.Popen:
    """Lanza el proceso servidor (este mismo programa con --server-mode)."""
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, SERVER_FLAG, str(port)]
    else:
        cmd = [sys.executable, os.path.abspath(__file__), SERVER_FLAG, str(port)]
    return subprocess.Popen(cmd, env=env)


def main() -> None:
    global streamlit_process

    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()

    def signal_handler(sig, frame):
        console.print("\n[yellow]Deteniendo MLTutor...[/yellow]")
        if streamlit_process and streamlit_process.poll() is None:
            streamlit_process.terminate()
            try:
                streamlit_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                streamlit_process.kill()
        console.print("[green]✓[/green] MLTutor detenido correctamente")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    console.print(Panel.fit(
        "[bold blue]🧠 MLTutor[/bold blue]\n"
        "[dim]Aprende Machine Learning de forma interactiva[/dim]",
        border_style="blue",
    ))

    port = find_free_port()
    url = f"http://localhost:{port}"

    env = os.environ.copy()
    # Por defecto usa CPU (más estable). Para GPU: USE_GPU=1
    env.setdefault("USE_GPU", "0")

    use_gpu = env.get("USE_GPU") == "1"
    backend = "GPU" if use_gpu else "CPU"

    console.print(f"\n[cyan]📊 Servidor:[/cyan] {url}")
    console.print(f"[cyan]⚙️  Backend:[/cyan] {backend}")
    console.print("[dim]Presiona Ctrl+C para detener[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Iniciando servidor...", total=None)
        streamlit_process = spawn_server(port, env)
        started = wait_for_port(port)
        if started:
            progress.update(task, description="[green]Servidor iniciado ✓")
        progress.stop()

    if not started:
        console.print("[red]❌ El servidor no ha podido iniciarse.[/red]")
        sys.exit(1)

    console.print("[green]✓[/green] Abriendo navegador...\n")
    webbrowser.open(url)

    console.print("[bold green]MLTutor está funcionando[/bold green]")
    console.print("[dim]El servidor seguirá corriendo. Presiona Ctrl+C cuando termines.[/dim]\n")

    try:
        streamlit_process.wait()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    # En Windows, PyInstaller necesita esto porque el launcher se relanza a sí mismo
    import multiprocessing
    multiprocessing.freeze_support()

    if SERVER_FLAG in sys.argv:
        idx = sys.argv.index(SERVER_FLAG)
        try:
            server_port = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            server_port = DEFAULT_PORT
        run_server(server_port)
        sys.exit(0)

    main()

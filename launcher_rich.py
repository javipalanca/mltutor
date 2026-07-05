#!/usr/bin/env python
"""
Launcher de MLTutor: app de escritorio que empotra la UI de Streamlit
en una ventana nativa mediante pywebview (WKWebView en macOS, WebView2
en Windows, Qt WebEngine en Linux). Si no hay backend gráfico disponible
cae de forma transparente al modo clásico: abrir el navegador.

Funciona tanto en desarrollo (python launcher_rich.py) como dentro de un
ejecutable PyInstaller. En modo congelado no existe un intérprete Python
externo, así que el launcher se relanza a sí mismo con --server-mode y
ejecuta Streamlit en el propio proceso.

Flags:
  --server-mode PORT   (interno) ejecuta el servidor Streamlit
  --browser            fuerza el modo clásico (navegador + terminal)
"""
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser

SERVER_FLAG = "--server-mode"
BROWSER_FLAG = "--browser"
DEFAULT_PORT = 8501
WINDOW_TITLE = "MLTutor"
WINDOW_SIZE = (1280, 860)
WINDOW_MIN_SIZE = (1000, 700)

streamlit_process = None


def setup_windowed_io() -> None:
    """En un ejecutable windowed (sin consola) stdout/stderr son None.

    Los redirigimos a un fichero de log (~/.mltutor/mltutor.log) para que
    Streamlit/rich no fallen al escribir y se pueda diagnosticar cualquier
    problema.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        log_dir = os.path.join(os.path.expanduser("~"), ".mltutor")
        os.makedirs(log_dir, exist_ok=True)
        log = open(
            os.path.join(log_dir, "mltutor.log"),
            "a",
            buffering=1,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        log = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log


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
        # Ocultar el botón Deploy y las opciones de desarrollador
        "--client.toolbarMode", "minimal",
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


def stop_server(timeout: float = 10.0) -> None:
    """Detiene el proceso servidor si sigue vivo."""
    if streamlit_process and streamlit_process.poll() is None:
        streamlit_process.terminate()
        try:
            streamlit_process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            streamlit_process.kill()


def open_native_window(url: str) -> bool:
    """Muestra la UI en una ventana nativa de escritorio.

    Bloquea hasta que el usuario cierra la ventana. Devuelve False si no
    hay backend gráfico disponible (el llamante hará fallback a navegador).
    """
    try:
        import webview
    except Exception:
        return False

    if sys.platform.startswith("linux"):
        # QtWebEngine no puede usar el sandbox de Chromium dentro de un
        # ejecutable PyInstaller
        if getattr(sys, "frozen", False):
            os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

    window = webview.create_window(
        WINDOW_TITLE,
        url,
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_size=WINDOW_MIN_SIZE,
    )

    # Cierre automático para pruebas (smoke tests): MLTUTOR_WINDOW_TIMEOUT=N
    autoclose = os.environ.get("MLTUTOR_WINDOW_TIMEOUT")

    def _autoclose_worker():
        time.sleep(float(autoclose))
        window.destroy()

    # Icono de la ventana (solo lo usan los backends GTK/Qt; en Windows y
    # macOS el icono sale del ejecutable/bundle)
    icon_path = resource_path(os.path.join("mltutor", "assets", "icon.png"))
    icon_kwargs = {"icon": icon_path} if os.path.exists(icon_path) else {}

    try:
        if autoclose:
            webview.start(_autoclose_worker, **icon_kwargs)
        else:
            webview.start(**icon_kwargs)
        return True
    except Exception:
        return False


def main() -> None:
    global streamlit_process

    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console(file=sys.stdout)
    force_browser = BROWSER_FLAG in sys.argv

    def shutdown(exit_code: int = 0):
        console.print("\n[yellow]Deteniendo MLTutor...[/yellow]")
        stop_server()
        console.print("[green]✓[/green] MLTutor detenido correctamente")
        sys.exit(exit_code)

    def signal_handler(sig, frame):
        shutdown(0)

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    console.print(Panel.fit(
        "[bold blue]🧠 MLTutor[/bold blue]\n"
        "[dim]Aprende Machine Learning de forma interactiva[/dim]",
        border_style="blue",
    ))

    port = find_free_port()
    url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    # Por defecto usa CPU (más estable). Para GPU: USE_GPU=1
    env.setdefault("USE_GPU", "0")

    use_gpu = env.get("USE_GPU") == "1"
    backend = "GPU" if use_gpu else "CPU"

    console.print(f"\n[cyan]📊 Servidor:[/cyan] {url}")
    console.print(f"[cyan]⚙️  Backend:[/cyan] {backend}\n")

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
        stop_server()
        sys.exit(1)

    if not force_browser:
        console.print("[green]✓[/green] Abriendo MLTutor...\n")
        if open_native_window(url):
            # El usuario ha cerrado la ventana: apagar el servidor y salir
            shutdown(0)
        console.print(
            "[yellow]No hay entorno gráfico compatible; usando el navegador.[/yellow]"
        )

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

    setup_windowed_io()

    if SERVER_FLAG in sys.argv:
        idx = sys.argv.index(SERVER_FLAG)
        try:
            server_port = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            server_port = DEFAULT_PORT
        run_server(server_port)
        sys.exit(0)

    main()

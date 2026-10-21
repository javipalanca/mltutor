"""Desktop entry point: python -m mltutor.desktop."""

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(
        description="MLTutor — aplicación de escritorio portable"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Comprobar ventana y pantallas y salir",
    )
    parser.add_argument(
        "--verify-models",
        action="store_true",
        help="Verificar entrenamiento y exportaciones del ejecutable",
    )
    parser.add_argument(
        "--screenshot", type=Path, help="Guardar una captura de la ventana inicial"
    )
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("USE_GPU", "0")
    # Bound CPU use so training remains usable on classroom laptops.
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    if getattr(sys, "frozen", False):
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    if sys.stdout is None or sys.stderr is None:
        log_dir = Path.home() / ".mltutor"
        try:
            log_dir.mkdir(exist_ok=True)
            log = (log_dir / "mltutor.log").open("a", encoding="utf-8", buffering=1)
        except OSError:
            log = open(os.devnull, "w")
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log
    if args.verify_models:
        from .checks import verify_models

        verify_models()
        return 0
    # Apple's Accelerate backend must initialise NumPy on the main thread;
    # importing it first inside QThread can crash in NumPy's macOS BLAS check.
    import numpy  # noqa: F401 -- initialise native BLAS on the main thread
    import matplotlib

    matplotlib.use("Agg")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from .window import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("MLTutor")
    app.setOrganizationName("MLTutor")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    if args.smoke_test or args.screenshot:
        pages = (
            iter(
                [
                    "🌲 Árboles de Decisión",
                    "📊 Regresión",
                    "🔍 K-Nearest Neighbors",
                    "🧠 Redes Neuronales",
                    "📁 Cargar CSV Personalizado",
                ]
            )
            if args.smoke_test
            else iter([])
        )

        def ready():
            if window.failed_message:
                print(window.failed_message, file=sys.stderr)
                app.exit(1)
                return
            if args.screenshot:

                def capture():
                    window.grab().save(str(args.screenshot))
                    app.exit(0)

                QTimer.singleShot(800, capture)
                return
            page = next(pages, None)
            if page:
                window.session.state.navigation = page
                QTimer.singleShot(0, window.evaluate)
            else:
                print("MLTutor Qt: todas las pantallas abren correctamente.")
                app.exit(0)

        window.page_ready.connect(ready)
        QTimer.singleShot(180000, lambda: app.exit(2))
    result = app.exec()
    if window.worker is not None:
        window.session.cancel.set()
        window.worker.wait()
        app.processEvents()
    window.close()
    window.deleteLater()
    app.processEvents()
    return result


if __name__ == "__main__":
    raise SystemExit(main())

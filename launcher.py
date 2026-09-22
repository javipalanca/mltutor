# launcher.py
import sys
import os
import subprocess
import webbrowser
import time
import urllib.request
import threading
import logging
import multiprocessing

# Configurar logging
log_file = os.path.expanduser("~/mltutor_debug.log")
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(process)d - %(levelname)s - %(message)s'
)

SERVER_FLAG = "--server-mode"
SERVER_PORT = 8501
SERVER_URL = f"http://localhost:{SERVER_PORT}"
streamlit_process = None

def resource_path(relative_path: str) -> str:
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def _run_streamlit_inprocess(app_path: str, port: int = 8501):
    logging.info(f"Starting Streamlit in-process: {app_path}")
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        try:
            from streamlit import cli as stcli
        except Exception as e:
            logging.error(f"Error importing Streamlit: {e}")
            print(f"Error importing Streamlit: {e}")
            return

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
        logging.info("Calling stcli.main()")
        stcli.main()
    except SystemExit as e:
        logging.info(f"Streamlit exited with code: {e}")
    except Exception as e:
        logging.error(f"Streamlit crashed: {e}", exc_info=True)

def is_server_running(url: str, timeout: int = 1) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False

def run_gui():
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                   QHBoxLayout, QLabel, QPushButton, QMessageBox, QFrame)
    from PySide6.QtCore import QTimer, Qt, QThread, Signal, Slot
    from PySide6.QtGui import QColor, QPainter, QBrush, QFont

    class StatusLight(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setFixedSize(20, 20)
            self.color = QColor("red")

        def set_color(self, color_name):
            self.color = QColor(color_name)
            self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QBrush(self.color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(0, 0, 20, 20)

    class ServerCheckThread(QThread):
        status_checked = Signal(bool)

        def run(self):
            running = is_server_running(f"{SERVER_URL}/_stcore/health")
            self.status_checked.emit(running)

    class LauncherWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("MLTutor Launcher")
            self.setFixedSize(350, 220)

            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)
            layout.setContentsMargins(20, 20, 20, 20)

            # Title
            title_label = QLabel("🧠 MLTutor")
            title_font = QFont("Arial", 18, QFont.Bold)
            title_label.setFont(title_font)
            title_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(title_label)

            # Status
            status_layout = QHBoxLayout()
            self.status_light = StatusLight()
            self.status_label = QLabel("Servidor detenido")
            status_layout.addStretch()
            status_layout.addWidget(self.status_light)
            status_layout.addWidget(self.status_label)
            status_layout.addStretch()
            layout.addLayout(status_layout)

            # Buttons
            btn_layout = QHBoxLayout()
            self.btn_start = QPushButton("Iniciar App")
            self.btn_start.clicked.connect(self.start_streamlit)
            self.btn_stop = QPushButton("Detener")
            self.btn_stop.clicked.connect(self.stop_streamlit)
            self.btn_stop.setEnabled(False)
            
            btn_layout.addWidget(self.btn_start)
            btn_layout.addWidget(self.btn_stop)
            layout.addLayout(btn_layout)

            self.btn_open = QPushButton("Abrir en Navegador")
            self.btn_open.clicked.connect(lambda: webbrowser.open(SERVER_URL))
            self.btn_open.setEnabled(False)
            layout.addWidget(self.btn_open)

            # Timer for status checking
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.check_status)
            self.timer.start(2000)
            
            self.check_thread = ServerCheckThread()
            self.check_thread.status_checked.connect(self.on_status_checked)

            # Start automatically
            QTimer.singleShot(100, self.start_streamlit)

        def check_status(self):
            global streamlit_process
            if streamlit_process is not None and streamlit_process.poll() is None:
                 if not self.check_thread.isRunning():
                     self.check_thread.start()
            else:
                self.update_ui_state("stopped")

        def on_status_checked(self, is_running):
            if is_running:
                self.update_ui_state("running")
            else:
                self.update_ui_state("starting")

        def update_ui_state(self, state):
            if state == "running":
                self.status_light.set_color("#00e676") # Green
                self.status_label.setText("Servidor activo")
                self.btn_start.setEnabled(False)
                self.btn_stop.setEnabled(True)
                self.btn_open.setEnabled(True)
            elif state == "starting":
                self.status_light.set_color("#ffea00") # Yellow
                self.status_label.setText("Iniciando...")
                self.btn_start.setEnabled(False)
                self.btn_stop.setEnabled(True)
                self.btn_open.setEnabled(False)
            else: # stopped
                self.status_light.set_color("#ff1744") # Red
                self.status_label.setText("Servidor detenido")
                self.btn_start.setEnabled(True)
                self.btn_stop.setEnabled(False)
                self.btn_open.setEnabled(False)

        def start_streamlit(self):
            global streamlit_process
            if streamlit_process is not None and streamlit_process.poll() is None:
                return

            logging.info("Starting Streamlit process...")
            self.update_ui_state("starting")
            
            app_path = resource_path("mltutor/app.py")
            env = os.environ.copy()
            if 'USE_GPU' not in env:
                env['USE_GPU'] = '0'

            try:
                if hasattr(sys, "_MEIPASS"):
                    exe = sys.executable
                    cmd = [exe, SERVER_FLAG]
                else:
                    cmd = [sys.executable, os.path.abspath(__file__), SERVER_FLAG]
                
                logging.info(f"Command: {cmd}")

                streamlit_process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logging.info(f"Streamlit process started with PID: {streamlit_process.pid}")
                
                # Wait and open browser in background
                threading.Thread(target=self._wait_and_open, daemon=True).start()
                
            except Exception as e:
                logging.error(f"Failed to start Streamlit: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"No se ha podido lanzar Streamlit:\n{e}")
                self.update_ui_state("stopped")

        def _wait_and_open(self):
            for _ in range(60):
                if is_server_running(f"{SERVER_URL}/_stcore/health"):
                    logging.info("Server is ready, opening browser")
                    webbrowser.open(SERVER_URL)
                    return
                time.sleep(0.5)
            logging.warning("Timed out waiting for server")

        def stop_streamlit(self):
            global streamlit_process
            if streamlit_process:
                logging.info(f"Stopping Streamlit process PID: {streamlit_process.pid}")
                streamlit_process.terminate()
                streamlit_process = None
            self.update_ui_state("stopped")

        def closeEvent(self, event):
            logging.info("Closing launcher")
            self.stop_streamlit()
            event.accept()

    app = QApplication(sys.argv)
    window = LauncherWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    multiprocessing.freeze_support()
    logging.info(f"Launcher started. Args: {sys.argv}")
    
    if SERVER_FLAG in sys.argv:
        logging.info("Running in SERVER mode")
        app_path = resource_path("mltutor/app.py")
        _run_streamlit_inprocess(app_path, port=SERVER_PORT)
        sys.exit(0)
    else:
        logging.info("Running in GUI mode")
        run_gui()
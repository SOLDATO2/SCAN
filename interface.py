import sys
import os
import subprocess
import cv2
import torch

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QSpinBox, QFileDialog, QTabWidget, QProgressBar, QHBoxLayout, QVBoxLayout, QMessageBox, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QFileInfo
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl
# from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtGui import QFont


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from subprocess import Popen
import time

# importa suas funções de interpolação
import adicionar_it

class QtMediaPlayerWidget(QWidget):
    def __init__(self, video_path, parent=None):
        super().__init__(parent)

        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        video_widget = QVideoWidget()
        video_widget.setMinimumSize(640, 480)

        layout = QVBoxLayout()
        layout.addWidget(video_widget)
        self.setLayout(layout)

        self.media_player.setVideoOutput(video_widget)
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))

    def play(self):
        self.media_player.play()

    def pause(self):
        self.media_player.pause()

    def stop(self):
        self.media_player.stop()

    def set_position(self, position):
        self.media_player.setPosition(position)

    def duration(self):
        return self.media_player.duration()

    def position(self):
        return self.media_player.position()


class InterpolationThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, float)

    def __init__(self, model_path, video_path, interp_factor, output_path):
        super().__init__()
        self.model_path = model_path
        self.video_path = video_path
        self.interp_factor = interp_factor
        self.output_path = output_path

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = adicionar_it.SCAN_EncDec(nf_start=32).to(device)
        ckpt = torch.load(self.model_path, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)
        model.eval()

        transform = adicionar_it.get_transform()
        cap = cv2.VideoCapture(self.video_path)
        fps_meta = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        n_pairs = len(frames) - 1
        new_frames = [frames[0]]
        for i in range(n_pairs):
            mids = adicionar_it.recursive_interpolate(
                model, frames[i], frames[i+1],
                self.interp_factor - 1, transform, device
            )
            new_frames.extend(mids + [frames[i+1]])
            self.progress.emit(int((i + 1) / n_pairs * 100))

        duration = len(frames) / fps_meta
        fps_calc = len(new_frames) / duration
        max_fps = 60.0
        include_audio = fps_calc <= max_fps
        out_fps = min(fps_calc, max_fps)

        adicionar_it.create_video(new_frames, self.output_path, out_fps)

        if include_audio:
            tmp = self.output_path + ".tmp.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-i", self.output_path,
                "-i", self.video_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest", tmp
            ]
            subprocess.run(cmd, check=True)
            os.replace(tmp, self.output_path)

        self.progress.emit(100)
        self.finished.emit(self.output_path, out_fps)

class DraggableButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():  # Verifica se o conteúdo arrastado é um arquivo
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            files = [u.toLocalFile() for u in event.mimeData().urls()]
            if files:  # Se houver arquivos, use o primeiro
                self.parent().model_path = files[0]  # Atualiza o caminho do modelo
                filename = QFileInfo(files[0]).fileName()
                self.setText(filename)  # Atualiza o texto do botão com o caminho do arquivo

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interpolação de Vídeo – SCAN_EncDec")
        self.resize(900, 600)
        self.setAcceptDrops(True)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        cfg = QWidget()
        layout = QVBoxLayout(cfg)
        layout.setSpacing(8)

        # Seção modelo
        self.model_path = None
        model_box = QVBoxLayout()
        model_box.setAlignment(Qt.AlignmentFlag.AlignTop)

        model_title = QLabel("Modelo")
        model_title.setFont(QFont("Helvetica", 16, QFont.DemiBold))
        # model_title.setMargin(4)
        model_title.setStyleSheet("color: #434343;")
        model_box.addWidget(model_title)


        btn1 = DraggableButton("Arrastar ou clicar para selecionar modelo", self)
        btn1.setMinimumHeight(80)
        btn1.setMinimumWidth(300)
        # btn1.setMaximumWidth(700)
        btn1.clicked.connect(lambda: self.select_model(btn1))
        btn1.setStyleSheet("background-color: #2b7fff; color: white; border-radius: 8px;")
        btn1.setFont(QFont("Helvetica", 12))
        model_box.addWidget(btn1)

        layout.addLayout(model_box)

        # Seção vídeo de entrada
        self.video_path = None
        video_box = QVBoxLayout()
        video_box.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        video_box.setSpacing(8)

        video_title = QLabel("Vídeo de Entrada")
        video_title.setFont(QFont("Helvetica", 16, QFont.DemiBold))
        video_title.setStyleSheet("color: #434343;")
        video_box.addWidget(video_title)

        btn2 = DraggableButton("Arrastar ou clicar para selecionar vídeo", self)
        btn2.setMinimumHeight(80)
        btn2.setMinimumWidth(300)
        # btn2.setMaximumWidth(700)
        btn2.setStyleSheet("background-color: #2b7fff; color: white; border-radius: 8px;")
        btn2.setFont(QFont("Helvetica", 12))
        btn2.clicked.connect(lambda: self.select_video(btn2))
        video_box.addWidget(btn2)
        

        self.info_label = QLabel("Resolução: —    FPS: —")
        self.info_label.setFont(QFont("Helvetica", 12))
        self.info_label.setStyleSheet("color: #434343;")
        video_box.addWidget(self.info_label)

        h4 = QHBoxLayout()
        interpolation_label = QLabel("Taxa de Interpolação:")
        interpolation_label.setFont(QFont("Helvetica", 12))
        interpolation_label.setStyleSheet("color: #434343;")
        h4.addWidget(interpolation_label)
        self.spin = QSpinBox()
        self.spin.setRange(2, 10)
        self.spin.setValue(2)
        self.spin.setStyleSheet("padding: 8px; border-radius: 8px; border: 1px solid #ccc;")
        self.spin.setFont(QFont("Helvetica", 12))
        h4.addWidget(self.spin)
        video_box.addLayout(h4)

        layout.addLayout(video_box)

        # Seção saída
        h3 = QVBoxLayout()
        h3.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        h3label = QLabel("Caminho de Saída")
        h3label.setFont(QFont("Helvetica", 16, QFont.DemiBold))
        h3label.setStyleSheet("color: #434343;")
        h3.addWidget(h3label)

        h3_row = QHBoxLayout()
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Diretório")
        self.output_path.setStyleSheet("padding: 8px; border-radius: 8px; border: 1px solid #ccc;")
        self.output_path.setFont(QFont("Helvetica", 12))
        h3_row.addWidget(self.output_path)

        btn3 = QPushButton("Selecionar Diretório")
        btn3.clicked.connect(self.select_output)
        btn3.setStyleSheet("background-color: #2b7fff; color: white; border-radius: 4px; padding: 8px;")
        btn3.setFont(QFont("Helvetica", 12))
        h3_row.addWidget(btn3)

        h3.addLayout(h3_row)
        layout.addLayout(h3)

        # Botão iniciar e progresso
        btn_interp_container = QHBoxLayout()
        self.btn_interp = QPushButton("Interpolar")
        self.btn_interp.setStyleSheet("background-color: #2b7fff; color: white; border-radius: 4px; padding: 12px;")
        self.btn_interp.setFont(QFont("Helvetica", 12))
        self.btn_interp.setMinimumWidth(240)
        self.btn_interp.clicked.connect(self.start_interpolation)
        btn_interp_container.addWidget(self.btn_interp)
        btn_interp_container.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addLayout(btn_interp_container)
        self.progress = QProgressBar()
        self.progress.setHidden(True)
        layout.addWidget(self.progress)
        

        self.tabs.addTab(cfg, "Configurações")

    def select_model(self, button):
        path, _ = QFileDialog.getOpenFileName(self, "Selecione o modelo", "", "Pytorch (*.pth *.tar)")
        if path:
            self.model_path = path
            filename = QFileInfo(path).fileName()
            button.setText(filename)  # Atualiza o texto do botão com o caminho do arquivo

    def select_video(self, button):
        path, _ = QFileDialog.getOpenFileName(self, "Selecione o vídeo", "", "Vídeos (*.mp4 *.avi *.mov)")
        if path:
            self.video_path = path
            cap = cv2.VideoCapture(path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            self.info_label.setText(f"Resolução: {w}×{h}    FPS: {fps:.2f}")
            filename = QFileInfo(path).fileName()
            button.setText(filename)  # Atualiza o texto do botão com o caminho do arquivo

    def select_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Salvar como", "interpolado.mp4", "Vídeo MP4 (*.mp4)")
        if path:
            self.output_path.setText(path)

    def start_interpolation(self):
        if not os.path.isfile(self.model_path) or not os.path.isfile(self.video_path):
            QMessageBox.warning(self, "Erro", "Modelo ou vídeo inválido.")
            return
        if not self.output_path.text().strip():
            QMessageBox.warning(self, "Erro", "Especifique o arquivo de saída.")
            return

        self.btn_interp.setEnabled(False)
        self.thread = InterpolationThread(
            self.model_path,
            self.video_path,
            self.spin.value(),
            self.output_path.text().strip()
        )
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

        self.progress.setHidden(False)

    def on_finished(self, interpolated_path):
        if hasattr(self, 'thread'):
            self.thread.quit()
            self.thread.wait()

        result = QWidget()
        vlayout = QVBoxLayout(result)
        hlayout = QHBoxLayout()
        player1 = QtMediaPlayerWidget(self.video_path.text())
        player2 = QtMediaPlayerWidget(interpolated_path)
        hlayout.addWidget(player1)
        hlayout.addWidget(player2)
        vlayout.addLayout(hlayout)

        btn_play = QPushButton("▶ Play Ambos")
        btn_play.clicked.connect(lambda: (player1.play(), player2.play()))
        btn_pause = QPushButton("⏸ Pause Ambos")
        btn_pause.clicked.connect(lambda: (player1.pause(), player2.pause()))

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 0)
        vlayout.addWidget(QLabel("Controle de Tempo"))
        vlayout.addWidget(slider)

        def sync_slider():
            duration = min(player1.duration(), player2.duration())
            slider.setRange(0, duration)
            slider.setValue(min(player1.position(), player2.position()))

        def slider_moved(value):
            player1.set_position(value)
            player2.set_position(value)

        player1.media_player.positionChanged.connect(sync_slider)
        player2.media_player.positionChanged.connect(sync_slider)
        slider.sliderMoved.connect(slider_moved)

        # Slider de volume
        slider_volume = QSlider(Qt.Horizontal)
        slider_volume.setRange(0, 100)
        slider_volume.setValue(50)  # valor inicial
        vlayout.addWidget(QLabel("Controle de Volume"))
        vlayout.addWidget(slider_volume)

        def slider_volume_moved(value):
            player1.media_player.setVolume(value)
            player2.media_player.setVolume(value)

        slider_volume.valueChanged.connect(slider_volume_moved)

        vlayout.addWidget(btn_play)
        vlayout.addWidget(btn_pause)

        self.tabs.addTab(result, "Resultado")
        self.tabs.setCurrentWidget(result)
        self.btn_interp.setEnabled(True)

    # def dragEnterEvent(self, event):
    #     if event.mimeData().hasUrls():
    #         event.accept()
    #     else:
    #         event.ignore()

    # def dropEvent(self, event):
    #     files = [u.toLocalFile() for u in event.mimeData().urls()]
    #     for f in files:
    #         print(f)

    def closeEvent(self, event):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        super().closeEvent(event)


class ReloadHandler(FileSystemEventHandler):
    def __init__(self, script_path):
        super().__init__()
        self.script_path = script_path
        self.process = None
        self.start_app()

    def start_app(self):
        if self.process:
            self.process.terminate()
        self.process = Popen([sys.executable, self.script_path])

    def on_modified(self, event):
        if event.src_path.endswith("interface.py"):
            print(f"Arquivo modificado: {event.src_path}. Reiniciando...")
            self.start_app()

def enable_hot_reload():
    script_path = os.path.abspath(__file__)
    event_handler = ReloadHandler(script_path)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(script_path), recursive=False)
    observer.start()
    print("Hot Reloading ativado. Monitorando alterações em interface.py...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    args = adicionar_it.get_args()

    if(args.reload):
        enable_hot_reload()
    else:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())

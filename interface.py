import sys
import os
import subprocess
import cv2
import torch

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QSpinBox, QFileDialog, QTabWidget, QProgressBar, QHBoxLayout, QVBoxLayout, QMessageBox, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interpolação de Vídeo – SCAN_EncDec")
        self.resize(900, 600)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        cfg = QWidget()
        layout = QVBoxLayout(cfg)

        # Seção modelo
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Modelo:"))
        self.model_path = QLineEdit()
        h1.addWidget(self.model_path)
        btn1 = QPushButton("…")
        btn1.clicked.connect(self.select_model)
        h1.addWidget(btn1)
        layout.addLayout(h1)

        # Seção vídeo de entrada
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Vídeo:"))
        self.video_path = QLineEdit()
        h2.addWidget(self.video_path)
        btn2 = QPushButton("…")
        btn2.clicked.connect(self.select_video)
        h2.addWidget(btn2)
        layout.addLayout(h2)

        self.info_label = QLabel("Resolução: —    FPS: —")
        layout.addWidget(self.info_label)

        # Seção saída
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Saída:"))
        self.output_path = QLineEdit()
        h3.addWidget(self.output_path)
        btn3 = QPushButton("…")
        btn3.clicked.connect(self.select_output)
        h3.addWidget(btn3)
        layout.addLayout(h3)

        # Taxa de interpolação
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("Taxa de Interpolação:"))
        self.spin = QSpinBox()
        self.spin.setRange(2, 10)
        self.spin.setValue(2)
        h4.addWidget(self.spin)
        layout.addLayout(h4)

        # Botão iniciar e progresso
        self.btn_interp = QPushButton("Interpolar")
        self.btn_interp.clicked.connect(self.start_interpolation)
        layout.addWidget(self.btn_interp)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.tabs.addTab(cfg, "Configurações")

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecione o modelo", "", "Pytorch (*.pth *.tar)")
        if path:
            self.model_path.setText(path)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Selecione o vídeo", "", "Vídeos (*.mp4 *.avi *.mov)")
        if path:
            self.video_path.setText(path)
            cap = cv2.VideoCapture(path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            self.info_label.setText(f"Resolução: {w}×{h}    FPS: {fps:.2f}")

    def select_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Salvar como", "interpolado.mp4", "Vídeo MP4 (*.mp4)")
        if path:
            self.output_path.setText(path)

    def start_interpolation(self):
        if not os.path.isfile(self.model_path.text()) or not os.path.isfile(self.video_path.text()):
            QMessageBox.warning(self, "Erro", "Modelo ou vídeo inválido.")
            return
        if not self.output_path.text().strip():
            QMessageBox.warning(self, "Erro", "Especifique o arquivo de saída.")
            return

        self.btn_interp.setEnabled(False)
        self.thread = InterpolationThread(
            self.model_path.text(),
            self.video_path.text(),
            self.spin.value(),
            self.output_path.text().strip()
        )
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

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

    def closeEvent(self, event):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

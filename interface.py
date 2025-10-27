import sys
import os
import subprocess
from typing import Final
import cv2
import torch

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QSpinBox, QFileDialog, QTabWidget, QProgressBar, QHBoxLayout, QVBoxLayout, QMessageBox, QSlider, QGraphicsOpacityEffect
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QFileInfo, QTimer, QRect, QPoint
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QFont, QIcon, QPainter, QColor, QBrush, QLinearGradient


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from subprocess import Popen
import time

# importa suas funções de interpolação
import adicionar_it

SPINBOX_STYLE: Final[str] = """
    QSpinBox {
        padding: 8px; 
        border-radius: 8px;
        border: 1px solid #ccc;
    }
    QSpinBox::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 28px;
        border-top-right-radius: 8px;
        border-bottom: 1px solid #ccc;
        border-left: 1px solid #ccc;
        padding-top: 2px;
    }
    QSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 28px;
        border-bottom-right-radius: 8px;
        border-top: 1px solid #ccc;
        border-left: 1px solid #ccc;
        padding-bottom: 1px;
    }
    QSpinBox::up-arrow {
        image: url('./assets/chevron-up.png');
        width: 20px;
        height: 20px;
    }
    QSpinBox::down-arrow {
        image: url('./assets/chevron-down.png');
        width: 20px;
        height: 20px;
    }
"""

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
        device = adicionar_it.get_device()
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
            if self.isInterruptionRequested():
                self.progress.emit(0)
                return  # Sai do método, encerrando a thread
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
        self.setMinimumHeight(80)
        self.setMinimumWidth(300)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("background-color: #2b7fff; color: white; border-radius: 8px;")
        self.setFont(QFont("Helvetica", 12))

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

class AnimatedProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.offset = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)

    def setValue(self, value):
        super().setValue(value)
        # self.offset = 0  # Reinicia o ciclo do brilho ao mudar o progresso

    def update_animation(self):
        self.offset = (self.offset + 6) % 1000
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        painter.setRenderHint(QPainter.Antialiasing)
        # Fundo
        painter.setBrush(QColor("#e6f0ff"))
        painter.setPen(QColor("#2b7fff"))
        painter.drawRoundedRect(rect, 8, 8)

        # Chunk animado (indeterminado) com listras diagonais customizadas
        if self.minimum() == 0 and self.maximum() == 0:
            chunk_width = rect.width() // 3
            x = self.offset - chunk_width
            chunk_rect = QRect(x, 0, chunk_width, rect.height())
            painter.save()
            painter.setClipRect(chunk_rect)
            stripe_w = 20
            color1 = QColor("#2b7fff")
            color2 = QColor("#90caf9")  # cor secundária da listra
            for i, sx in enumerate(range(-stripe_w*2, chunk_rect.width()+stripe_w*2, stripe_w)):
                points = [
                    QPoint(sx + int(self.offset/2), 0),
                    QPoint(sx + stripe_w, 0),
                    QPoint(sx + stripe_w//2, rect.height()),
                    QPoint(sx - stripe_w//2, rect.height())
                ]
                painter.setBrush(QBrush(color1 if i % 2 == 0 else color2))
                painter.setPen(Qt.NoPen)
                painter.drawPolygon(*points)
            # ...dentro do método paintEvent, após o for das listras...
            glow_width = int(chunk_rect.width() * 0.8)
            glow_x = x + (self.offset % (chunk_rect.width() - glow_width)) if chunk_rect.width() > glow_width else x
            glow_rect = QRect(glow_x, 0, glow_width, chunk_rect.height())

            gradient = QLinearGradient(glow_rect.left(), 0, glow_rect.right(), 0)
            gradient.setColorAt(0.0, QColor(255, 255, 255, 0))
            gradient.setColorAt(0.5, QColor(255, 255, 255, 200))  # Centro mais brilhante
            gradient.setColorAt(1.0, QColor(255, 255, 255, 0))

            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.NoPen)
            painter.drawRect(glow_rect)
            painter.restore()
        else:
            # Barra normal
            progress = (self.value() - self.minimum()) / (self.maximum() - self.minimum()) if self.maximum() > self.minimum() else 0
            chunk_rect = QRect(rect)
            chunk_rect.setWidth(int(rect.width() * progress))
            painter.setBrush(QColor("#2b7fff"))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(chunk_rect, 8, 8)

            # Efeito luminoso percorre toda a barra, mas aparece só na parte preenchida
            if chunk_rect.width() > 0 and self.value() < 100:
                painter.save()
                painter.setClipRect(chunk_rect)  # Limita o efeito à parte preenchida
                bar_width = rect.width()         # Usa a barra toda para o ciclo do brilho
                glow_width = int(bar_width * 0.2)
                cycle = max(1, bar_width + glow_width)
                glow_x = (self.offset % cycle) - glow_width
                glow_rect = QRect(glow_x, 0, glow_width, rect.height())

                gradient = QLinearGradient(glow_rect.left(), 0, glow_rect.right(), 0)
                gradient.setColorAt(0.0, QColor(255, 255, 255, 0))
                gradient.setColorAt(0.2, QColor(255, 255, 255, 0))
                gradient.setColorAt(0.5, QColor(255, 255, 255, 150))
                gradient.setColorAt(0.8, QColor(255, 255, 255, 0))
                gradient.setColorAt(1.0, QColor(255, 255, 255, 0))

                painter.setBrush(QBrush(gradient))
                painter.setPen(Qt.NoPen)
                painter.drawRect(glow_rect)
                painter.restore()

        # Texto centralizado (apenas se não for indeterminado)
        if not (self.minimum() == 0 and self.maximum() == 0):
            painter.setPen(QColor("white" if self.value() > 50 else "#2b7fff"))
            painter.setFont(QFont("Helvetica", 14, QFont.Medium))
            text = f"{self.value() if self.value() >= 0 else 0}%"
            painter.drawText(rect, Qt.AlignCenter, text)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interpolação de Vídeo – SCAN_EncDec")
        self.resize(1000, 800)
        self.setAcceptDrops(True)
        self.opacity_effect = QGraphicsOpacityEffect()
        # TODO: Trocar por um ícone mais apropriado
        self.setWindowIcon(QIcon("./assets/3-d-cube.svg"))

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        cfg = QWidget()
        layout = QVBoxLayout(cfg)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 32, 16, 24)

        # Seção modelo
        self.model_path = None
        model_box = QVBoxLayout()
        model_box.setAlignment(Qt.AlignmentFlag.AlignTop)

        model_title = QLabel("Modelo")
        model_title.setFont(QFont("Helvetica", 16, QFont.DemiBold))
        model_title.setStyleSheet("color: #434343;")
        model_box.addWidget(model_title)

        model_subtitle = QLabel("Selecione o modelo de interpolação que será usado para interpolar o vídeo.")
        model_subtitle.setFont(QFont("Helvetica", 12, QFont.Light))
        model_subtitle.setStyleSheet("color: #99a1af;")
        model_box.addWidget(model_subtitle)

        btn1 = DraggableButton("Arraste ou clique para selecionar o modelo (.pth, .tar)", self)       
        btn1.clicked.connect(lambda: self.select_model(btn1))
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

        video_subtitle = QLabel("Selecione o vídeo que será interpolado.")
        video_subtitle.setFont(QFont("Helvetica", 12, QFont.Light))
        video_subtitle.setStyleSheet("color: #99a1af;")
        video_box.addWidget(video_subtitle)

        btn2 = DraggableButton("Arraste ou clique para selecionar o vídeo (.mp4, .avi, .mov)", self)
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
        self.spin = QSpinBox(self)
        self.spin.setRange(2, 10)
        self.spin.setValue(2)
        self.spin.setStyleSheet(SPINBOX_STYLE)
        self.spin.setFont(QFont("Helvetica", 12))
        self.spin.setToolTip("Número de quadros interpolados entre cada quadro original.")
        h4.addWidget(self.spin)
        video_box.addLayout(h4)

        layout.addLayout(video_box)

        # Seção saída
        h3 = QVBoxLayout()
        h3.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        h3label = QLabel("Diretório de Saída")
        h3label.setFont(QFont("Helvetica", 16, QFont.DemiBold))
        h3label.setStyleSheet("color: #434343;")
        h3.addWidget(h3label)

        h3_subtitle = QLabel("Selecione o diretório em que o vídeo interpolado será enviado.")
        h3_subtitle.setFont(QFont("Helvetica", 12, QFont.Light))
        h3_subtitle.setStyleSheet("color: #99a1af;")
        h3.addWidget(h3_subtitle)

        h3_row = QHBoxLayout()
        h3_row.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Diretório")
        self.output_path.setStyleSheet("padding: 10px; border-radius: 8px; border: 1px solid #ccc;")
        self.output_path.setFont(QFont("Helvetica", 12))
        h3_row.addWidget(self.output_path)

        btn3 = QPushButton("Selecionar Diretório")
        btn3.clicked.connect(self.select_output)
        btn3.setStyleSheet("background-color: #2b7fff; color: white; border-radius: 4px; padding: 12px;")
        btn3.setFont(QFont("Helvetica", 12))
        btn3.setCursor(Qt.CursorShape.PointingHandCursor)
        h3_row.addWidget(btn3)

        # Resolução de saída
        h_res = QHBoxLayout()
        h_res.setAlignment(Qt.AlignmentFlag.AlignLeft)
        res_label = QLabel("Resolução de Saída:")
        res_label.setFont(QFont("Helvetica", 12))
        res_label.setStyleSheet("color: #434343;")
        h_res.addWidget(res_label)
        self.spinWidth = QSpinBox(self)
        self.spinWidth.setRange(1, 10000)
        self.spinWidth.setEnabled(False)
        self.spinWidth.setStyleSheet(SPINBOX_STYLE)
        self.spinWidth.setFont(QFont("Helvetica", 12))
        self.spinWidth.setFixedWidth(256)
        self.spinWidth.setToolTip("Largura do vídeo interpolado.")
        h_res.addWidget(self.spinWidth)

        x_label = QLabel("×")
        x_label.setFont(QFont("Helvetica", 12))
        x_label.setStyleSheet("color: #434343;")
        h_res.addWidget(x_label)
        self.spinHeight = QSpinBox(self)
        self.spinHeight.setRange(1, 10000)
        self.spinHeight.setEnabled(False)
        self.spinHeight.setStyleSheet(SPINBOX_STYLE)
        self.spinHeight.setFont(QFont("Helvetica", 12))
        self.spinHeight.setFixedWidth(256)
        self.spinHeight.setToolTip("Altura do vídeo interpolado.")
        h_res.addWidget(self.spinHeight)
        self.show()

        h3.addLayout(h3_row)
        h3.addLayout(h_res)
        layout.addLayout(h3)

        # Botão iniciar e progresso
        last_section = QVBoxLayout()
        last_section.setSpacing(16)
        last_section.setContentsMargins(0, 16, 0, 0)
        btn_interp_container = QHBoxLayout()
        self.btn_interp = QPushButton("Interpolar")
        self.btn_interp.setStyleSheet("background-color: #2b7fff; color: white; border-radius: 4px; padding: 12px;")
        self.btn_interp.setFont(QFont("Helvetica", 12))
        self.btn_interp.setMinimumWidth(240)
        self.btn_interp.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_interp.clicked.connect(self.start_interpolation)
        btn_interp_container.addWidget(self.btn_interp)
        btn_interp_container.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        last_section.addLayout(btn_interp_container)
       
        self.progress = AnimatedProgressBar()
        self.progress.setHidden(True)

        last_section.addWidget(self.progress)

        layout.addLayout(last_section)

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
             # Atualiza resolução de saída
            self.spinWidth.setMaximum(w)
            self.spinHeight.setMaximum(h)
            self.spinWidth.setValue(w)
            self.spinHeight.setValue(h)
            self.spinWidth.setEnabled(True)
            self.spinHeight.setEnabled(True)

    def select_output(self):
        if not self.video_path:
            QMessageBox.warning(self, "Erro", "Selecione primeiro um vídeo de entrada.")
            return
        input_ext = os.path.splitext(self.video_path)[1].lower()
        filters = "Vídeo MP4 (*.mp4);;Vídeo AVI (*.avi);;Vídeo MOV (*.mov)"
        default_filter = "Vídeo MP4 (*.mp4)"
        if input_ext == ".avi":
            default_filter = "Vídeo AVI (*.avi)"
        elif input_ext == ".mov":
            default_filter = "Vídeo MOV (*.mov)"
        default_name = f"interpolado{input_ext if input_ext in ['.mp4','.avi','.mov'] else '.mp4'}"
        path, _ = QFileDialog.getSaveFileName(self, "Salvar como", default_name, filters, default_filter)
        if path:
            self.output_path.setText(path)

    def start_interpolation(self):
        if not os.path.isfile(self.model_path) or not os.path.isfile(self.video_path):
            QMessageBox.warning(self, "Erro", "Modelo ou vídeo inválido.")
            return
        if not self.output_path.text().strip():
            QMessageBox.warning(self, "Erro", "Especifique o arquivo de saída.")
            return

        self.opacity_effect.setOpacity(0.5)
        self.btn_interp.setGraphicsEffect(self.opacity_effect)
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
        player1 = QtMediaPlayerWidget(self.video_path, self)
        player2 = QtMediaPlayerWidget(interpolated_path, self)
        hlayout.addWidget(player1)
        hlayout.addWidget(player2)
        vlayout.addLayout(hlayout)

        # Sliders de volume
        volume_slider_layout = QHBoxLayout()
        volume_slider_column1 = QVBoxLayout()
        slider1_label = QLabel("Controle de Volume - Player 1")
        slider1_label.setFont(QFont("Helvetica", 12, QFont.Light))
        volume_slider_column1.addWidget(slider1_label)

        slider_volume = QSlider(Qt.Horizontal)
        slider_volume.setRange(0, 100)
        slider_volume.setValue(50)  # valor inicial
        volume_slider_column1.addWidget(slider_volume)
        volume_slider_layout.addLayout(volume_slider_column1)

        volume_slider_column2 = QVBoxLayout()
        slider2_label = QLabel("Controle de Volume - Player 2")
        slider2_label.setFont(QFont("Helvetica", 12, QFont.Light))
        volume_slider_column2.addWidget(slider2_label)

        slider_volume2 = QSlider(Qt.Horizontal)
        slider_volume2.setRange(0, 100)
        slider_volume2.setValue(50)  # valor inicial
        volume_slider_column2.addWidget(slider_volume2)
        volume_slider_layout.addLayout(volume_slider_column2)

        vlayout.addLayout(volume_slider_layout)

        def slider_volume_moved1(value):
            player1.media_player.setVolume(value)

        def slider_volume_moved2(value):
            player2.media_player.setVolume(value)

        slider_volume.valueChanged.connect(slider_volume_moved1)
        slider_volume2.valueChanged.connect(slider_volume_moved2)

        # Slider de tempo
        time_layout = QHBoxLayout()
        time_column = QVBoxLayout()
        time_label = QLabel("Controle de Tempo")
        time_label.setFont(QFont("Helvetica", 12, QFont.Light))
        time_column.addWidget(time_label)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 0)
        time_column.addWidget(slider)
        time_layout.addLayout(time_column)

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

        buttons_row = QHBoxLayout()
        btn_play = QPushButton("Play Ambos")
        btn_play.setIcon(QIcon("./assets/play_icon.png"))  # Adicione um ícone de play
        btn_play.setStyleSheet("background-color: #2b7fff; color: white; border-radius: 4px; padding: 9px;")
        btn_play.setFont(QFont("Helvetica", 11, QFont.Medium))
        btn_play.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_play.clicked.connect(lambda: (player1.play(), player2.play()))

        btn_pause = QPushButton("Pause Ambos")
        btn_pause.setIcon(QIcon("./assets/pause_icon.png"))  # Adicione um ícone de pause
        btn_pause.setStyleSheet("color: #2b7fff; border-radius: 4px; padding: 8px; border: 1px solid #2b7fff;")
        btn_pause.setFont(QFont("Helvetica", 11, QFont.Medium))
        btn_pause.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_pause.clicked.connect(lambda: (player1.pause(), player2.pause()))

        buttons_row.addWidget(btn_play)
        buttons_row.addWidget(btn_pause)

        time_layout.addLayout(buttons_row)
        vlayout.addLayout(time_layout)

        self.opacity_effect.setOpacity(1.0)
        self.btn_interp.setGraphicsEffect(self.opacity_effect)
        self.btn_interp.setEnabled(True)        
        self.tabs.addTab(result, "Apuração")
        self.tabs.setCurrentWidget(result)

    def closeEvent(self, event):
        self.cancel_interpolation()  
        if hasattr(self, 'thread') and hasattr(self.thread, 'isRunning') and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        super().closeEvent(event)
    
    def cancel_interpolation(self):
        if hasattr(self, 'thread') and hasattr(self.thread, 'isRunning') and self.thread.isRunning():
            self.thread.requestInterruption()


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

import sys, json, traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QFileDialog, QProgressBar, QLabel, QMessageBox
)

from core.audio import extract_audio
from core.asr import Transcriber
from core.categorize import classify_text

class WorkerSignals(QObject):
    log = Signal(str)
    progress = Signal(int)
    finished = Signal(dict)
    failed = Signal(str)

class PipelineWorker(QRunnable):
    def __init__(self, video_path: str, transcriber: Transcriber):
        super().__init__()
        self.video_path = video_path
        self.transcriber = transcriber
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.log.emit("Starting pipeline...")
            out_dir = Path("outputs") / Path(self.video_path).stem
            out_dir.mkdir(parents=True, exist_ok=True)

            wav_path = str(out_dir / "audio.wav")
            self.signals.log.emit("Step 1/3: Extracting audio...")
            self.signals.progress.emit(20)
            extract_audio(self.video_path, wav_path)

            self.signals.log.emit("Step 2/3: Transcribing (FA/EN)...")
            self.signals.progress.emit(60)
            asr = self.transcriber.transcribe(wav_path)

            self.signals.log.emit("Step 3/3: Classifying content...")
            self.signals.progress.emit(85)
            pred = classify_text(asr["text"])

            result = {"video": self.video_path, "asr": asr, "prediction": pred}
            (out_dir / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            self.signals.progress.emit(100)
            self.signals.finished.emit(result)
        except Exception:
            self.signals.failed.emit(traceback.format_exc())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Content Analyzer (FA/EN)")

        self.pool = QThreadPool()
        self.transcriber = Transcriber(model_size="medium")  # برای سیستم ضعیف: "small"

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self.btn = QPushButton("Select Video and Analyze")
        self.btn.clicked.connect(self.select_video_and_run)

        self.status = QLabel("Ready.")
        self.progress = QProgressBar()
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        layout.addWidget(self.btn)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addWidget(self.log)

        self.resize(900, 650)

    def select_video_and_run(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.mkv *.mov *.avi)"
        )
        if not path:
            return

        self.log.clear()
        self.progress.setValue(0)
        self.status.setText("Running...")

        worker = PipelineWorker(path, self.transcriber)
        worker.signals.log.connect(self.on_log)
        worker.signals.progress.connect(self.progress.setValue)
        worker.signals.finished.connect(self.on_finished)
        worker.signals.failed.connect(self.on_failed)
        self.pool.start(worker)

    def on_log(self, msg: str):
        self.log.append(msg)

    def on_finished(self, result: dict):
        label = result["prediction"]["label"]
        conf = result["prediction"]["confidence"]
        lang = result["asr"]["language"]
        self.status.setText("Done.")
        QMessageBox.information(self, "Done", f"Language: {lang}\nCategory: {label}\nConfidence: {conf:.2f}\nSaved: outputs/.../result.json")
        self.log.append("\n--- Transcript ---\n" + result["asr"]["text"][:5000])

    def on_failed(self, err: str):
        self.status.setText("Failed.")
        QMessageBox.critical(self, "Error", err)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

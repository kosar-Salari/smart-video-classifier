"""
app_production.py
نسخه تمیز و بهبود یافته:
- همیشه ASR از طریق eboo (حذف UI انتخاب سرویس/توکن)
- توکن از ENV: EBOO_API_TOKEN
- نمایش نتیجه دسته‌بندی در مدال (Dialog)
- بهبود شدید ظاهر (QSS), راست‌به‌چپ, UX بهتر
"""

import os
from dotenv import load_dotenv
load_dotenv()

import sys
import json
import traceback
import subprocess
import wave
from pathlib import Path
from datetime import datetime

from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool, Qt
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QFileDialog, QProgressBar, QLabel, QMessageBox,
    QGroupBox, QCheckBox, QHBoxLayout, QTabWidget, QSplitter,
    QSpinBox, QDialog, QDialogButtonBox, QFormLayout, QFrame
)

from core.audio import extract_audio
from core.eboo_api import EbooClient, EbooAPIError

try:
    from core.production_categorizer import ProductionCategorizer, CATEGORIES_DETAILED
    ML_AVAILABLE = True
except Exception as e:
    ML_AVAILABLE = False
    CATEGORIES_DETAILED = {}
    print(f"ML categorizer not available: {e}")


# =========================
# تنظیمات ثابت برنامه
# =========================
APP_TITLE = "سیستم تحلیل ویدیو (نسخه تولیدی)"
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# توکن را از ENV بگیر (کاربر وارد نمی‌کند)
EBOO_API_TOKEN = os.getenv("EBOO_API_TOKEN", "").strip()
EBOO_LANGUAGE = "fa"  # شما گفتید همیشه با eboo کار می‌کنیم و فارسی است


def get_wav_duration_seconds(wav_path: str) -> int:
    with wave.open(wav_path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            return 0
        return int(frames / float(rate))


def split_wav_ffmpeg(input_wav: str, out_dir: Path, segment_sec: int) -> list[str]:
    """
    Segment wav into N-second chunks using ffmpeg.
    Produces: out_dir/seg_000.wav, seg_001.wav, ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "seg_%03d.wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_wav,
        "-f", "segment",
        "-segment_time", str(int(segment_sec)),
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        pattern,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return sorted([str(p) for p in out_dir.glob("seg_*.wav")])


# =========================
# Dialog: نتیجه دسته‌بندی
# =========================
class CategoryDialog(QDialog):
    def __init__(self, categories: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("نتیجه دسته‌بندی")
        self.setModal(True)
        self.setMinimumWidth(520)

        main = QVBoxLayout(self)

        header = QLabel("نتیجه نهایی دسته‌بندی محتوا")
        header.setObjectName("DialogHeader")
        main.addWidget(header)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setObjectName("Divider")
        main.addWidget(line)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        label = categories.get("label") or "unknown"
        label_fa = categories.get("label_fa") or label
        conf = categories.get("confidence", 0)

        lbl_label = QLabel(label_fa)
        lbl_label.setObjectName("CategoryPrimary")

        try:
            conf_txt = f"{float(conf):.2f}"
        except Exception:
            conf_txt = str(conf)

        lbl_conf = QLabel(conf_txt)
        lbl_conf.setObjectName("CategoryConfidence")

        form.addRow("دسته:", lbl_label)
        form.addRow("اطمینان:", lbl_conf)

        main.addLayout(form)

        top = categories.get("top_categories")
        if isinstance(top, list) and top:
            main.addWidget(QLabel("گزینه‌های پیشنهادی (Top):"))

            top_box = QTextEdit()
            top_box.setReadOnly(True)
            top_box.setFixedHeight(140)

            lines = []
            for item in top[:8]:
                try:
                    name, score = item
                    name_fa = name
                    if isinstance(CATEGORIES_DETAILED, dict) and name in CATEGORIES_DETAILED:
                        name_fa = CATEGORIES_DETAILED[name].name_fa
                    lines.append(f"- {name_fa}  |  {float(score):.2f}")
                except Exception:
                    continue

            top_box.setPlainText("\n".join(lines).strip())
            main.addWidget(top_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        main.addWidget(buttons)


# =========================
# Worker
# =========================
class WorkerSignals(QObject):
    progress = Signal(int)
    progress_detail = Signal(str)
    log = Signal(str)
    result = Signal(dict)
    error = Signal(str)


class EnhancedPipelineWorker(QRunnable):
    def __init__(self, video_path: str, output_dir: Path, settings: dict):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.settings = settings
        self.signals = WorkerSignals()

    def run(self):
        try:
            if not EBOO_API_TOKEN:
                raise RuntimeError(
                    "توکن eboo تنظیم نشده است. لطفاً Environment Variable به نام EBOO_API_TOKEN را تنظیم کنید."
                )

            out_dir = self.output_dir / f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # مرحله 1: استخراج صدا
            wav_path = str(out_dir / "audio.wav")
            self.signals.log.emit("مرحله ۱/۳: استخراج صدا")
            self.signals.progress.emit(10)
            self.signals.progress_detail.emit("در حال استخراج صدا از ویدیو...")

            extract_audio(self.video_path, wav_path)
            self.signals.log.emit("صدا استخراج شد.")
            self.signals.progress.emit(20)

            # مرحله 2: ASR از طریق eboo (همیشه)
            self.signals.log.emit("مرحله ۲/۳: تبدیل صوت به متن (eboo)")
            self.signals.progress.emit(25)

            segment_sec = int(self.settings.get("segment_sec", 120))
            normalize = bool(self.settings.get("normalize", True))

            client = EbooClient(token=EBOO_API_TOKEN)
            duration_sec = get_wav_duration_seconds(wav_path)
            self.signals.log.emit(f"مدت فایل صوتی: {duration_sec} ثانیه")

            # چک اعتبار
            credit = client.checkcredit()
            audio_credit = int(credit.get("AudioTranscribeCredit", "0") or 0)
            self.signals.log.emit(f"اعتبار صوتی (ثانیه): {audio_credit}")

            if audio_credit and audio_credit < duration_sec:
                raise RuntimeError(
                    f"اعتبار کافی نیست. مدت فایل {duration_sec} ثانیه است ولی اعتبار {audio_credit} ثانیه است."
                )

            # قطعه‌بندی
            segments = [wav_path]
            if duration_sec > segment_sec:
                self.signals.progress_detail.emit("در حال قطعه‌بندی صوت برای پایداری...")
                seg_dir = out_dir / "audio_segments"
                segments = split_wav_ffmpeg(wav_path, seg_dir, segment_sec)

            full_text_parts: list[str] = []
            api_meta: list[dict] = []

            for idx, seg in enumerate(segments, start=1):
                self.signals.progress_detail.emit(f"ارسال قطعه {idx}/{len(segments)} به سرور...")
                self.signals.log.emit(f"پردازش قطعه {idx}/{len(segments)}")

                try:
                    add_resp = client.addfile_by_upload(seg)
                except EbooAPIError as e:
                    raise RuntimeError(
                        "ثبت/آپلود به eboo شکست خورد. اگر تکرار شد، احتمالاً سمت سرور مشکل است "
                        "یا باید روش filelink استفاده شود.\n"
                        f"جزئیات: {e}"
                    )

                filetoken = add_resp.get("FileToken")
                if not filetoken:
                    raise RuntimeError(f"FileToken دریافت نشد: {add_resp}")

                client.convert_audio(filetoken=filetoken, language=EBOO_LANGUAGE, resetdata=False)
                out_resp = client.wait_for_audio_text(filetoken=filetoken, poll_interval_sec=3.0, max_wait_sec=30 * 60)

                raw_text = (out_resp.get("Output") or "").strip()
                if not raw_text:
                    raise RuntimeError(f"خروجی متن قطعه {idx} خالی است: {out_resp}")

                text = raw_text
                if normalize:
                    try:
                        from core.advanced_asr import EnhancedPersianNormalizer
                        text = EnhancedPersianNormalizer().normalize(raw_text)
                    except Exception:
                        text = raw_text

                full_text_parts.append(text)
                api_meta.append({
                    "segment_index": idx,
                    "segment_path": seg,
                    "add_resp": add_resp,
                    "output_resp": out_resp,
                    "filetoken": filetoken,
                })

                # حذف فایل از سرور (best-effort)
                try:
                    client.deletefile(filetoken)
                except Exception:
                    pass

                # پیشرفت تقریبی
                base = 25
                span = 45
                self.signals.progress.emit(base + int(span * (idx / max(len(segments), 1))))

            merged_text = "\n".join([p.strip() for p in full_text_parts if p.strip()]).strip()
            asr = {
                "language": EBOO_LANGUAGE,
                "text": merged_text,
                "word_count": len(merged_text.split()),
                "duration": duration_sec,
                "transcription_quality": "api",
                "api_segments_meta": api_meta,
            }

            self.signals.log.emit("متن دریافت شد.")
            self.signals.progress.emit(70)

            # مرحله 3: دسته‌بندی
            self.signals.log.emit("مرحله ۳/۳: دسته‌بندی محتوا")
            self.signals.progress_detail.emit("در حال تحلیل محتوا...")
            self.signals.progress.emit(80)

            if ML_AVAILABLE and self.settings.get("use_ml", True):
                categorizer = ProductionCategorizer(use_ml=True)
                pred = categorizer.classify(asr["text"], segments=full_text_parts)

            else:
                pred = {"label": "other", "label_fa": "سایر", "confidence": 0.1}

            self.signals.progress.emit(92)

            result = {
                "input_video": self.video_path,
                "output_dir": str(out_dir),
                "audio_file": wav_path,
                "transcript": asr,
                "categories": pred,
                "timestamp": datetime.now().isoformat(),
            }

            with open(out_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            with open(out_dir / "transcript.txt", "w", encoding="utf-8") as f:
                f.write(asr["text"])

            self.signals.progress.emit(100)
            self.signals.progress_detail.emit("تمام شد.")
            self.signals.log.emit(f"خروجی در: {out_dir}")

            self.signals.result.emit(result)

        except subprocess.CalledProcessError as e:
            self.signals.error.emit(
                "خطا در اجرای ffmpeg. لطفاً نصب بودن ffmpeg و قرار داشتن آن در PATH را بررسی کنید.\n\n"
                + str(e) + "\n\n" + traceback.format_exc()
            )
        except Exception as e:
            self.signals.error.emit(f"خطا: {str(e)}\n\n{traceback.format_exc()}")


# =========================
# Settings Panel (ساده و تمیز)
# =========================
class SettingsPanel(QGroupBox):
    def __init__(self):
        super().__init__("تنظیمات پردازش")
        layout = QVBoxLayout()

        self.normalize_check = QCheckBox("نرمال‌سازی متن فارسی")
        self.normalize_check.setChecked(True)
        layout.addWidget(self.normalize_check)

        self.ml_check = QCheckBox("دسته‌بندی هوشمند (ML)")
        self.ml_check.setChecked(True)
        layout.addWidget(self.ml_check)

        seg_row = QHBoxLayout()
        seg_row.addWidget(QLabel("قطعه‌بندی صوت (ثانیه):"))
        self.segment_spin = QSpinBox()
        self.segment_spin.setRange(30, 600)
        self.segment_spin.setValue(120)
        seg_row.addWidget(self.segment_spin)
        layout.addLayout(seg_row)

        hint = QLabel("نکته: قطعه‌بندی کمتر، پایداری بیشتر در API و کاهش خطاهای 500.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.setLayout(layout)

    def get_settings(self) -> dict:
        return {
            "normalize": self.normalize_check.isChecked(),
            "use_ml": self.ml_check.isChecked(),
            "segment_sec": self.segment_spin.value(),
        }


# =========================
# Results Panel
# =========================
class ResultsPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.tabs = QTabWidget()

        self.transcript_text = QTextEdit()
        self.transcript_text.setReadOnly(True)
        self.transcript_text.setFont(QFont("Tahoma", 10))
        self.tabs.addTab(self.transcript_text, "متن")

        self.json_text = QTextEdit()
        self.json_text.setReadOnly(True)
        self.json_text.setFont(QFont("Consolas", 9))
        self.tabs.addTab(self.json_text, "JSON")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def update_results(self, result: dict):
        self.transcript_text.setPlainText(result.get("transcript", {}).get("text", ""))
        self.json_text.setPlainText(json.dumps(result, ensure_ascii=False, indent=2))


# =========================
# Main Window
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1280, 820)
        self.setLayoutDirection(Qt.RightToLeft)

        self.threadpool = QThreadPool()
        self.video_path = None
        self.last_output_dir = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Top bar
        top = QHBoxLayout()

        self.select_btn = QPushButton("انتخاب ویدیو")
        self.select_btn.clicked.connect(self.select_video)
        self.select_btn.setObjectName("PrimaryButton")
        top.addWidget(self.select_btn)

        self.process_btn = QPushButton("شروع پردازش")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        self.process_btn.setObjectName("SuccessButton")
        top.addWidget(self.process_btn)

        self.open_output_btn = QPushButton("باز کردن پوشه خروجی")
        self.open_output_btn.clicked.connect(self.open_output_folder)
        self.open_output_btn.setEnabled(False)
        self.open_output_btn.setObjectName("SecondaryButton")
        top.addWidget(self.open_output_btn)

        top.addStretch(1)
        main_layout.addLayout(top)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.settings_panel = SettingsPanel()
        left_layout.addWidget(self.settings_panel)

        log_group = QGroupBox("گزارش پردازش")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setObjectName("LogBox")
        log_layout.addWidget(self.log_text)
        left_layout.addWidget(log_group)

        splitter.addWidget(left_widget)

        self.results_panel = ResultsPanel()
        splitter.addWidget(self.results_panel)

        splitter.setSizes([420, 860])
        main_layout.addWidget(splitter)

        # Progress
        progress_group = QGroupBox("وضعیت")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("آماده")
        self.progress_label.setObjectName("StatusLabel")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        main_layout.addWidget(progress_group)

        # Apply theme
        self.apply_styles()

        # Token warning (non-blocking)
        if not EBOO_API_TOKEN:
            self.log_text.append("هشدار: توکن eboo تنظیم نشده است. ENV: EBOO_API_TOKEN")

    def apply_styles(self):
        self.setFont(QFont("Tahoma", 10))
        qss = """
        QWidget { background: #0f172a; color: #e5e7eb; }
        QGroupBox {
            border: 1px solid #24324a; border-radius: 12px;
            margin-top: 12px; padding: 10px;
            background: #111b31;
        }
        QGroupBox::title {
            subcontrol-origin: margin; right: 12px; padding: 0 8px;
            color: #cbd5e1; font-weight: 700;
        }
        QTabWidget::pane { border: 1px solid #24324a; border-radius: 12px; }
        QTabBar::tab {
            background: #0b1224; border: 1px solid #24324a; padding: 8px 14px;
            border-top-left-radius: 10px; border-top-right-radius: 10px;
            margin-left: 6px;
        }
        QTabBar::tab:selected { background: #111b31; }
        QTextEdit {
            background: #0b1224; border: 1px solid #24324a; border-radius: 10px;
            padding: 10px;
        }
        QProgressBar {
            border: 1px solid #24324a; border-radius: 10px; text-align: center;
            background: #0b1224; height: 18px;
        }
        QProgressBar::chunk { background: #22c55e; border-radius: 10px; }
        QLabel#StatusLabel { color: #cbd5e1; padding: 2px 6px; }
        QLabel#Hint { color: #94a3b8; }
        QTextEdit#LogBox { font-family: Consolas; }
        QPushButton {
            border: 1px solid #24324a; border-radius: 12px;
            padding: 10px 14px; background: #0b1224;
        }
        QPushButton:hover { background: #0e1830; }
        QPushButton:disabled { color: #6b7280; border-color: #1f2a40; }
        QPushButton#PrimaryButton { background: #1d4ed8; border-color: #1d4ed8; }
        QPushButton#PrimaryButton:hover { background: #1e40af; }
        QPushButton#SuccessButton { background: #16a34a; border-color: #16a34a; }
        QPushButton#SuccessButton:hover { background: #15803d; }
        QPushButton#SecondaryButton { background: #334155; border-color: #334155; }
        QPushButton#SecondaryButton:hover { background: #2b3647; }

        QLabel#DialogHeader { font-size: 16px; font-weight: 800; color: #e2e8f0; padding: 4px 2px; }
        QFrame#Divider { color: #24324a; }
        QLabel#CategoryPrimary { font-size: 14px; font-weight: 800; color: #22c55e; }
        QLabel#CategoryConfidence { font-weight: 700; color: #cbd5e1; }
        """
        self.setStyleSheet(qss)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "انتخاب فایل ویدیو",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*)"
        )
        if path:
            self.video_path = path
            name = Path(path).name
            self.select_btn.setText(f"ویدیو: {name}")
            self.process_btn.setEnabled(True)
            self.log_text.append(f"ویدیو انتخاب شد: {path}")

    def start_processing(self):
        if not self.video_path:
            return

        settings = self.settings_panel.get_settings()

        worker = EnhancedPipelineWorker(self.video_path, OUTPUTS_DIR, settings)
        worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.progress_detail.connect(self.progress_label.setText)
        worker.signals.log.connect(self.log_text.append)
        worker.signals.result.connect(self.on_result)
        worker.signals.error.connect(self.on_error)

        self.process_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.open_output_btn.setEnabled(False)

        self.progress_bar.setValue(0)
        self.progress_label.setText("شروع پردازش...")

        self.threadpool.start(worker)

    def on_result(self, result: dict):
        self.results_panel.update_results(result)

        self.process_btn.setEnabled(True)
        self.select_btn.setEnabled(True)

        out_dir = result.get("output_dir")
        self.last_output_dir = out_dir
        self.open_output_btn.setEnabled(bool(out_dir))

        # نمایش مدال دسته‌بندی
        dlg = CategoryDialog(result.get("categories", {}) or {}, self)
        dlg.exec()

        QMessageBox.information(self, "موفق", f"پردازش کامل شد.\n\nخروجی:\n{out_dir}")

    def on_error(self, error: str):
        self.process_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.progress_label.setText("خطا")
        QMessageBox.critical(self, "خطا", error)

    def open_output_folder(self):
        if not self.last_output_dir:
            return
        path = str(self.last_output_dir)

        # Cross-platform open folder
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # noqa
            elif sys.platform.startswith("darwin"):
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception:
            QMessageBox.warning(self, "هشدار", "باز کردن پوشه خروجی ممکن نشد.")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

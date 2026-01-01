"""
app_production.py
نسخه حرفه‌ای بهبود یافته
"""
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime

from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool, Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QFileDialog, QProgressBar, QLabel, QMessageBox,
    QComboBox, QGroupBox, QCheckBox, QHBoxLayout, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QSpinBox
)
from PySide6.QtGui import QFont

from core.audio import extract_audio

try:
    from core.advanced_asr import transcribe_advanced
    ASR_AVAILABLE = True
except Exception as e:
    ASR_AVAILABLE = False
    print(f"⚠️ advanced_asr not available: {e}")

try:
    from core.production_categorizer import ProductionCategorizer, CATEGORIES_DETAILED
    ML_AVAILABLE = True
except Exception as e: 
    ML_AVAILABLE = False
    CATEGORIES_DETAILED = {}
    print(f"⚠️ production_categorizer not available: {e}")


class WorkerSignals(QObject):
    log = Signal(str)
    progress = Signal(int)
    progress_detail = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)


class EnhancedPipelineWorker(QRunnable):
    """Worker پیشرفته با گزارش‌دهی بهتر"""
    
    def __init__(self, video_path:  str, settings: dict):
        super().__init__()
        self.video_path = video_path
        self.settings = settings
        self.signals = WorkerSignals()

    def run(self):
        try:
            start_time = datetime. now()
            self.signals.log. emit("🚀 شروع پردازش...")
            self.signals.log.emit(f"   📁 فایل:  {Path(self.video_path).name}")
            
            out_dir = Path("outputs") / Path(self.video_path).stem
            out_dir. mkdir(parents=True, exist_ok=True)

            # ═══ مرحله 1: استخراج صدا ═══
            wav_path = str(out_dir / "audio.wav")
            self.signals.log. emit("\n📹 مرحله 1/3: استخراج صدا...")
            self.signals.progress. emit(10)
            self.signals.progress_detail.emit("در حال استخراج صدا از ویدیو...")
            
            extract_audio(self. video_path, wav_path)
            self.signals.log.emit("   ✓ صدا استخراج شد")
            self.signals. progress.emit(20)

            # ═══ مرحله 2: رونویسی ═══
            model_size = self.settings['model_size']
            self.signals.log. emit(f"\n🎤 مرحله 2/3: رونویسی با Whisper ({model_size})...")
            self.signals.progress_detail.emit(f"در حال بارگذاری مدل {model_size}...")
            self.signals.progress. emit(25)
            
            if ASR_AVAILABLE: 
                asr = transcribe_advanced(
                    wav_path,
                    model_size=model_size,
                    enable_normalization=self.settings['normalize'],
                    beam_size=self.settings. get('beam_size', 5),
                )
            else: 
                from core.asr import Transcriber
                transcriber = Transcriber(model_size=model_size)
                asr = transcriber.transcribe(wav_path)
            
            word_count = asr.get('word_count', len(asr['text']. split()))
            quality = asr.get('transcription_quality', 'unknown')
            
            self.signals.log.emit(f"   ✓ رونویسی کامل شد")
            self.signals.log.emit(f"   📊 تعداد کلمات:  {word_count}")
            self.signals.log.emit(f"   📊 کیفیت:  {quality}")
            self.signals.log.emit(f"   📊 زبان: {asr.get('language', 'fa')} ({asr.get('language_probability', 0):.1%})")
            self.signals.progress.emit(70)

            # ═══ مرحله 3: دسته‌بندی ═══
            self. signals.log.emit("\n🏷️  مرحله 3/3: دسته‌بندی محتوا...")
            self.signals. progress_detail.emit("در حال تحلیل و دسته‌بندی...")
            
            if ML_AVAILABLE: 
                categorizer = ProductionCategorizer(use_ml=self.settings.get('use_ml', True))
                pred = categorizer.classify(asr["text"])
            else: 
                pred = {"label": "other", "label_fa": "سایر", "confidence": 0.1}
            
            self.signals. log.emit(f"   ✓ دسته‌بندی:  {pred.get('label_fa', pred['label'])}")
            self.signals. log.emit(f"   📊 اطمینان: {pred['confidence']:.1%}")
            self.signals.log.emit(f"   📊 روش: {pred. get('method', 'unknown')}")
            
            # نمایش دسته‌های احتمالی
            if 'top_categories' in pred and len(pred['top_categories']) > 1:
                self.signals.log.emit("   📊 سایر احتمالات:")
                for cat, score in pred['top_categories'][1:4]:
                    if score > 0.05:
                        cat_info = CATEGORIES_DETAILED.get(cat)
                        cat_fa = cat_info.name_fa if cat_info else cat
                        self.signals.log.emit(f"      • {cat_fa}:  {score:.1%}")
            
            self.signals.progress.emit(90)

            # ═══ ذخیره نتایج ═══
            elapsed = (datetime.now() - start_time).total_seconds()
            
            result = {
                "video":  self.video_path,
                "processed_at": datetime. now().isoformat(),
                "processing_time_seconds":  round(elapsed, 1),
                "asr": asr,
                "prediction": pred,
                "settings": self. settings
            }
            
            result_path = out_dir / "result.json"
            result_path.write_text(
                json. dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            
            # ذخیره فایل متنی جداگانه
            text_path = out_dir / "transcript.txt"
            text_path.write_text(asr["text"], encoding="utf-8")
            
            self.signals.log.emit(f"\n💾 ذخیره شد:")
            self.signals.log.emit(f"   • {result_path}")
            self.signals.log.emit(f"   • {text_path}")
            self.signals. log.emit(f"\n⏱️  زمان پردازش:  {elapsed:.1f} ثانیه")
            
            self.signals. progress.emit(100)
            self.signals.finished.emit(result)

        except Exception: 
            self.signals.failed.emit(traceback.format_exc())


class SettingsPanel(QGroupBox):
    """پنل تنظیمات پیشرفته"""
    
    def __init__(self):
        super().__init__("⚙️ تنظیمات")
        
        layout = QVBoxLayout()
        
        # ═══ انتخاب مدل Whisper ═══
        model_group = QGroupBox("مدل Whisper")
        model_layout = QVBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "large-v3 (بهترین کیفیت - ~3GB - کند)",
            "medium (متوسط - ~1.5GB - سریع‌تر)",
            "small (سریع - ~500MB - کیفیت کمتر)",
            "base (خیلی سریع - ~150MB - کیفیت پایین)"
        ])
        self.model_combo. setCurrentIndex(0)
        model_layout.addWidget(self.model_combo)
        
        # Beam size
        beam_layout = QHBoxLayout()
        beam_layout.addWidget(QLabel("Beam Size (دقت بیشتر = کندتر):"))
        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(1, 10)
        self.beam_spin. setValue(5)
        beam_layout. addWidget(self. beam_spin)
        model_layout.addLayout(beam_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # ═══ تنظیمات پردازش ═══
        process_group = QGroupBox("پردازش متن")
        process_layout = QVBoxLayout()
        
        self.normalize_check = QCheckBox("تصحیح خودکار متن فارسی (hazm + parsivar)")
        self.normalize_check.setChecked(True)
        self.normalize_check. setToolTip("تصحیح نیم‌فاصله، حروف عربی به فارسی، و اشتباهات رایج")
        process_layout.addWidget(self.normalize_check)
        
        process_group.setLayout(process_layout)
        layout.addWidget(process_group)
        
        # ═══ تنظیمات دسته‌بندی ═══
        cat_group = QGroupBox("دسته‌بندی")
        cat_layout = QVBoxLayout()
        
        self.ml_check = QCheckBox("استفاده از ML (zero-shot classification)")
        self.ml_check.setChecked(ML_AVAILABLE)
        self.ml_check.setEnabled(ML_AVAILABLE)
        self.ml_check. setToolTip("ترکیب Regex + ML برای دقت بالاتر")
        cat_layout.addWidget(self.ml_check)
        
        if not ML_AVAILABLE: 
            warning = QLabel("⚠️ transformers نصب نیست - فقط Regex")
            warning.setStyleSheet("color:  orange;")
            cat_layout.addWidget(warning)
        
        cat_group.setLayout(cat_layout)
        layout.addWidget(cat_group)
        
        self.setLayout(layout)
    
    def get_settings(self) -> dict:
        model_text = self.model_combo.currentText()
        model_size = model_text.split()[0]
        
        return {
            "model_size":  model_size,
            "beam_size": self. beam_spin.value(),
            "normalize": self.normalize_check.isChecked(),
            "use_ml": self. ml_check.isChecked()
        }


class ResultsPanel(QWidget):
    """پنل نمایش نتایج"""
    
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        
        # تب‌ها برای نتایج مختلف
        self.tabs = QTabWidget()
        
        # تب متن رونویسی
        self.transcript_text = QTextEdit()
        self.transcript_text.setReadOnly(True)
        self.transcript_text.setFont(QFont("Tahoma", 11))
        self.tabs.addTab(self.transcript_text, "📝 متن رونویسی")
        
        # تب جزئیات دسته‌بندی
        self.category_text = QTextEdit()
        self.category_text.setReadOnly(True)
        self.tabs.addTab(self.category_text, "🏷️ دسته‌بندی")
        
        # تب JSON خام
        self.json_text = QTextEdit()
        self.json_text. setReadOnly(True)
        self.json_text.setFont(QFont("Consolas", 10))
        self.tabs.addTab(self.json_text, "📋 JSON")
        
        layout.addWidget(self.tabs)
    
    def show_result(self, result:  dict):
        """نمایش نتایج"""
        
        # متن رونویسی
        asr = result.get("asr", {})
        transcript = asr.get("text", "")
        self.transcript_text. setPlainText(transcript)
        
        # جزئیات دسته‌بندی
        pred = result.get("prediction", {})
        cat_info = f"""
═══════════════════════════════════════════════════
🏷️  نتیجه دسته‌بندی
═══════════════════════════════════════════════════

📌 دسته اصلی: {pred. get('label_fa', pred.get('label', 'نامشخص'))}
📌 دسته (انگلیسی): {pred.get('label', 'unknown')}
📊 میزان اطمینان:  {pred.get('confidence', 0):.1%}
🔧 روش تشخیص: {pred.get('method', 'unknown')}
📁 گروه:  {pred.get('category_group', 'نامشخص')}

═══════════════════════════════════════════════════
📊 سایر احتمالات
═══════════════════════════════════════════════════
"""
        
        if 'top_categories' in pred: 
            for cat, score in pred['top_categories'][:5]:
                cat_obj = CATEGORIES_DETAILED.get(cat)
                cat_fa = cat_obj.name_fa if cat_obj else cat
                bar = "█" * int(score * 20)
                cat_info += f"\n{cat_fa: 20s} {bar:20s} {score:.1%}"
        
        if 'matched_keywords' in pred: 
            cat_info += "\n\n═══════════════════════════════════════════════════"
            cat_info += "\n🔍 کلمات کلیدی یافت شده"
            cat_info += "\n═══════════════════════════════════════════════════\n"
            for cat, keywords in pred['matched_keywords']. items():
                if keywords:
                    cat_obj = CATEGORIES_DETAILED.get(cat)
                    cat_fa = cat_obj.name_fa if cat_obj else cat
                    kw_str = ", ".join([f"{k}({c})" for k, c in keywords[: 5]])
                    cat_info += f"\n• {cat_fa}:  {kw_str}"
        
        self.category_text. setPlainText(cat_info)
        
        # JSON
        self.json_text. setPlainText(
            json.dumps(result, ensure_ascii=False, indent=2)
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎬 تحلیلگر حرفه‌ای ویدیوی فارسی v2.0")
        
        self.pool = QThreadPool()
        
        # ویجت اصلی
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        
        # عنوان
        title = QLabel("🎬 سیستم تحلیل و دسته‌بندی هوشمند ویدیوی فارسی")
        title.setFont(QFont("Tahoma", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("padding: 15px; color: #2c3e50;")
        main_layout.addWidget(title)
        
        # Splitter برای تقسیم صفحه
        splitter = QSplitter(Qt. Horizontal)
        
        # ═══ پنل چپ:  تنظیمات و کنترل ═══
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # تنظیمات
        self.settings_panel = SettingsPanel()
        left_layout.addWidget(self.settings_panel)
        
        # دکمه اصلی
        self.btn = QPushButton("📁 انتخاب ویدیو و شروع تحلیل")
        self.btn.setStyleSheet("""
            QPushButton {
                padding: 20px;
                font-size: 14px;
                font-weight: bold;
                background-color: #3498db;
                color: white;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.btn.clicked. connect(self.select_video_and_run)
        left_layout.addWidget(self.btn)
        
        # وضعیت
        self.status = QLabel("آماده برای تحلیل")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("padding: 10px; font-size: 12px;")
        left_layout.addWidget(self.status)
        
        # نوار پیشرفت
        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar {
                height: 25px;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color:  #27ae60;
                border-radius: 5px;
            }
        """)
        left_layout.addWidget(self.progress)
        
        # لاگ
        log_label = QLabel("📋 گزارش پردازش:")
        left_layout.addWidget(log_label)
        
        self.log = QTextEdit()
        self.log. setReadOnly(True)
        self.log.setFont(QFont("Consolas", 10))
        self.log.setStyleSheet("background-color: #2c3e50; color: #ecf0f1;")
        left_layout.addWidget(self.log)
        
        splitter.addWidget(left_panel)
        
        # ═══ پنل راست: نتایج ═══
        self.results_panel = ResultsPanel()
        splitter.addWidget(self.results_panel)
        
        # تنظیم نسبت splitter
        splitter.setSizes([400, 600])
        main_layout.addWidget(splitter)
        
        # راهنمای پایین
        info = QLabel(
            "💡 نکات:  "
            "• large-v3 بهترین کیفیت (اولین بار ~3GB دانلود می‌شود) "
            "• beam_size بالاتر = دقیق‌تر ولی کندتر "
            "• حتماً تصحیح فارسی را فعال کنید"
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 10px; background:  #f8f9fa; border-radius: 5px;")
        main_layout. addWidget(info)
        
        self.resize(1200, 800)
        self.center_window()
    
    def center_window(self):
        """قرار دادن پنجره در مرکز صفحه"""
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self. move(x, y)

    def select_video_and_run(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "انتخاب ویدیو",
            "",
            "Video Files (*.mp4 *.mkv *.mov *. avi *.webm *.flv *.wmv)"
        )
        
        if not path:
            return
        
        settings = self.settings_panel.get_settings()
        
        self.log.clear()
        self.progress.setValue(0)
        self.status.setText("⏳ در حال پردازش...")
        self.btn.setEnabled(False)
        
        worker = EnhancedPipelineWorker(path, settings)
        worker.signals.log.connect(self. on_log)
        worker.signals.progress. connect(self.progress. setValue)
        worker.signals.progress_detail.connect(self.on_progress_detail)
        worker.signals.finished.connect(self. on_finished)
        worker.signals.failed. connect(self.on_failed)
        self.pool.start(worker)

    def on_log(self, msg:  str):
        self.log.append(msg)

    def on_progress_detail(self, msg: str):
        self.status.setText(f"⏳ {msg}")

    def on_finished(self, result:  dict):
        self.status.setText("✅ تحلیل کامل شد!")
        self.btn.setEnabled(True)
        
        # نمایش در پنل نتایج
        self.results_panel.show_result(result)
        
        # پیام موفقیت
        pred = result.get("prediction", {})
        asr = result.get("asr", {})
        
        msg = (
            f"✅ تحلیل کامل شد!\n\n"
            f"🌐 زبان:  {asr.get('language', 'fa')}\n"
            f"📝 تعداد کلمات: {asr.get('word_count', 'نامشخص')}\n"
            f"📊 کیفیت رونویسی: {asr.get('transcription_quality', 'نامشخص')}\n\n"
            f"🏷️  دسته:  {pred.get('label_fa', pred.get('label', 'نامشخص'))}\n"
            f"📊 اطمینان: {pred.get('confidence', 0):.1%}\n\n"
            f"⏱️  زمان پردازش: {result.get('processing_time_seconds', 0):.1f} ثانیه\n"
            f"📁 فایل: outputs/{Path(result['video']).stem}/"
        )
        
        QMessageBox.information(self, "✅ تمام شد", msg)

    def on_failed(self, err: str):
        self.status.setText("❌ خطا در پردازش")
        self.btn.setEnabled(True)
        
        QMessageBox. critical(
            self, 
            "خطا",
            f"خطا در پردازش:\n\n{err[: 500]}..."
        )
        self.log.append(f"\n❌ خطا:\n{err}")


def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║   🎬 سیستم تحلیل و دسته‌بندی هوشمند ویدیوی فارسی v2.0   ║
    ╚════════════════════════════════════════════════════════════╝
    
    📦 کتابخانه‌های مورد نیاز: 
    
       pip install faster-whisper hazm parsivar PySide6 imageio-ffmpeg
       
       برای ML (اختیاری ولی پیشنهادی):
       pip install transformers torch
       
    ═══════════════════════════════════════════════════════════════
    """)
    
    print(f"   ✓ ASR پیشرفته: {'فعال ✅' if ASR_AVAILABLE else 'غیرفعال ❌'}")
    print(f"   ✓ دسته‌بندی ML: {'فعال ✅' if ML_AVAILABLE else 'غیرفعال ❌'}")
    print(f"   ✓ تعداد دسته‌ها: {len(CATEGORIES_DETAILED)}")
    print()
    
    app = QApplication(sys.argv)
    
    # تنظیم فونت پیش‌فرض
    font = QFont("Tahoma", 10)
    app.setFont(font)
    
    # استایل
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys. exit(app.exec())


if __name__ == "__main__":
    main()
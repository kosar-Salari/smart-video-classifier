"""
core/advanced_asr.py
ASR پیشرفته با تصحیح خودکار فارسی - نسخه اصلاح شده
"""
from faster_whisper import WhisperModel
from typing import Dict, List, Tuple
import re

try:
    from hazm import Normalizer, Lemmatizer
    HAZM_AVAILABLE = True
except ImportError:
    HAZM_AVAILABLE = False
    print("⚠️ hazm not available")

# parsivar را با احتیاط لود می‌کنیم
PARSIVAR_AVAILABLE = False
parsivar_normalizer = None

try:
    from parsivar import Normalizer as ParsivarNormalizer
    parsivar_normalizer = ParsivarNormalizer()
    PARSIVAR_AVAILABLE = True
except Exception as e:
    print(f"⚠️ parsivar Normalizer not available: {e}")

# SpellCheck را جداگانه تست می‌کنیم (معمولاً مشکل‌دار است)
SPELL_CHECK_AVAILABLE = False
spell_checker = None

try:
    from parsivar import SpellCheck
    spell_checker = SpellCheck()
    SPELL_CHECK_AVAILABLE = True
except Exception as e: 
    print(f"⚠️ parsivar SpellCheck not available (این عادی است): {e}")


class EnhancedPersianNormalizer:
    """تصحیح و نرمال‌سازی پیشرفته متن فارسی"""
    
    def __init__(self):
        # hazm
        self.hazm_normalizer = Normalizer() if HAZM_AVAILABLE else None
        self.lemmatizer = Lemmatizer() if HAZM_AVAILABLE else None
        
        # parsivar (فقط normalizer، نه spell checker)
        self.parsivar_normalizer = parsivar_normalizer
        self.spell_checker = spell_checker if SPELL_CHECK_AVAILABLE else None
        
        # دیکشنری تصحیحات رایج Whisper برای فارسی
        self.whisper_corrections = {
            # اشتباهات رایج Whisper
            "میشه": "می‌شه",
            "میشود": "می‌شود",
            "نمیشه": "نمی‌شه",
            "میخوام": "می‌خوام",
            "میخواهم": "می‌خواهم",
            "نمیتونم": "نمی‌تونم",
            "میتونم": "می‌تونم",
            "میکنم": "می‌کنم",
            "نمیکنم": "نمی‌کنم",
            "میگم": "می‌گم",
            "میگویم": "می‌گویم",
            "میدونم": "می‌دونم",
            "نمیدونم": "نمی‌دونم",
            "میخونم": "می‌خونم",
            "مینویسم": "می‌نویسم",
            "میبینم": "می‌بینم",
            "میرم": "می‌رم",
            "میام": "می‌آم",
            "میایم": "می‌آیم",
            "میکنیم": "می‌کنیم",
            "میکنید": "می‌کنید",
            "میکنند": "می‌کنند",
            "بزار": "بذار",
            "بزارید": "بذارید",
            "اینجوری": "این‌جوری",
            "اونجوری": "اون‌جوری",
            "چجوری": "چه‌جوری",
            "همینجوری": "همین‌جوری",
            
            # اسامی خاص
            "اوکراین": "اوکراین",
            "روسیه": "روسیه",
            "آمریکا": "آمریکا",
            "اسراییل": "اسرائیل",
            "اسرائیل": "اسرائیل",
        }
        
        # الگوهای regex برای تصحیح
        self.regex_patterns = [
            # حروف عربی به فارسی
            (r'ي', 'ی'),
            (r'ك', 'ک'),
            (r'ة', 'ه'),
            (r'ؤ', 'و'),
            (r'إ', 'ا'),
            (r'أ', 'ا'),
            (r'ٱ', 'ا'),
            
            # اعداد عربی به فارسی
            (r'٠', '۰'), (r'١', '۱'), (r'٢', '۲'), (r'٣', '۳'),
            (r'٤', '۴'), (r'٥', '۵'), (r'٦', '۶'), (r'٧', '۷'),
            (r'٨', '۸'), (r'٩', '۹'),
            
            # نیم‌فاصله برای پیشوندها
            (r'\bمی\s+', 'می‌'),
            (r'\bنمی\s+', 'نمی‌'),
            (r'\bبر\s+می\s+', 'برمی‌'),
            (r'\bهم\s+', 'هم‌'),
            
            # نیم‌فاصله برای پسوندها
            (r'\s+ها\b', '‌ها'),
            (r'\s+های\b', '‌های'),
            (r'\s+ای\b', '‌ای'),
            (r'\s+ام\b', '‌ام'),
            (r'\s+ات\b', '‌ات'),
            (r'\s+اش\b', '‌اش'),
            (r'\s+تر\b', '‌تر'),
            (r'\s+ترین\b', '‌ترین'),
            
            # کلمات ترکیبی
            (r'\bبین\s+الملل', 'بین‌الملل'),
            (r'\bبین\s+المللی', 'بین‌المللی'),
            (r'\bما\s+فوق', 'مافوق'),
            (r'\bصد\s+در\s+صد', 'صددرصد'),
            
            # فضای اضافی و علائم
            (r'\s+([،. ؛: ! ؟\)\]\}])', r'\1'),
            (r'([\(\[\{])\s+', r'\1'),
            (r'\s{2,}', ' '),
            
            # خط تیره‌ها
            (r'--+', '—'),
        ]
    
    def normalize(self, text: str) -> str:
        if not text:
            return text
        
        # مرحله 1: نرمال‌سازی اولیه با hazm
        if self.hazm_normalizer:
            try:
                text = self.hazm_normalizer.normalize(text)
            except Exception as e: 
                print(f"⚠️ hazm normalization failed: {e}")
        
        # مرحله 2: نرمال‌سازی با parsivar (اگر موجود)
        if self.parsivar_normalizer: 
            try:
                text = self.parsivar_normalizer.normalize(text)
            except Exception as e:
                print(f"⚠️ parsivar normalization failed:  {e}")
        
        # مرحله 3: تصحیحات دیکشنری
        for wrong, correct in self. whisper_corrections. items():
            text = text.replace(wrong, correct)
        
        # مرحله 4: تصحیحات regex
        for pattern, replacement in self.regex_patterns:
            try:
                text = re.sub(pattern, replacement, text)
            except Exception: 
                pass
        
        return text. strip()


def transcribe_advanced(
    wav_path: str,
    model_size: str = "large-v3",
    enable_normalization: bool = True,
    language: str = "fa",
    beam_size: int = 5,
    patience: float = 1.0,
    temperature:  Tuple[float, ... ] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> Dict: 
    """
    رونویسی پیشرفته با تنظیمات بهینه برای فارسی
    """
    
    print(f"🔄 Loading Whisper model: {model_size}...")
    
    # تنظیمات بهینه برای GPU/CPU
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
            print("   ✓ Using GPU (CUDA)")
        else:
            device = "cpu"
            compute_type = "int8"
            print("   ✓ Using CPU")
    except ImportError: 
        device = "cpu"
        compute_type = "int8"
        print("   ✓ Using CPU (torch not available)")
    
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root="D:/models/",
        num_workers=4,
    )
    
    # Normalizer
    normalizer = None
    if enable_normalization:
        try:
            normalizer = EnhancedPersianNormalizer()
            print("   ✓ Persian normalizer ready")
        except Exception as e:
            print(f"   ⚠️ Normalizer failed: {e}")
    
    # Prompt بهینه برای فارسی
    initial_prompt = """
    این یک ویدیوی فارسی است. 
    محتوا ممکن است شامل اخبار، آموزش، سیاست، تاریخ، ورزش، سرگرمی، آشپزی،
    فناوری، مذهب، سلامت، بازی، یا ولاگ باشد.
    لطفاً متن را به فارسی استاندارد رونویسی کنید.
    """
    
    print("🎤 Transcribing...")
    segments, info = model.transcribe(
        wav_path,
        language=language,
        beam_size=beam_size,
        patience=patience,
        temperature=temperature,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 100,
            "speech_pad_ms": 30,
        },
        initial_prompt=initial_prompt. strip(),
        condition_on_previous_text=True,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        word_timestamps=True,
    )
    
    # پردازش segments
    text_parts = []
    processed_segments = []
    word_list = []
    
    for seg in segments:
        text = seg.text. strip()
        
        if normalizer:
            try:
                text = normalizer.normalize(text)
            except Exception: 
                pass
        
        if text:
            text_parts.append(text)
            
            seg_data = {
                "start":  round(seg.start, 2),
                "end": round(seg. end, 2),
                "text":  text,
            }
            
            # اضافه کردن کلمات با timestamp
            if hasattr(seg, 'words') and seg.words:
                seg_data["words"] = []
                for w in seg.words:
                    word_text = w.word
                    if normalizer:
                        try:
                            word_text = normalizer.normalize(word_text)
                        except: 
                            pass
                    seg_data["words"].append({
                        "word": word_text,
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "probability": round(w.probability, 3)
                    })
                word_list.extend(seg_data["words"])
            
            processed_segments. append(seg_data)
    
    full_text = " ". join(text_parts).strip()
    
    # محاسبه میانگین احتمال کلمات
    avg_word_prob = 0.0
    if word_list: 
        avg_word_prob = sum(w["probability"] for w in word_list) / len(word_list)
    
    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "text":  full_text,
        "segments": processed_segments,
        "word_count": len(full_text.split()),
        "duration":  round(info.duration, 2) if hasattr(info, 'duration') else None,
        "avg_word_confidence": round(avg_word_prob, 3),
        "transcription_quality": _assess_quality(avg_word_prob, info.language_probability),
    }


def _assess_quality(avg_word_prob: float, lang_prob: float) -> str:
    """ارزیابی کیفیت رونویسی"""
    score = (avg_word_prob * 0.7) + (lang_prob * 0.3)
    if score >= 0.85:
        return "excellent"
    elif score >= 0.7:
        return "good"
    elif score >= 0.5:
        return "fair"
    else: 
        return "poor"
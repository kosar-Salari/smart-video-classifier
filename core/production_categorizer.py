"""
core/production_categorizer.py
دسته‌بندی پیشرفته با ML و Hybrid Approach - نسخه قوی
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re
import json

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    ML_AVAILABLE = True
except ImportError: 
    ML_AVAILABLE = False
    print("⚠️ transformers not available")

try:
    from hazm import word_tokenize, Lemmatizer, POSTagger
    HAZM_AVAILABLE = True
except ImportError:
    HAZM_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# تعریف دسته‌بندی‌ها - طیف وسیع
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Category:
    name: str
    name_fa: str
    description: str
    keywords: List[str]
    weight_boost: float = 1.0


CATEGORIES_DETAILED = {
    # ═══ اخبار و سیاست ═══
    "news": Category(
        "news", "اخبار",
        "اخبار روز، گزارش‌های خبری",
        ["خبر", "اخبار", "گزارش", "فوری", "رویداد", "حادثه", "اتفاق", "خبرنگار", 
         "خبرگزاری", "بولتن", "تیتر", "سرخط", "breaking", "news"],
    ),
    "politics_domestic": Category(
        "politics_domestic", "سیاست داخلی",
        "سیاست داخلی ایران",
        ["مجلس", "دولت", "رئیس‌جمهور", "وزیر", "نماینده", "قوه", "رهبر", "رهبری",
         "شورای نگهبان", "مجمع تشخیص", "قانون", "لایحه", "استیضاح", "انتخابات",
         "کاندیدا", "نامزد", "رأی", "صندوق", "حوزه انتخابیه"],
    ),
    "politics_international": Category(
        "politics_international", "سیاست بین‌الملل",
        "روابط بین‌الملل و سیاست جهانی",
        ["دیپلماسی", "سفیر", "سفارت", "وزارت خارجه", "سازمان ملل", "برجام",
         "تحریم", "مذاکره", "پیمان", "معاهده", "ناتو", "اتحادیه اروپا",
         "روابط خارجی", "بین‌الملل", "کشور", "دولت‌ها"],
    ),
    "geopolitics": Category(
        "geopolitics", "ژئوپلیتیک",
        "تحلیل ژئوپلیتیکی و منطقه‌ای",
        ["ژئوپلیتیک", "منطقه", "خاورمیانه", "آسیا", "اروپا", "آمریکا",
         "روسیه", "چین", "قدرت", "نفوذ", "استراتژی", "منافع ملی"],
    ),
    
    # ═══ اقتصاد ═══
    "economy": Category(
        "economy", "اقتصاد",
        "اخبار و تحلیل اقتصادی",
        ["اقتصاد", "بازار", "بورس", "سهام", "ارز", "دلار", "تورم", "رکود",
         "رشد اقتصادی", "تولید ناخالص", "بانک مرکزی", "نرخ بهره", "سرمایه",
         "سرمایه‌گذاری", "تجارت", "واردات", "صادرات", "گمرک"],
    ),
    "crypto": Category(
        "crypto", "ارز دیجیتال",
        "ارزهای دیجیتال و بلاکچین",
        ["بیت‌کوین", "اتریوم", "کریپتو", "ارز دیجیتال", "بلاکچین", "ماینینگ",
         "استخراج", "کیف پول", "صرافی", "توکن", "NFT", "دیفای",
         "bitcoin", "ethereum", "crypto", "blockchain"],
    ),
    
    # ═══ نظامی و امنیتی ═══
    "military": Category(
        "military", "نظامی",
        "اخبار و تحلیل نظامی",
        ["ارتش", "سپاه", "نیروی هوایی", "نیروی دریایی", "نظامی", "جنگ",
         "درگیری", "عملیات", "رزمایش", "موشک", "پهپاد", "تانک", "جنگنده",
         "سلاح", "مهمات", "تسلیحات", "ناو", "زیردریایی"],
    ),
    "defense": Category(
        "defense", "دفاعی",
        "صنایع دفاعی و فناوری نظامی",
        ["پدافند", "سامانه موشکی", "رادار", "جنگ الکترونیک", "سایبری",
         "پهپاد", "موشک بالستیک", "کروز", "ماهواره نظامی"],
    ),
    
    # ═══ تاریخ ═══
    "history_ancient": Category(
        "history_ancient", "تاریخ باستان",
        "تاریخ باستان و تمدن‌های قدیمی",
        ["باستان", "هخامنشی", "ساسانی", "اشکانی", "مادها", "کوروش",
         "داریوش", "تخت جمشید", "شوش", "تمدن", "امپراتوری", "پادشاه"],
    ),
    "history_medieval":  Category(
        "history_medieval", "تاریخ میانه",
        "تاریخ قرون وسطی و اسلامی",
        ["قرون وسطی", "صفوی", "قاجار", "عباسی", "اموی", "سلجوقی",
         "مغول", "تیموری", "شاه عباس", "نادرشاه"],
    ),
    "history_modern":  Category(
        "history_modern", "تاریخ معاصر",
        "تاریخ معاصر و قرن بیستم",
        ["انقلاب", "مشروطه", "پهلوی", "جنگ جهانی", "جنگ سرد",
         "استعمار", "ملی شدن نفت", "کودتا", "جنگ تحمیلی"],
    ),
    "history_world": Category(
        "history_world", "تاریخ جهان",
        "تاریخ جهانی و تمدن‌ها",
        ["نازی", "هیتلر", "استالین", "چرچیل", "روزولت", "امپراتوری روم",
         "یونان باستان", "مصر باستان", "جنگ جهانی اول", "جنگ جهانی دوم"],
    ),
    
    # ═══ مذهب و معنویت ═══
    "religion_islam": Category(
        "religion_islam", "اسلام",
        "آموزش‌های اسلامی",
        ["قرآن", "نماز", "روزه", "حج", "زکات", "خمس", "امام", "پیامبر",
         "حدیث", "روایت", "فقه", "احکام", "مسجد", "حرم", "زیارت"],
    ),
    "religion_shia": Category(
        "religion_shia", "تشیع",
        "مذهب شیعه",
        ["امام حسین", "کربلا", "عاشورا", "محرم", "اربعین", "امام رضا",
         "مشهد", "نجف", "امام علی", "حضرت زهرا", "ائمه"],
    ),
    "religion_other": Category(
        "religion_other", "ادیان",
        "سایر ادیان و معنویت",
        ["مسیحیت", "یهودیت", "بودیسم", "هندوئیسم", "زرتشت", "عرفان",
         "تصوف", "معنویت", "مدیتیشن", "یوگا"],
    ),
    
    # ═══ علم و فناوری ═══
    "tech_ai": Category(
        "tech_ai", "هوش مصنوعی",
        "هوش مصنوعی و یادگیری ماشین",
        ["هوش مصنوعی", "یادگیری ماشین", "یادگیری عمیق", "شبکه عصبی",
         "ChatGPT", "GPT", "AI", "machine learning", "deep learning",
         "داده", "الگوریتم", "مدل", "آموزش مدل"],
    ),
    "tech_programming": Category(
        "tech_programming", "برنامه‌نویسی",
        "برنامه‌نویسی و توسعه نرم‌افزار",
        ["برنامه‌نویسی", "کدنویسی", "پایتون", "جاوا", "جاوااسکریپت",
         "وب", "اپلیکیشن", "فرانت‌اند", "بک‌اند", "دیتابیس", "API",
         "گیت", "گیتهاب", "لینوکس", "سرور"],
    ),
    "tech_hardware": Category(
        "tech_hardware", "سخت‌افزار",
        "سخت‌افزار و گجت‌ها",
        ["موبایل", "گوشی", "لپ‌تاپ", "کامپیوتر", "پردازنده", "گرافیک",
         "رم", "حافظه", "باتری", "آیفون", "سامسونگ", "شیائومی",
         "اپل", "گوگل", "مایکروسافت"],
    ),
    "tech_internet": Category(
        "tech_internet", "اینترنت",
        "اینترنت و شبکه‌های اجتماعی",
        ["اینترنت", "فیلترینگ", "VPN", "شبکه اجتماعی", "اینستاگرام",
         "تلگرام", "توییتر", "یوتیوب", "تیک‌تاک", "فیسبوک"],
    ),
    
    # ═══ سلامت ═══
    "health_medicine": Category(
        "health_medicine", "پزشکی",
        "پزشکی و درمان",
        ["پزشک", "دکتر", "بیمارستان", "درمان", "دارو", "بیماری",
         "سرطان", "قلب", "دیابت", "فشار خون", "جراحی", "عمل"],
    ),
    "health_mental": Category(
        "health_mental", "سلامت روان",
        "روانشناسی و سلامت روان",
        ["روانشناسی", "افسردگی", "اضطراب", "استرس", "روان‌درمانی",
         "مشاوره", "روانپزشک", "خودشناسی", "ذهن‌آگاهی"],
    ),
    "health_fitness": Category(
        "health_fitness", "تناسب اندام",
        "ورزش و تناسب اندام",
        ["تناسب اندام", "فیتنس", "بدنسازی", "ورزش", "رژیم", "لاغری",
         "عضله", "چربی", "کالری", "پروتئین", "مکمل"],
    ),
    "health_nutrition": Category(
        "health_nutrition", "تغذیه",
        "تغذیه سالم",
        ["تغذیه", "رژیم غذایی", "ویتامین", "مواد مغذی", "سبزیجات",
         "میوه", "پروتئین", "کربوهیدرات", "چربی سالم"],
    ),
    
    # ═══ ورزش ═══
    "sports_football": Category(
        "sports_football", "فوتبال",
        "فوتبال",
        ["فوتبال", "لیگ برتر", "لیگ قهرمانان", "جام جهانی", "گل",
         "بازی", "تیم", "مربی", "بازیکن", "داور", "پنالتی",
         "استقلال", "پرسپولیس", "رئال", "بارسلونا"],
    ),
    "sports_other": Category(
        "sports_other", "سایر ورزش‌ها",
        "سایر رشته‌های ورزشی",
        ["بسکتبال", "والیبال", "کشتی", "تکواندو", "شنا", "دوومیدانی",
         "تنیس", "شطرنج", "المپیک", "مدال", "قهرمانی"],
    ),
    
    # ═══ آشپزی و غذا ═══
    "cooking_persian": Category(
        "cooking_persian", "آشپزی ایرانی",
        "غذاهای ایرانی",
        ["قورمه سبزی", "قیمه", "کباب", "جوجه", "چلو", "پلو",
         "خورش", "آش", "دیزی", "زرشک پلو", "تهدیگ"],
    ),
    "cooking_international": Category(
        "cooking_international", "آشپزی بین‌المللی",
        "غذاهای بین‌المللی",
        ["پیتزا", "پاستا", "سوشی", "برگر", "استیک", "سالاد",
         "فست فود", "ایتالیایی", "چینی", "ژاپنی"],
    ),
    "cooking_baking": Category(
        "cooking_baking", "شیرینی‌پزی",
        "کیک و شیرینی",
        ["کیک", "شیرینی", "دسر", "بیسکویت", "کلوچه", "باقلوا",
         "زولبیا", "نان", "خمیر", "فر", "پخت"],
    ),
    
    # ═══ سرگرمی ═══
    "entertainment_movie": Category(
        "entertainment_movie", "فیلم",
        "فیلم و سینما",
        ["فیلم", "سینما", "کارگردان", "بازیگر", "اسکار", "هالیوود",
         "سریال", "نتفلیکس", "فیلمبرداری", "سناریو"],
    ),
    "entertainment_music":  Category(
        "entertainment_music", "موسیقی",
        "موسیقی",
        ["موسیقی", "آهنگ", "خواننده", "کنسرت", "آلبوم", "ترانه",
         "ملودی", "ریتم", "پاپ", "راک", "سنتی", "رپ"],
    ),
    "entertainment_comedy": Category(
        "entertainment_comedy", "طنز",
        "طنز و کمدی",
        ["طنز", "کمدی", "خنده", "شوخی", "جوک", "استندآپ",
         "کمدین", "خنده‌دار", "شاد"],
    ),
    
    # ═══ گیمینگ ═══
    "gaming_pc": Category(
        "gaming_pc", "بازی PC",
        "بازی‌های کامپیوتری",
        ["گیم", "بازی", "پلی", "گیمر", "استریم", "لول", "مرحله",
         "کنسول", "پلی‌استیشن", "ایکس‌باکس", "نینتندو", "PC"],
    ),
    "gaming_mobile": Category(
        "gaming_mobile", "بازی موبایل",
        "بازی‌های موبایلی",
        ["بازی موبایل", "کلش", "پابجی", "فری فایر", "کال آف دیوتی موبایل"],
    ),
    
    # ═══ آموزش ═══
    "education_academic": Category(
        "education_academic", "آموزش دانشگاهی",
        "دروس دانشگاهی",
        ["دانشگاه", "استاد", "درس", "امتحان", "کنکور", "تحصیل",
         "لیسانس", "فوق لیسانس", "دکتری", "پایان‌نامه"],
    ),
    "education_language": Category(
        "education_language", "آموزش زبان",
        "آموزش زبان‌های خارجی",
        ["زبان انگلیسی", "آیلتس", "تافل", "گرامر", "لغت", "مکالمه",
         "زبان آلمانی", "زبان فرانسه", "زبان عربی"],
    ),
    "education_skills": Category(
        "education_skills", "آموزش مهارت",
        "آموزش مهارت‌های عملی",
        ["آموزش", "یادگیری", "تمرین", "مهارت", "دوره", "کلاس",
         "ورکشاپ", "کارگاه", "گواهینامه"],
    ),
    
    # ═══ سبک زندگی ═══
    "lifestyle_vlog": Category(
        "lifestyle_vlog", "ولاگ",
        "ویدیوهای روزمره",
        ["ولاگ", "روزمره", "روتین", "زندگی", "یه روز", "همراه من",
         "با من بیا", "روزانه"],
    ),
    "lifestyle_travel": Category(
        "lifestyle_travel", "سفر",
        "سفر و گردشگری",
        ["سفر", "گردشگری", "توریست", "هتل", "پرواز", "ویزا",
         "جاذبه", "دیدنی", "طبیعت", "ماجراجویی"],
    ),
    "lifestyle_fashion": Category(
        "lifestyle_fashion", "مد و زیبایی",
        "مد و آرایش",
        ["مد", "فشن", "لباس", "استایل", "آرایش", "میکاپ",
         "زیبایی", "مو", "اکسسوری", "برند"],
    ),
    
    # ═══ خودرو ═══
    "automotive":  Category(
        "automotive", "خودرو",
        "خودرو و وسایل نقلیه",
        ["خودرو", "ماشین", "موتور", "بنز", "بی‌ام‌و", "تویوتا",
         "ایران‌خودرو", "سایپا", "تست", "بررسی", "سرعت"],
    ),
    
    # ═══ کسب‌وکار ═══
    "business":  Category(
        "business", "کسب‌وکار",
        "کارآفرینی و کسب‌وکار",
        ["کسب‌وکار", "استارتاپ", "کارآفرینی", "درآمد", "سود",
         "فروش", "مارکتینگ", "بازاریابی", "مشتری", "برند"],
    ),
    
    # ═══ کودک و خانواده ═══
    "family_kids": Category(
        "family_kids", "کودک",
        "محتوای کودکان",
        ["کودک", "بچه", "کارتون", "انیمیشن", "بازی کودک",
         "آموزش کودک", "قصه", "شعر کودک"],
    ),
    "family_parenting": Category(
        "family_parenting", "والدین",
        "فرزندپروری",
        ["فرزندپروری", "والدین", "مادر", "پدر", "تربیت",
         "نوزاد", "بارداری", "شیردهی"],
    ),
    
    # ═══ متفرقه ═══
    "documentary": Category(
        "documentary", "مستند",
        "فیلم‌های مستند",
        ["مستند", "داکیومنتری", "حیات وحش", "طبیعت", "علمی",
         "تحقیق", "بررسی", "گزارش مستند"],
    ),
    "podcast": Category(
        "podcast", "پادکست",
        "پادکست و گفتگو",
        ["پادکست", "گفتگو", "مصاحبه", "بحث", "میزگرد",
         "نظر", "دیدگاه"],
    ),
    "asmr": Category(
        "asmr", "ASMR",
        "ویدیوهای آرامش‌بخش",
        ["ASMR", "آرامش", "خواب", "ریلکس", "صدای آرام"],
    ),
    "other":  Category(
        "other", "سایر",
        "دسته‌بندی نشده",
        [],
        weight_boost=0.5
    ),
}

# گروه‌بندی دسته‌ها برای تحلیل سلسله‌مراتبی
CATEGORY_GROUPS = {
    "news_politics": ["news", "politics_domestic", "politics_international", "geopolitics"],
    "economy":  ["economy", "crypto"],
    "military_defense": ["military", "defense"],
    "history":  ["history_ancient", "history_medieval", "history_modern", "history_world"],
    "religion":  ["religion_islam", "religion_shia", "religion_other"],
    "technology": ["tech_ai", "tech_programming", "tech_hardware", "tech_internet"],
    "health":  ["health_medicine", "health_mental", "health_fitness", "health_nutrition"],
    "sports":  ["sports_football", "sports_other"],
    "cooking": ["cooking_persian", "cooking_international", "cooking_baking"],
    "entertainment": ["entertainment_movie", "entertainment_music", "entertainment_comedy"],
    "gaming": ["gaming_pc", "gaming_mobile"],
    "education": ["education_academic", "education_language", "education_skills"],
    "lifestyle": ["lifestyle_vlog", "lifestyle_travel", "lifestyle_fashion"],
    "other": ["automotive", "business", "family_kids", "family_parenting", 
              "documentary", "podcast", "asmr", "other"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# موتور دسته‌بندی
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedRegexCategorizer:
    """دسته‌بندی پیشرفته با Regex و TF-IDF-like scoring"""
    
    def __init__(self):
        self.categories = CATEGORIES_DETAILED
        self.lemmatizer = Lemmatizer() if HAZM_AVAILABLE else None
        
        # ساخت الگوهای regex برای هر دسته
        self.compiled_patterns = {}
        for cat_name, cat in self.categories.items():
            patterns = []
            for keyword in cat. keywords:
                # ساخت الگوی انعطاف‌پذیر
                pattern = self._make_flexible_pattern(keyword)
                patterns.append((re.compile(pattern, re.IGNORECASE), keyword))
            self.compiled_patterns[cat_name] = patterns
    
    def _make_flexible_pattern(self, keyword:  str) -> str:
        """ساخت الگوی regex انعطاف‌پذیر"""
        # اجازه نیم‌فاصله و فاصله
        keyword = keyword.replace(" ", r"[\s\u200c]*")
        keyword = keyword.replace("‌", r"[\s\u200c]*")
        return rf"\b{keyword}\b"
    
    def classify(self, text: str, top_n: int = 3) -> Dict: 
        """دسته‌بندی با امتیازدهی پیشرفته"""
        if not text or len(text. strip()) < 20:
            return {
                "label": "other",
                "label_fa": "سایر",
                "confidence": 0.1,
                "top_categories": [("other", 0.1)],
                "all_scores": {},
                "method": "insufficient_text"
            }
        
        text_lower = text. lower()
        text_len = len(text. split())
        
        # محاسبه امتیاز هر دسته
        scores = defaultdict(float)
        matches = defaultdict(list)
        
        for cat_name, patterns in self.compiled_patterns.items():
            cat = self.categories[cat_name]
            
            for pattern, keyword in patterns:
                found = pattern.findall(text_lower)
                if found: 
                    count = len(found)
                    # امتیاز TF-IDF-like
                    tf = count / max(text_len, 1)
                    # کلمات کمتر رایج امتیاز بیشتر
                    idf = 1.0 + (len(keyword) / 10)
                    
                    score = tf * idf * cat.weight_boost
                    scores[cat_name] += score
                    matches[cat_name].append((keyword, count))
        
        # نرمال‌سازی امتیازات
        total = sum(scores.values())
        if total <= 0:
            return {
                "label": "other",
                "label_fa": "سایر",
                "confidence": 0.15,
                "top_categories": [("other", 0.15)],
                "all_scores":  {},
                "method": "no_match"
            }
        
        normalized = {k: v / total for k, v in scores.items()}
        
        # مرتب‌سازی
        sorted_cats = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        top_cats = sorted_cats[:top_n]
        
        best_cat = top_cats[0][0]
        best_conf = top_cats[0][1]
        
        # اگر اطمینان کم است، other برگردان
        if best_conf < 0.1:
            best_cat = "other"
            best_conf = 0.2
        
        return {
            "label": best_cat,
            "label_fa": self.categories[best_cat]. name_fa,
            "confidence": float(best_conf),
            "top_categories": [(c, float(s)) for c, s in top_cats],
            "all_scores": {k: float(v) for k, v in normalized. items()},
            "matched_keywords": dict(matches),
            "method": "regex_tfidf"
        }


class HybridCategorizer: 
    """دسته‌بندی هیبرید:  ترکیب Regex + ML"""
    
    def __init__(self, use_ml: bool = True):
        self.regex_categorizer = AdvancedRegexCategorizer()
        self.ml_classifier = None
        self.ml_available = False
        
        if use_ml and ML_AVAILABLE: 
            self._init_ml_classifier()
    
    def _init_ml_classifier(self):
        """راه‌اندازی مدل ML"""
        try: 
            print("🔄 Loading ML classifier...")
            
            # استفاده از مدل چندزبانه برای فارسی
            model_name = "MoritzLaworther/multilingual-e5-small"
            
            self.ml_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch. cuda.is_available() else -1
            )
            
            self.ml_available = True
            print("   ✓ ML classifier ready")
            
        except Exception as e: 
            print(f"   ⚠️ ML initialization failed: {e}")
            self.ml_available = False
    
    def classify(self, text:  str, use_ml: bool = True) -> Dict:
        """دسته‌بندی هیبرید"""
        
        # مرحله 1: Regex
        regex_result = self. regex_categorizer. classify(text)
        
        # اگر regex اطمینان بالا دارد، همان را برگردان
        if regex_result["confidence"] >= 0.4:
            regex_result["method"] = "regex_high_confidence"
            return regex_result
        
        # مرحله 2: ML (اگر موجود و فعال)
        if use_ml and self. ml_available and self.ml_classifier:
            try: 
                ml_result = self._ml_classify(text)
                
                # ترکیب نتایج
                combined = self._combine_results(regex_result, ml_result)
                return combined
                
            except Exception as e: 
                print(f"⚠️ ML classification failed: {e}")
        
        # Fallback به regex
        return regex_result
    
    def _ml_classify(self, text: str) -> Dict:
        """دسته‌بندی با ML"""
        # محدود کردن طول متن
        text = text[:1000] if len(text) > 1000 else text
        
        # لیبل‌های فارسی برای zero-shot
        labels = [cat.name_fa for cat in CATEGORIES_DETAILED. values()]
        
        result = self.ml_classifier(
            text,
            candidate_labels=labels,
            multi_label=False
        )
        
        # تبدیل به فرمت استاندارد
        label_fa = result['labels'][0]
        confidence = result['scores'][0]
        
        # پیدا کردن label انگلیسی
        label_en = "other"
        for cat_name, cat in CATEGORIES_DETAILED. items():
            if cat. name_fa == label_fa:
                label_en = cat_name
                break
        
        return {
            "label": label_en,
            "label_fa": label_fa,
            "confidence": float(confidence),
            "method": "ml_zero_shot"
        }
    
    def _combine_results(self, regex_result: Dict, ml_result: Dict) -> Dict:
        """ترکیب نتایج regex و ML"""
        
        # وزن‌دهی
        regex_weight = 0.6
        ml_weight = 0.4
        
        # اگر هر دو یک نتیجه دارند
        if regex_result["label"] == ml_result["label"]:
            combined_conf = min(
                regex_result["confidence"] + ml_result["confidence"],
                0.95
            )
            return {
                "label": regex_result["label"],
                "label_fa": regex_result["label_fa"],
                "confidence": combined_conf,
                "top_categories": regex_result. get("top_categories", []),
                "method": "hybrid_agreement",
                "regex_result": regex_result,
                "ml_result": ml_result
            }
        
        # اگر متفاوت هستند، وزن‌دهی کنیم
        regex_score = regex_result["confidence"] * regex_weight
        ml_score = ml_result["confidence"] * ml_weight
        
        if regex_score >= ml_score: 
            winner = regex_result
            method = "hybrid_regex_wins"
        else:
            winner = ml_result
            method = "hybrid_ml_wins"
        
        return {
            "label": winner["label"],
            "label_fa": winner["label_fa"],
            "confidence": max(regex_score, ml_score),
            "method": method,
            "regex_result": regex_result,
            "ml_result": ml_result
        }


class ProductionCategorizer: 
    """دسته‌بند اصلی برای تولید"""
    
    def __init__(self, use_ml: bool = True):
        self.categorizer = HybridCategorizer(use_ml=use_ml)
        self.categories = CATEGORIES_DETAILED
    
    def classify(self, text: str) -> Dict:
        """دسته‌بندی متن"""
        result = self.categorizer.classify(text)
        
        # اضافه کردن اطلاعات گروه
        label = result["label"]
        group = None
        for group_name, members in CATEGORY_GROUPS.items():
            if label in members:
                group = group_name
                break
        
        result["category_group"] = group
        result["category_description"] = self.categories[label].description
        
        return result
    
    def get_all_categories(self) -> List[Dict]:
        """لیست تمام دسته‌ها"""
        return [
            {
                "name": cat.name,
                "name_fa": cat. name_fa,
                "description": cat.description
            }
            for cat in self.categories.values()
        ]


def classify_text(text: str) -> Dict:
    """تابع ساده برای استفاده سریع"""
    categorizer = ProductionCategorizer(use_ml=ML_AVAILABLE)
    return categorizer. classify(text)


# ═══════════════════════════════════════════════════════════════════════════════
# تست
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_texts = [
        "امروز می‌خوام طرز تهیه قورمه سبزی رو یاد بدم.  مواد لازم شامل سبزی قورمه، گوشت و لوبیا قرمز است.",
        "در جنگ جهانی دوم، هیتلر به لهستان حمله کرد و این آغاز جنگ بود.",
        "بازی امروز استقلال و پرسپولیس خیلی هیجان‌انگیز بود.  گل اول رو استقلال زد.",
        "امروز می‌خوام در مورد هوش مصنوعی و ChatGPT صحبت کنم.",
        "نماز و روزه از واجبات دین اسلام هستند.",
    ]
    
    categorizer = ProductionCategorizer(use_ml=False)
    
    for text in test_texts:
        result = categorizer.classify(text)
        print(f"\n{'='*60}")
        print(f"متن: {text[: 50]}...")
        print(f"دسته:  {result['label_fa']} ({result['label']})")
        print(f"اطمینان: {result['confidence']:.1%}")
        print(f"روش: {result['method']}")
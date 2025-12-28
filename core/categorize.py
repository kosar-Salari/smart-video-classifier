import re

CATEGORIES = [
    "cooking", "military", "news", "education", "entertainment",
    "sports", "tech", "vlog", "gaming", "religion", "politics", "other"
]

_RULES = {
    "cooking": [
        r"\b(آشپزی|طرز\s*تهیه|مواد\s*لازم|دستور)\b",
        r"\b(شیرینی|کیک|دسر|غذا|خوراک|نان)\b",
        r"\b(آرد|خمیر|بیکینگ\s*پودر|روغن|تابه|فر|قابلمه|همزن|پیمانه|قالب|زردچوبه|ادویه|زعفران)\b",
        r"\b(سرخ|سرخ\s*کردن|تفت|پخت|بپز|برش|ورز)\b",
        r"\b(recipe|cook|cooking|ingredients|bake|fry|oven|dough)\b",
    ],
    "education": [
        r"\b(آموزش|یادگیری|راهنما|درس|کلاس)\b",
        r"\b(tutorial|lesson|learn|course|guide)\b",
    ],
    "vlog": [
        # فقط شاخص‌های واقعی ولاگ؛ کلمات عمومی را حذف کن
        r"\b(ولاگ|روزمرگی|امروز\s*با\s*هم|روتین|my\s*day)\b",
        r"\b(vlog|daily\s*vlog|day\s*in\s*my\s*life|routine)\b",
    ],
    "military": [
        r"\b(نظامی|ارتش|موشک|پهپاد|تسلیحات|رزمایش|جنگ|پدافند)\b",
        r"\b(military|army|missile|drone|weapon|war|defense)\b",
    ],
    "news": [
        r"\b(خبر|گزارش|فوری|تیتر)\b",
        r"\b(news|breaking|report|headline)\b",
    ],
    "sports": [
        r"\b(فوتبال|بسکتبال|والیبال|مسابقه|لیگ|گل)\b",
        r"\b(football|soccer|basketball|match|league|goal)\b",
    ],
    "tech": [
        r"\b(هوش\s*مصنوعی|برنامه\s*نویسی|پایتون|نرم\s*افزار|سخت\s*افزار|گجت)\b",
        r"\b(ai|python|software|hardware|gadget|tech)\b",
    ],
    "gaming": [
        r"\b(بازی|گیم|کنسول|استریم|لول)\b",
        r"\b(game|gaming|console|stream|level)\b",
    ],
    "religion": [
        r"\b(قرآن|نماز|مسجد|دعا|حدیث|امام)\b",
        r"\b(quran|prayer|mosque|dua|hadith)\b",
    ],
    "politics": [
        r"\b(دولت|وزیر|مجلس|انتخابات|سیاست|حزب)\b",
        r"\b(government|election|politics|parliament|party)\b",
    ],
    "entertainment": [
        r"\b(فیلم|سریال|موسیقی|کلیپ|بازیگر|شو)\b",
        r"\b(movie|series|music|clip|actor|show)\b",
    ],
}

# وزن‌دهی: cooking مهم‌تر برای جلوگیری از باخت به کلمات عمومی
_WEIGHTS = {
    "cooking": 3.0,
    "education": 1.5,
    "vlog": 1.0,
}

def _count_hits(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))

def classify_text(text: str):
    t = (text or "").lower()

    scores = {c: 0.0 for c in CATEGORIES}
    for cat, patterns in _RULES.items():
        w = _WEIGHTS.get(cat, 1.0)
        for p in patterns:
            scores[cat] += w * _count_hits(p, t)

    best = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())

    if total <= 0:
        return {"label": "other", "confidence": 0.2, "scores": scores}

    confidence = scores[best] / total
    return {"label": best, "confidence": float(confidence), "scores": scores}

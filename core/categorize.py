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
        r"\b(آموزش|یادگیری|راهنما|درس|کلاس|دانشگاه|مدرسه|استاد)\b",
        r"\b(tutorial|lesson|learn|course|guide|university|school)\b",
    ],
    "vlog": [
        r"\b(ولاگ|روزمرگی|امروز\s*با\s*هم|روتین|my\s*day)\b",
        r"\b(vlog|daily\s*vlog|day\s*in\s*my\s*life|routine)\b",
    ],
    "military": [
        r"\b(نظامی|ارتش|موشک|پهپاد|تسلیحات|رزمایش|جنگ|پدافند|حمله|هجوم)\b",
        r"\b(military|army|missile|drone|weapon|war|defense|attack)\b",
    ],
    "news": [
        r"\b(خبر|گزارش|فوری|تیتر|رویداد|اتفاق|امروز|دیروز)\b",
        r"\b(news|breaking|report|headline|event|today|yesterday)\b",
    ],
    "sports": [
        r"\b(فوتبال|بسکتبال|والیبال|مسابقه|لیگ|گل|تیم|بازیکن)\b",
        r"\b(football|soccer|basketball|match|league|goal|team|player)\b",
    ],
    "tech": [
        r"\b(هوش\s*مصنوعی|برنامه\s*نویسی|پایتون|نرم\s*افزار|سخت\s*افزار|گجت|تکنولوژی)\b",
        r"\b(ai|python|software|hardware|gadget|tech|technology)\b",
    ],
    "gaming": [
        r"\b(بازی|گیم|کنسول|استریم|لول|گیمر|پلی\s*استیشن)\b",
        r"\b(game|gaming|console|stream|level|gamer|playstation)\b",
    ],
    "religion": [
        r"\b(قرآن|نماز|مسجد|دعا|حدیث|امام|دین|مذهب)\b",
        r"\b(quran|prayer|mosque|dua|hadith|imam|religion)\b",
    ],
    "politics": [
        # توسعه یافته برای تشخیص بهتر محتوای سیاسی
        r"\b(دولت|وزیر|مجلس|انتخابات|سیاست|حزب|نظام|مسئول|مسئولین)\b",
        r"\b(اقتصاد|اقتصادی|تحریم|تحریمها|تورم|قیمت|بازار)\b",
        r"\b(روابط|بین\s*الملل|بینوملل|کشور|کشورها|ملت|مردم)\b",
        r"\b(تحول|اصلاحات|بحران|شرایط|وضعیت|اوضاع)\b",
        r"\b(شاخص|رشد|توسعه|پیشرفت|عقب\s*نشینی)\b",
        r"\b(government|election|politics|parliament|party|minister)\b",
        r"\b(economy|economic|sanctions|inflation|crisis)\b",
    ],
    "entertainment": [
        r"\b(فیلم|سریال|موسیقی|کلیپ|بازیگر|شو|سینما)\b",
        r"\b(movie|series|music|clip|actor|show|cinema)\b",
    ],
}

# وزن‌دهی بهبود یافته
_WEIGHTS = {
    "cooking": 3.0,
    "politics": 2.5,  # افزایش وزن سیاست
    "news": 2.0,      # افزایش وزن اخبار
    "education": 1.5,
    "military": 1.5,
    "vlog": 1.0,
}

def _count_hits(pattern: str, text: str) -> int:
    """شمارش تطابق‌ها"""
    return len(re.findall(pattern, text, flags=re.IGNORECASE))

def classify_text(text: str):
    """دسته‌بندی متن با Regex"""
    t = (text or "").lower()

    scores = {c: 0.0 for c in CATEGORIES}
    
    # محاسبه امتیازها
    for cat, patterns in _RULES.items():
        w = _WEIGHTS.get(cat, 1.0)
        for p in patterns:
            hits = _count_hits(p, t)
            scores[cat] += w * hits

    # پیدا کردن بهترین دسته
    best = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())

    # اگر هیچ تطابقی پیدا نشد
    if total <= 0:
        return {
            "label": "other",
            "confidence": 0.2,
            "scores": scores
        }

    confidence = scores[best] / total
    
    # اگر اطمینان خیلی پایین، other برگردون
    if confidence < 0.15:
        return {
            "label": "other",
            "confidence": float(confidence),
            "scores": scores
        }

    return {
        "label": best,
        "confidence": float(confidence),
        "scores": scores
    }


# تست
if __name__ == "__main__":
    test_text = """
    آنچه از بزرگان نظام میشنویم تحوول رو غیر قابل انتظار می کنه.
    اگر اینچنین باشه ما باید منتظر شرایت بطری باشه.
    فارق از هملهی که به ایران شد، مسئولین ما هیچ گونه تحوولی از سال گذشته 
    به امروز در اقتصاد نتوانستان به وجود بیا برا.
    شاخس ها همه رو به بطر شدن، روابت بینوملل رو به بهبود نیست.
    """
    
    result = classify_text(test_text)
    print(f"دسته: {result['label']}")
    print(f"اطمینان: {result['confidence']:.2%}")
    print("\nامتیازات:")
    for cat, score in sorted(result['scores'].items(), key=lambda x: x[1], reverse=True):
        if score > 0:
            print(f"  {cat:15}: {score:.1f}")
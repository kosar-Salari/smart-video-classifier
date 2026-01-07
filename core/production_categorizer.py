"""
core/production_categorizer.py
دسته‌بندی فوق‌قوی فارسی (Hybrid: Keyword + Multilingual Embedding + Hierarchical + Segment Voting)

نکته:
- تعریف دسته‌ها و گروه‌ها در فایل جدا: core/category_catalog.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import math
import re

# -----------------------------------------------------------------------------
# Import categories catalog (separated file)
# -----------------------------------------------------------------------------
# اگر پروژه شما به‌صورت پکیج اجرا می‌شود (recommended):
#   from core.category_catalog import Category, CATEGORIES_DETAILED, CATEGORY_GROUPS
# اگر گاهی اسکریپتی اجرا می‌کنید و import مشکل دارد، این try/except کمک می‌کند.
try:
    from core.category_catalog import Category, CATEGORIES_DETAILED, CATEGORY_GROUPS
except Exception:
    # fallback برای اجراهای غیرپکیجی (مثلاً وقتی core روی PYTHONPATH نیست)
    from category_catalog import Category, CATEGORIES_DETAILED, CATEGORY_GROUPS  # type: ignore


# -----------------------------
# Optional deps
# -----------------------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModel

    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModel = None

try:
    from hazm import word_tokenize

    HAZM_AVAILABLE = True
except Exception:
    HAZM_AVAILABLE = False
    word_tokenize = None

# تلاش برای استفاده از نرمالایزر خود پروژه
try:
    from core.advanced_asr import EnhancedPersianNormalizer  # type: ignore

    NORMALIZER_AVAILABLE = True
except Exception:
    NORMALIZER_AVAILABLE = False
    EnhancedPersianNormalizer = None


# =============================================================================
# Text utilities (Persian-friendly)
# =============================================================================

_PERSIAN_LETTERS = r"\u0600-\u06FF"
_WORD_CHARS = rf"A-Za-z0-9_{_PERSIAN_LETTERS}"


def _safe_lower(s: str) -> str:
    return (s or "").lower()


def _normalize_basic(text: str) -> str:
    """
    نرمال‌سازی سبک و سریع (حتی اگر hazm/parsivar نباشند).
    """
    if not text:
        return ""

    t = text

    # unify Arabic variants to Persian
    t = (
        t.replace("ي", "ی")
        .replace("ك", "ک")
        .replace("ة", "ه")
        .replace("ؤ", "و")
        .replace("أ", "ا")
        .replace("إ", "ا")
        .replace("ٱ", "ا")
    )

    # normalize ZWNJ variants
    t = t.replace("\u200d", "\u200c")  # joiner -> ZWNJ

    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _normalize_text(text: str, normalizer: Optional[Any] = None) -> str:
    t = text or ""
    t = _normalize_basic(t)
    if normalizer is not None:
        try:
            t = normalizer.normalize(t)
        except Exception:
            pass
    return t.strip()


def _tokenize(text: str) -> List[str]:
    t = text.strip()
    if not t:
        return []
    if HAZM_AVAILABLE:
        try:
            return [x for x in word_tokenize(t) if x and x.strip()]
        except Exception:
            pass
    # fallback ساده
    return re.findall(rf"[{_WORD_CHARS}]+", t)


def _build_flexible_keyword_pattern(keyword: str) -> re.Pattern:
    """
    Regex مناسب فارسی:
    - فاصله/نیم‌فاصله بین اجزای عبارت آزاد
    - مرز کلمه بر اساس حروف فارسی/لاتین/عدد
    """
    kw = keyword.strip()
    kw = kw.replace("‌", " ")  # ZWNJ -> space for building
    parts = [re.escape(p) for p in re.split(r"\s+", kw) if p]
    if not parts:
        return re.compile(r"(?!x)x")

    mid = r"(?:[\s\u200c]+)"
    body = mid.join(parts)

    # Persian-friendly boundaries (avoid \b for ZWNJ cases)
    pattern = rf"(?<![{_WORD_CHARS}]){body}(?![{_WORD_CHARS}])"
    return re.compile(pattern, re.IGNORECASE)


# =============================================================================
# Strong Keyword Scorer
# =============================================================================

class KeywordScorer:
    """
    امتیازدهی Keyword قوی‌تر:
    - الگوی مرزکلمه‌ی مناسب فارسی
    - وزن‌دهی بر اساس طول keyword، تعداد تکرار، و weight_boost دسته
    - نرمال‌سازی یک‌دست برای متن و کلیدواژه
    """

    def __init__(self, categories: Dict[str, Category], normalizer: Optional[Any] = None):
        self.categories = categories
        self.normalizer = normalizer
        self.compiled: Dict[str, List[Tuple[re.Pattern, str]]] = {}
        self._compile()

    def _compile(self):
        for cat_name, cat in self.categories.items():
            pats: List[Tuple[re.Pattern, str]] = []
            for kw in cat.keywords:
                nkw = _normalize_text(kw, self.normalizer)
                if not nkw:
                    continue
                pats.append((_build_flexible_keyword_pattern(nkw), nkw))
            self.compiled[cat_name] = pats

    def score(self, text: str) -> Tuple[Dict[str, float], Dict[str, List[Tuple[str, int]]]]:
        """
        خروجی:
          scores_raw: امتیاز خام هر دسته
          matches: keywordهای match شده
        """
        t = _normalize_text(text, self.normalizer)
        tlow = _safe_lower(t)
        tokens = _tokenize(t)
        token_len = max(len(tokens), 1)

        scores = defaultdict(float)
        matches: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        for cat_name, pats in self.compiled.items():
            cat = self.categories[cat_name]
            cat_score = 0.0

            for pat, kw in pats:
                found = pat.findall(tlow)
                if not found:
                    continue
                count = len(found)

                # TF-like
                tf = count / token_len

                # IDF-like (تقریبی): keyword طولانی‌تر، خاص‌تر
                # به‌علاوه اگر چندکلمه‌ای باشد هم ارزش بیشتر
                word_count = max(len(re.split(r"\s+", kw.strip())), 1)
                specificity = 1.0 + (len(kw) / 12.0) + (0.35 * (word_count - 1))

                s = tf * specificity * cat.weight_boost
                cat_score += s
                matches[cat_name].append((kw, count))

            if cat_score > 0:
                scores[cat_name] += cat_score

        return dict(scores), dict(matches)


# =============================================================================
# Multilingual Embedding (E5)
# =============================================================================

class MultilingualE5Embedder:
    """
    Embedding چندزبانه با E5.
    مدل پیش‌فرض: intfloat/multilingual-e5-small
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", device: Optional[str] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers/torch not available")

        self.model_name = model_name
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self._cache: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _mean_pool(last_hidden: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: List[str], is_query: bool) -> "torch.Tensor":
        prefix = "query: " if is_query else "passage: "
        batch = [prefix + (t or "") for t in texts]

        tok = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        tok = {k: v.to(self.device) for k, v in tok.items()}

        with torch.no_grad():
            out = self.model(**tok)
            emb = self._mean_pool(out.last_hidden_state, tok["attention_mask"])

        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

    def encode_cached(self, text: str, is_query: bool) -> "torch.Tensor":
        key = ("Q:" if is_query else "P:") + text
        if key in self._cache:
            return self._cache[key]
        emb = self.encode([text], is_query=is_query)[0]
        self._cache[key] = emb
        return emb


# =============================================================================
# Hybrid strong classifier (Hierarchical + segment voting)
# =============================================================================

class StrongHybridCategorizer:
    def __init__(
        self,
        categories: Dict[str, Category],
        use_semantic: bool = True,
        semantic_model_name: str = "intfloat/multilingual-e5-small",
    ):
        self.categories = categories

        # normalizer
        self.normalizer = EnhancedPersianNormalizer() if NORMALIZER_AVAILABLE else None

        # keyword scorer
        self.keyword_scorer = KeywordScorer(categories, normalizer=self.normalizer)

        # semantic embedder
        self.use_semantic = bool(use_semantic and TRANSFORMERS_AVAILABLE)
        self.embedder: Optional[MultilingualE5Embedder] = None
        self.category_embs: Dict[str, Any] = {}
        self.group_embs: Dict[str, Any] = {}

        if self.use_semantic:
            try:
                self.embedder = MultilingualE5Embedder(model_name=semantic_model_name)
                self._build_prototypes()
            except Exception:
                self.use_semantic = False
                self.embedder = None

    def _category_prototype_text(self, cat: Category) -> str:
        kws = "، ".join(cat.keywords[:40])
        return f"{cat.name_fa}. {cat.description}. کلیدواژه‌ها: {kws}"

    def _group_prototype_text(self, group_name: str, members: List[str]) -> str:
        names = []
        kw_sample = []
        for m in members:
            c = self.categories[m]
            names.append(c.name_fa)
            kw_sample.extend(c.keywords[:6])
        kw_sample = list(dict.fromkeys(kw_sample))[:30]
        return f"گروه {group_name}. زیرگروه‌ها: {', '.join(names)}. کلیدواژه‌ها: {'، '.join(kw_sample)}"

    def _build_prototypes(self):
        assert self.embedder is not None

        for k, cat in self.categories.items():
            proto = _normalize_text(self._category_prototype_text(cat), self.normalizer)
            self.category_embs[k] = self.embedder.encode_cached(proto, is_query=False)

        for g, members in CATEGORY_GROUPS.items():
            proto = _normalize_text(self._group_prototype_text(g, members), self.normalizer)
            self.group_embs[g] = self.embedder.encode_cached(proto, is_query=False)

    @staticmethod
    def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        mx = max(scores.values())
        exps = {k: math.exp(v - mx) for k, v in scores.items()}
        s = sum(exps.values()) or 1.0
        return {k: v / s for k, v in exps.items()}

    @staticmethod
    def _topk(d: Dict[str, float], k: int) -> List[Tuple[str, float]]:
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]

    def _semantic_scores(self, text: str, candidates: List[str]) -> Dict[str, float]:
        if not self.use_semantic or not self.embedder:
            return {}

        q = _normalize_text(text, self.normalizer)
        if len(q) < 20:
            return {}

        qemb = self.embedder.encode_cached(q[:1200], is_query=True)

        sims = {}
        for c in candidates:
            cemb = self.category_embs.get(c)
            if cemb is None:
                continue
            sims[c] = float(torch.dot(qemb, cemb).item())
        return sims

    def _semantic_group_scores(self, text: str) -> Dict[str, float]:
        if not self.use_semantic or not self.embedder:
            return {}
        q = _normalize_text(text, self.normalizer)
        if len(q) < 20:
            return {}
        qemb = self.embedder.encode_cached(q[:1200], is_query=True)
        sims = {}
        for g, gemb in self.group_embs.items():
            sims[g] = float(torch.dot(qemb, gemb).item())
        return sims

    def _combine(
        self,
        kw_raw: Dict[str, float],
        sem: Dict[str, float],
        candidates: List[str],
        alpha_kw: float,
        alpha_sem: float,
    ) -> Dict[str, float]:
        out = defaultdict(float)

        kw = self._softmax(kw_raw) if kw_raw else {}
        sem_pos = {k: (v + 1.0) / 2.0 for k, v in sem.items()} if sem else {}

        for c in candidates:
            out[c] += alpha_kw * kw.get(c, 0.0)
            out[c] += alpha_sem * sem_pos.get(c, 0.0)

        s = sum(out.values()) or 1.0
        return {k: float(v / s) for k, v in out.items()}

    def _classify_single_text(self, text: str) -> Dict[str, Any]:
        t = _normalize_text(text, self.normalizer)
        if not t or len(t.strip()) < 20:
            return {
                "label": "other",
                "label_fa": self.categories["other"].name_fa,
                "confidence": 0.10,
                "top_categories": [("other", 0.10)],
                "method": "insufficient_text",
            }

        kw_raw, kw_matches = self.keyword_scorer.score(t)

        group_candidates = list(CATEGORY_GROUPS.keys())
        group_sem = self._semantic_group_scores(t) if self.use_semantic else {}

        group_kw_raw = {}
        for g, members in CATEGORY_GROUPS.items():
            group_kw_raw[g] = sum(kw_raw.get(m, 0.0) for m in members)

        group_comb = self._combine(
            kw_raw=group_kw_raw,
            sem=group_sem,
            candidates=group_candidates,
            alpha_kw=0.55,
            alpha_sem=0.45 if self.use_semantic else 0.0,
        )

        best_group = max(group_comb, key=group_comb.get) if group_comb else "other"
        top_groups = self._topk(group_comb, 2) if group_comb else [("other", 1.0)]

        candidate_cats: List[str] = []
        for g, _ in top_groups:
            candidate_cats.extend(CATEGORY_GROUPS.get(g, []))
        candidate_cats = list(dict.fromkeys(candidate_cats))
        if not candidate_cats:
            candidate_cats = list(self.categories.keys())

        sem_scores = self._semantic_scores(t, candidate_cats) if self.use_semantic else {}

        combined = self._combine(
            kw_raw=kw_raw,
            sem=sem_scores,
            candidates=candidate_cats,
            alpha_kw=0.55,
            alpha_sem=0.45 if self.use_semantic else 0.0,
        )

        if not combined:
            return {
                "label": "other",
                "label_fa": self.categories["other"].name_fa,
                "confidence": 0.15,
                "top_categories": [("other", 0.15)],
                "method": "no_signal",
            }

        top = self._topk(combined, 6)
        best_label, best_score = top[0]

        gap = (top[0][1] - top[1][1]) if len(top) >= 2 else top[0][1]
        conf = float(min(0.98, max(0.12, (best_score * 0.85) + (gap * 0.55))))

        if conf < 0.20 and best_label != "other":
            best_label = "other"
            conf = 0.20

        return {
            "label": best_label,
            "label_fa": self.categories[best_label].name_fa,
            "confidence": conf,
            "top_categories": [(k, float(v)) for k, v in top],
            "category_group": best_group,
            "method": "hybrid_semantic_keyword_hierarchical" if self.use_semantic else "keyword_hierarchical",
            "debug": {
                "top_groups": top_groups,
                "keyword_matches": kw_matches,
                "kw_raw_top": self._topk(kw_raw, 6),
                "semantic_top": self._topk(sem_scores, 6) if sem_scores else [],
            },
        }

    def classify(self, text: str, segments: Optional[List[str]] = None) -> Dict[str, Any]:
        segs = [s for s in (segments or []) if isinstance(s, str) and s.strip()]
        if segs:
            seg_preds = []
            weights = []
            for s in segs:
                pred = self._classify_single_text(s)
                seg_preds.append(pred)
                weights.append(max(20, len(s.split())))

            total_w = sum(weights) or 1.0

            agg = defaultdict(float)
            for pred, w in zip(seg_preds, weights):
                for lbl, sc in (pred.get("top_categories") or []):
                    agg[lbl] += float(sc) * (w / total_w)

            ssum = sum(agg.values()) or 1.0
            agg = {k: v / ssum for k, v in agg.items()}

            top = self._topk(agg, 8)
            best_label, best_score = top[0]

            agree = [float(p.get("confidence", 0.0)) for p in seg_preds if p.get("label") == best_label]
            agree_mean = sum(agree) / max(len(agree), 1)

            conf = float(min(0.98, max(0.15, (best_score * 0.75) + (agree_mean * 0.35))))

            full_pred = self._classify_single_text(text)
            full_top = dict(full_pred.get("top_categories") or [])
            if full_top:
                blended = defaultdict(float)
                for k, v in agg.items():
                    blended[k] += 0.70 * float(v)
                for k, v in full_top.items():
                    blended[k] += 0.30 * float(v)
                s2 = sum(blended.values()) or 1.0
                blended = {k: float(v / s2) for k, v in blended.items()}
                top = self._topk(blended, 8)
                best_label, best_score = top[0]
                conf = float(min(0.98, max(0.15, (best_score * 0.80) + (agree_mean * 0.25))))

            group = None
            for g, members in CATEGORY_GROUPS.items():
                if best_label in members:
                    group = g
                    break

            return {
                "label": best_label,
                "label_fa": self.categories[best_label].name_fa,
                "confidence": conf,
                "top_categories": [(k, float(v)) for k, v in top],
                "category_group": group,
                "category_description": self.categories[best_label].description,
                "method": "segment_voting_hybrid",
                "debug": {"segments_used": len(segs)},
            }

        pred = self._classify_single_text(text)
        label = pred.get("label", "other")
        group = pred.get("category_group")
        if group is None:
            for g, members in CATEGORY_GROUPS.items():
                if label in members:
                    group = g
                    break

        pred["category_group"] = group
        pred["category_description"] = self.categories[label].description
        return pred


# =============================================================================
# Public API (سازگار با کد شما)
# =============================================================================

class ProductionCategorizer:
    """
    کلاس اصلی برای تولید.
    """

    def __init__(self, use_ml: bool = True):
        self.categories = CATEGORIES_DETAILED
        self.engine = StrongHybridCategorizer(
            categories=self.categories,
            use_semantic=bool(use_ml),
            semantic_model_name="intfloat/multilingual-e5-small",
        )

    def classify(self, text: str, segments: Optional[List[str]] = None) -> Dict[str, Any]:
        return self.engine.classify(text=text, segments=segments)

    def get_all_categories(self) -> List[Dict[str, str]]:
        return [
            {"name": c.name, "name_fa": c.name_fa, "description": c.description}
            for c in self.categories.values()
        ]

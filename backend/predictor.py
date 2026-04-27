"""
predictor.py — 미국 뷰티 시장 적합성 예측 파이프라인

파이프라인: 규제성분게이트 → 피처 전처리 → ML 예측 → 결과 반환

ML 모델 교체 방법 (모델 파일 경로만 변경):
    pred = MarketFitPredictor(model_path='path/to/new_model.pkl')

사용 예시:
    predictor = MarketFitPredictor()
    result = predictor.predict({
        '공급가(USD)': 35.0,
        'target_category': 'skincare',
        '카테고리(중)': 'moisturizer',
        'SPF_Index': None,
        'ingredients': 'Water, Niacinamide, Glycerin, Hyaluronic Acid, ...',
    })
    # result = {
    #   'gate': {'pass': True, 'reason': '통과'},
    #   'prediction': 1,
    #   'probability': 0.7231,
    #   'threshold': 0.35,
    #   'category': 'skincare',
    # }
"""

import re
import numpy as np
import joblib
import os

# ── 기본 모델 경로 (교체 시 이 값 또는 생성자 인자 변경)
DEFAULT_MODEL_PATH = r"C:\workspace\finalproject\data\model_output\lgbm_v21.pkl"

# ── 규제 성분 (regulation.py와 동기화)
_BANNED = [
    "bithionol", "chlorofluorocarbon propellants", "chloroform",
    "halogenated salicylanilides", "hexachlorophene", "mercury compounds",
    "methylene chloride", "prohibited cattle materials",
    "vinyl chloride", "zirconium-containing complexes",
]
_SUNSCREEN_APPROVED = [
    "aminobenzoic acid", "avobenzone", "cinoxate", "dioxybenzone",
    "homosalate", "menthyl anthranilate", "octocrylene",
    "octyl methoxycinnamate", "octyl salicylate", "oxybenzone",
    "padimate o", "phenylbenzimidazole sulfonic acid", "sulisobenzone",
    "titanium dioxide", "trolamine salicylate", "zinc oxide",
]


class MarketFitPredictor:
    """
    미국 뷰티 시장 적합성 예측기

    Parameters
    ----------
    model_path : str
        lgbm_v21.pkl (또는 교체 모델) 경로.
        모델을 바꿀 때 이 인자만 수정하면 나머지 파이프라인은 그대로 동작.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        data = joblib.load(model_path)
        self.model                = data["model"]
        self.feat_cols            = data["feat_cols"]
        self.gt_map               = data["gt_map"]
        self.new_position_targets = data.get("new_position_targets", {})
        self.thresholds           = data.get("category_thresholds", {
            "skincare": 0.35, "cleansing": 0.35, "masks": 0.30, "suncare": 0.40,
        })
        # 카테고리별 학습 데이터 가격 배열 (price_rank_in_cat 계산용)
        self._price_arrays = data.get("price_arrays", {})

    # ──────────────────────────────────────────
    # STEP 1. 규제 성분 게이트
    # ──────────────────────────────────────────
    @staticmethod
    def check_regulation(ingredients_text: str, category: str) -> dict:
        """
        규제 성분 포함 여부 + 선케어 승인 성분 체크.

        Returns
        -------
        {"pass": bool, "reason": str}
        """
        text = ingredients_text.strip().lower()
        for banned in _BANNED:
            if banned in text:
                return {"pass": False, "reason": f"규제 성분 포함: {banned}"}
        if category == "suncare":
            if not any(a in text for a in _SUNSCREEN_APPROVED):
                return {"pass": False, "reason": "선크림 승인 성분 미포함"}
        return {"pass": True, "reason": "통과"}

    # ──────────────────────────────────────────
    # STEP 2. 피처 전처리
    # ──────────────────────────────────────────
    def _preprocess(self, product: dict) -> np.ndarray:
        price    = float(product.get("공급가(USD)") or 0)
        category = str(product.get("target_category") or "").lower()
        cat_mid  = str(product.get("카테고리(중)") or "").lower().strip()
        spf      = float(product.get("SPF_Index") or 0)

        raw_ing = product.get("ingredients") or ""
        if isinstance(raw_ing, list):
            raw_ing = ", ".join(str(x) for x in raw_ing if x)
        ing_lower = str(raw_ing).lower()

        # 성분 리스트 (위치 피처 계산용)
        ing_list  = [x.strip() for x in re.split(r"[,\n]", ing_lower) if x.strip()]
        total_ing = max(len(ing_list), 1)

        def find_pos(pattern: str) -> float:
            """성분 목록에서 pattern의 정규화된 위치 반환 (없으면 -1)."""
            for i, ing in enumerate(ing_list):
                if pattern in ing:
                    return 1.0 - i / total_ing
            return -1.0

        row = {}

        # ── 가격
        row["price_low"]  = int(0 < price <= 30)
        row["price_mid"]  = int(30 < price <= 71)
        row["price_high"] = int(price > 71)
        row["log_price"]  = np.log1p(price)

        arr = np.array(self._price_arrays.get(category, []))
        row["price_rank_in_cat"] = float((arr <= price).mean()) if len(arr) > 0 else 0.5

        # ── 기본
        row["SPF_Index"]       = spf
        row["ingredient_count"] = total_ing

        # ── 카테고리(대) 원핫
        for cat in ("skincare", "cleansing", "masks", "suncare"):
            row[f"cat_{cat}"] = int(category == cat)

        # ── 카테고리(중) 원핫 — feat_cols에 있는 mid_* 컬럼만
        for col in self.feat_cols:
            if col.startswith("mid_"):
                row[col] = int(cat_mid == col[4:])

        # ── GT 트렌드 성분 이진 피처
        active_gt = []
        for feat, kw in self.gt_map.items():
            present = int(kw.lower() in ing_lower)
            row[feat] = present
            if present:
                active_gt.append(feat)

        gt_in_feat = [c for c in self.gt_map if c in self.feat_cols]
        row["us_trend_ratio"] = (
            sum(row.get(c, 0) for c in gt_in_feat) / len(gt_in_feat)
            if gt_in_feat else 0.0
        )

        # ── 성분 위치 피처 (new_position_targets)
        for feat, pattern in self.new_position_targets.items():
            row[f"{feat}_position"]   = find_pos(pattern.lower())
            row[f"{feat}_above_1pct"] = -1  # phenoxyethanol 기준 생략

        # ── 레거시 위치 피처
        legacy_pos_map = {
            "niacinamide_position":           "niacinamide",
            "ceramide_position":              "ceramide",
            "hyaluronic_acid_position":       "hyaluronic acid",
            "zinc_oxide_position":            "zinc oxide",
            "centella_position":              "centella",
            "amino_acid_surfactant_position": "cocoyl",
            "niacinamide_above_1pct":         None,
            "ceramide_above_1pct":            None,
            "hyaluronic_acid_above_1pct":     None,
        }
        for col, pattern in legacy_pos_map.items():
            if col not in self.feat_cols:
                continue
            row[col] = find_pos(pattern) if pattern else -1

        # us_trend_ingredient_position: 가장 앞쪽에 나온 트렌드 성분의 위치
        if "us_trend_ingredient_position" in self.feat_cols:
            positions = [find_pos(kw.lower()) for kw in self.gt_map.values()]
            valid = [p for p in positions if p > 0]
            row["us_trend_ingredient_position"] = max(valid) if valid else -1
        if "us_trend_ingredient_above_1pct" in self.feat_cols:
            row["us_trend_ingredient_above_1pct"] = -1

        # ── 클렌징 전용 피처
        row["is_sulfate"] = int(any(x in ing_lower for x in (
            "sodium lauryl sulfate", "sodium laureth sulfate",
            "ammonium lauryl sulfate", "ammonium laureth sulfate",
        )))
        row["is_amino_surfactant"] = int(any(x in ing_lower for x in (
            "sodium cocoyl", "potassium cocoyl", "sodium lauroyl",
            "cocoamidopropyl betaine", "cocamidopropyl betaine",
        )))
        row["is_low_ph_acid"] = int(any(x in ing_lower for x in (
            "lactic acid", "gluconolactone", "lactobionic acid", "polyhydroxy",
        )))
        row["surfactant_premium_score"] = int(
            row["is_amino_surfactant"] == 1 and row["is_sulfate"] == 0
        )
        row["is_mild_surfactant"] = int(any(x in ing_lower for x in (
            "cocobetaine", "coco-betaine", "lauryl glucoside", "coco glucoside",
            "decyl glucoside", "caprylyl glucoside",
            "disodium cocoamphodiacetate", "sodium cocoamphoacetate",
        )))
        row["is_foam_cleanser"] = int("foam" in cat_mid and category == "cleansing")
        row["is_gel_cleanser"]  = int("gel"  in cat_mid and category == "cleansing")
        row["is_salicylic"]     = int("salicylic acid" in ing_lower)

        # ── 선케어 전용 피처
        row["is_physical_filter"] = int(
            "zinc oxide" in ing_lower or "titanium dioxide" in ing_lower
        )
        row["has_chemical_filter"] = int(any(x in ing_lower for x in (
            "avobenzone", "octinoxate", "oxybenzone",
            "octisalate", "homosalate", "octocrylene",
        )))
        row["is_chemical_only"] = int(
            row["has_chemical_filter"] == 1 and row["is_physical_filter"] == 0
        )
        row["spf_tier"] = (
            3 if spf > 50 else 2 if spf > 30 else 1 if spf > 0 else 0
        )

        # citric acid position
        row["citric_acid_position"] = find_pos("citric acid")

        # ── 교호작용: category × 신호
        interact_pairs = {
            "interact_cleansing_amino":       ("cat_cleansing", "is_amino_surfactant"),
            "interact_cleansing_mild":        ("cat_cleansing", "is_mild_surfactant"),
            "interact_cleansing_salicylic":   ("cat_cleansing", "is_salicylic"),
            "interact_cleansing_sha":         ("cat_cleansing", "gt_sodium_hyaluronate"),
            "interact_cleansing_panthenol":   ("cat_cleansing", "gt_panthenol"),
            "interact_cleansing_niacinamide": ("cat_cleansing", "gt_niacinamide"),
            "interact_cleansing_trend":       ("cat_cleansing", "us_trend_ratio"),
            "interact_cleansing_price_low":   ("cat_cleansing", "price_low"),
            "interact_cleansing_price_mid":   ("cat_cleansing", "price_mid"),
            "interact_cleansing_citric":      ("cat_cleansing", "citric_acid_position"),
            "interact_skincare_trend":        ("cat_skincare",  "us_trend_ratio"),
            "interact_skincare_price_high":   ("cat_skincare",  "price_high"),
            "interact_suncare_physical":      ("cat_suncare",   "is_physical_filter"),
            "interact_suncare_spf_tier":      ("cat_suncare",   "spf_tier"),
            "interact_suncare_price_high":    ("cat_suncare",   "price_high"),
            "interact_suncare_price_mid":     ("cat_suncare",   "price_mid"),
            "interact_masks_trend":           ("cat_masks",     "us_trend_ratio"),
            "interact_masks_price_low":       ("cat_masks",     "price_low"),
            "interact_masks_price_mid":       ("cat_masks",     "price_mid"),
        }
        for col, (a, b) in interact_pairs.items():
            row[col] = row.get(a, 0) * row.get(b, 0)

        # ── ingredient_count × category
        for cat in ("skincare", "cleansing", "masks", "suncare"):
            row[f"interact_ic_{cat}"] = row["ingredient_count"] * row.get(f"cat_{cat}", 0)

        # ── log_price × category
        for cat in ("skincare", "cleansing", "masks", "suncare"):
            row[f"interact_logprice_{cat}"] = row["log_price"] * row.get(f"cat_{cat}", 0)

        # ── us_trend_ratio × price bin
        row["interact_trend_price_low"]  = row["us_trend_ratio"] * row["price_low"]
        row["interact_trend_price_mid"]  = row["us_trend_ratio"] * row["price_mid"]
        row["interact_trend_price_high"] = row["us_trend_ratio"] * row["price_high"]

        # ── position × category
        pos_cat_pairs = {
            "pos_clean_niacinamide": ("niacinamide_position",           "cat_cleansing"),
            "pos_clean_amino":       ("amino_acid_surfactant_position",  "cat_cleansing"),
            "pos_clean_trend":       ("us_trend_ingredient_position",    "cat_cleansing"),
            "pos_skin_niacinamide":  ("niacinamide_position",            "cat_skincare"),
            "pos_skin_ceramide":     ("ceramide_position",               "cat_skincare"),
            "pos_skin_ha":           ("hyaluronic_acid_position",        "cat_skincare"),
            "pos_skin_tocopheryl":   ("tocopheryl_acetate_position",     "cat_skincare"),
            "pos_skin_retinol":      ("retinol_position",                "cat_skincare"),
            "pos_sun_zinc":          ("zinc_oxide_position",             "cat_suncare"),
            "pos_sun_trend":         ("us_trend_ingredient_position",    "cat_suncare"),
            "pos_mask_ha":           ("hyaluronic_acid_position",        "cat_masks"),
            "pos_mask_niacinamide":  ("niacinamide_position",            "cat_masks"),
        }
        for col, (pos_col, cat_col) in pos_cat_pairs.items():
            row[col] = row.get(cat_col, 0) * row.get(pos_col, 0)

        # ── feat_cols 순서대로 벡터 구성 (없는 피처는 0으로 채움)
        return np.array([row.get(c, 0) for c in self.feat_cols], dtype=float)

    # ──────────────────────────────────────────
    # STEP 3. 전체 파이프라인
    # ──────────────────────────────────────────
    def predict(self, product: dict) -> dict:
        """
        단일 제품 시장 적합성 예측.

        Parameters
        ----------
        product : dict
            {
                '공급가(USD)': float,
                'target_category': str,   # 'skincare'|'cleansing'|'masks'|'suncare'
                '카테고리(중)': str,       # 예: 'moisturizer', 'foam cleanser'
                'SPF_Index': float|None,
                'ingredients': str|list,  # 성분 텍스트 또는 리스트
            }

        Returns
        -------
        dict
            {
                'gate': {'pass': bool, 'reason': str},
                'prediction': int|None,   # 1=시장적합, 0=부적합, None=게이트 실패
                'probability': float|None,
                'threshold': float|None,
                'category': str,
            }
        """
        category = str(product.get("target_category") or "").lower()
        raw_ing  = product.get("ingredients") or ""
        if isinstance(raw_ing, list):
            raw_ing = ", ".join(str(x) for x in raw_ing if x)

        # 1. 규제 성분 게이트
        gate = self.check_regulation(raw_ing, category)
        if not gate["pass"]:
            return {
                "gate": gate, "prediction": None,
                "probability": None, "score": None,
                "threshold": None, "category": category,
            }

        # 2. 전처리 + 예측
        vec        = self._preprocess(product).reshape(1, -1)
        threshold  = self.thresholds.get(category, 0.50)
        proba      = float(self.model.predict_proba(vec)[0, 1])
        prediction = int(proba >= threshold)

        return {
            "gate":        gate,
            "prediction":  prediction,
            "probability": round(proba, 4),
            "score":       round(proba * 100, 1),   # 0~100점
            "threshold":   threshold,
            "category":    category,
        }

    def predict_batch(self, products: list) -> list:
        """여러 제품 일괄 예측."""
        return [self.predict(p) for p in products]


# ── 간단 동작 테스트
if __name__ == "__main__":
    pred = MarketFitPredictor()

    examples = [
        {
            "name": "스킨케어 (니아신아마이드 포함)",
            "공급가(USD)": 38.0,
            "target_category": "skincare",
            "카테고리(중)": "moisturizer",
            "SPF_Index": None,
            "ingredients": "Water, Niacinamide, Glycerin, Hyaluronic Acid, Ceramide, Peptide",
        },
        {
            "name": "클렌징 (규제 성분 포함)",
            "공급가(USD)": 25.0,
            "target_category": "cleansing",
            "카테고리(중)": "foam cleanser",
            "SPF_Index": None,
            "ingredients": "Water, Chloroform, Glycerin",
        },
        {
            "name": "선케어 (승인 성분 포함)",
            "공급가(USD)": 42.0,
            "target_category": "suncare",
            "카테고리(중)": "sunscreen",
            "SPF_Index": 50.0,
            "ingredients": "Water, Zinc Oxide, Titanium Dioxide, Glycerin, Niacinamide",
        },
        {
            "name": "마스크 (저가)",
            "공급가(USD)": 12.0,
            "target_category": "masks",
            "카테고리(중)": "sheet mask",
            "SPF_Index": None,
            "ingredients": "Water, Hyaluronic Acid, Centella Asiatica, Glycerin",
        },
    ]

    for ex in examples:
        name = ex.pop("name")
        result = pred.predict(ex)
        label  = "✅ 적합" if result["prediction"] == 1 else (
                 "❌ 부적합" if result["prediction"] == 0 else "🚫 게이트 차단"
        )
        print(f"\n[{name}]")
        print(f"  게이트  : {result['gate']['reason']}")
        print(f"  점수    : {result['score']}점  (threshold={result['threshold']})")
        print(f"  결과    : {label}")

"""
predictor.py — 미국 뷰티 시장 적합성 점수 산출

파이프라인: 피처 전처리 → ML 예측 → 결과 반환
규제 성분 체크는 UI 단계에서 사전 처리됨 (이 파일 담당 아님)

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
    #   'prediction': 1,         # 1=시장적합, 0=부적합
    #   'probability': 0.7231,
    #   'score': 72.3,           # 0~100점
    #   'threshold': 0.35,
    #   'category': 'skincare',
    # }
"""

import os
import re
import numpy as np
import joblib
import shap

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "lgbm_final.pkl")


class MarketFitPredictor:
    """
    미국 뷰티 시장 적합성 예측기

    Parameters
    ----------
    model_path : str
        lgbm_final.pkl (또는 교체 모델) 경로.
        모델을 바꿀 때 이 인자만 수정하면 나머지 파이프라인은 그대로 동작.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        data = joblib.load(model_path)
        self.model                = data["model"]
        self.feat_cols            = data["feat_cols"]
        self.gt_map               = data["gt_map"]
        self.new_position_targets = data.get("new_position_targets", {})
        self.thresholds           = data.get("category_thresholds", {
            "skincare": 0.35, "cleansing": 0.40, "masks": 0.35, "suncare": 0.45,
        })
        self._price_arrays = data.get("price_arrays", {})
        self._explainer = shap.TreeExplainer(self.model)

    # 액티브 성분 집합 (top5_active_count / top30pct_active_count 계산용)
    _ACTIVE_SET = {s.lower() for s in (
        {'Niacinamide', 'Nicotinamide', 'Vitamin B3'} |
        {'Ceramide', 'Ceramide NP', 'Ceramide AP', 'Ceramide EOP', 'Ceramide NG',
         'Ceramide NS', 'Ceramide EOS', 'Ceramide 1', 'Ceramide 2', 'Ceramide 3', 'Ceramide 6 II'} |
        {'Sodium Hyaluronate', 'Hyaluronic Acid', 'Hyaluronic acid',
         'Hydrolyzed Hyaluronic Acid', 'Sodium Acetylated Hyaluronate',
         'Hydroxypropyltrimonium Hyaluronate', 'Sodium Hyaluronate Crosspolymer',
         'Hydrolyzed Sodium Hyaluronate'} |
        {'Zinc Oxide', 'Zinc oxide'} |
        {'Centella Asiatica Extract', 'Centella asiatica Extract', 'Asiaticoside',
         'Madecassoside', 'Asiatic Acid', 'Madecassic Acid', 'Centella Asiatica Leaf Extract'} |
        {'PDRN', 'Polydeoxyribonucleotide', 'Pdrn', 'Pdrn Sodium Dna',
         'PDRN Sodium DNA', 'Salmon DNA', 'Salmon PDRN'} |
        {'Sodium Cocoyl Glycinate', 'Sodium Lauroyl Glutamate', 'Sodium Cocoyl Glutamate',
         'Disodium Cocoyl Glutamate', 'Potassium Cocoyl Glycinate',
         'Sodium Methyl Cocoyl Taurate', 'Sodium Lauroyl Methyl Isethionate',
         'Sodium Lauroyl Oat Amino Acids'} |
        {'Retinol', 'Bakuchiol', 'Tranexamic Acid', 'Alpha-Arbutin', 'Ascorbic Acid',
         '3-O-Ethyl Ascorbic Acid', 'Ascorbyl Glucoside', 'Adenosine', 'Peptides',
         'Palmitoyl Tripeptide-1', 'Palmitoyl Tetrapeptide-7',
         'Galactomyces Ferment Filtrate', 'Bifida Ferment Lysate'}
    )}

    # ──────────────────────────────────────────
    # 피처 전처리
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

        ing_list  = [x.strip() for x in re.split(r"[,\n]", ing_lower) if x.strip()]
        total_ing = max(len(ing_list), 1)

        def find_pos(pattern: str) -> float:
            for i, ing in enumerate(ing_list):
                if pattern in ing:
                    return 1.0 - i / total_ing
            return -1.0

        row = {}

        # 가격
        row["price_low"]  = int(0 < price <= 30)
        row["price_mid"]  = int(30 < price <= 71)
        row["price_high"] = int(price > 71)
        row["log_price"]  = np.log1p(price)

        arr = np.array(self._price_arrays.get(category, []))
        row["price_rank_in_cat"] = float((arr <= price).mean()) if len(arr) > 0 else 0.5

        row["SPF_Index"]        = spf
        row["ingredient_count"] = total_ing

        # 카테고리(대) 원핫
        for cat in ("skincare", "cleansing", "masks", "suncare"):
            row[f"cat_{cat}"] = int(category == cat)

        # 카테고리(중) 원핫
        for col in self.feat_cols:
            if col.startswith("mid_"):
                row[col] = int(cat_mid == col[4:])

        # GT 트렌드 성분
        for feat, kw in self.gt_map.items():
            row[feat] = int(kw.lower() in ing_lower)

        gt_in_feat = [c for c in self.gt_map if c in self.feat_cols]
        row["us_trend_ratio"] = (
            sum(row.get(c, 0) for c in gt_in_feat) / len(gt_in_feat)
            if gt_in_feat else 0.0
        )

        # 성분 위치 피처
        for feat, pattern in self.new_position_targets.items():
            row[f"{feat}_position"]   = find_pos(pattern.lower())
            row[f"{feat}_above_1pct"] = -1

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

        if "us_trend_ingredient_position" in self.feat_cols:
            positions = [find_pos(kw.lower()) for kw in self.gt_map.values()]
            valid = [p for p in positions if p > 0]
            row["us_trend_ingredient_position"] = max(valid) if valid else -1
        if "us_trend_ingredient_above_1pct" in self.feat_cols:
            row["us_trend_ingredient_above_1pct"] = -1

        # top5_active_count / top30pct_active_count
        ing_original = [x.strip() for x in re.split(r"[,\n]", str(product.get("ingredients") or "")) if x.strip()]
        n_ing = max(len(ing_original), 1)
        row["top5_active_count"]    = sum(1 for v in ing_original[:5] if v.lower() in self._ACTIVE_SET)
        row["top30pct_active_count"] = sum(1 for v in ing_original[:max(1, int(n_ing * 0.3))] if v.lower() in self._ACTIVE_SET)

        # 클렌징 전용
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

        # 선케어 전용
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

        row["citric_acid_position"] = find_pos("citric acid")

        # 교호작용
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

        for cat in ("skincare", "cleansing", "masks", "suncare"):
            row[f"interact_ic_{cat}"]       = row["ingredient_count"] * row.get(f"cat_{cat}", 0)
            row[f"interact_logprice_{cat}"] = row["log_price"]        * row.get(f"cat_{cat}", 0)

        row["interact_trend_price_low"]  = row["us_trend_ratio"] * row["price_low"]
        row["interact_trend_price_mid"]  = row["us_trend_ratio"] * row["price_mid"]
        row["interact_trend_price_high"] = row["us_trend_ratio"] * row["price_high"]

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

        return np.array([row.get(c, 0) for c in self.feat_cols], dtype=float)

    # ──────────────────────────────────────────
    # 예측
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
                '카테고리(중)': str,
                'SPF_Index': float|None,
                'ingredients': str|list,
            }

        Returns
        -------
        dict
            {
                'prediction': int,    # 1=시장적합, 0=부적합
                'probability': float,
                'score': float,       # 0~100점
                'threshold': float,
                'category': str,
            }
        """
        vec   = self._preprocess(product).reshape(1, -1)
        proba = float(self.model.predict_proba(vec)[0, 1])

        # SHAP 상위 3개 피처
        shap_vals = self._explainer.shap_values(vec)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        else:
            sv = shap_vals[0]
        top_idx = np.argsort(np.abs(sv))[::-1][:10]
        top_features = [
            {"feature": self.feat_cols[i], "shap_value": round(float(sv[i]), 4)}
            for i in top_idx
        ]

        return {
            "score":        round(proba * 100, 1),
            "top_features": top_features,
        }

    def predict_batch(self, products: list) -> list:
        """여러 제품 일괄 예측."""
        return [self.predict(p) for p in products]


if __name__ == "__main__":
    pred = MarketFitPredictor()

    examples = [
        {
            "name": "스킨케어",
            "공급가(USD)": 38.0,
            "target_category": "skincare",
            "카테고리(중)": "moisturizer",
            "SPF_Index": None,
            "ingredients": "Water, Niacinamide, Glycerin, Hyaluronic Acid, Ceramide, Peptide",
        },
        {
            "name": "선케어",
            "공급가(USD)": 42.0,
            "target_category": "suncare",
            "카테고리(중)": "sunscreen",
            "SPF_Index": 50.0,
            "ingredients": "Water, Zinc Oxide, Titanium Dioxide, Glycerin, Niacinamide",
        },
        {
            "name": "마스크",
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
        label = "적합" if result["prediction"] == 1 else "부적합"
        print(f"[{name}] {result['score']}점 ({label}, threshold={result['threshold']})")

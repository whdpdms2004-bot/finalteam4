"""
predictor.py — 미국 뷰티 시장 적합성 점수 산출 (lgbm_v99 기준)

파이프라인: 피처 전처리 → ML 예측 → SHAP → 결과 반환
모델 교체: MarketFitPredictor(model_path='path/to/new.pkl')
"""

import os
import re
import numpy as np
import joblib
import shap

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "data", "model_output", "lgbm_v99.pkl"
)

# gt_trend_stats 계산용 매핑: (성분 감지 패턴, gt_trend_stats 키)
# 감지 패턴은 ingredient string 검색용, stats 키는 gt_trend_stats dict 조회용
_GT_KW_MAP = {
    "gt_niacinamide":        ("niacinamide",        "niacinamide"),
    "gt_ceramide":           ("ceramide",           "ceramide"),
    "gt_sodium_hyaluronate": ("sodium hyaluronate", "hyaluronic acid"),
    "gt_retinol":            ("retinol",            "retinol"),
    "gt_peptide":            ("peptide",            "peptide"),
    "gt_vitamin_c":          ("ascorbic",           "vitamin c"),
    "gt_panthenol":          ("panthenol",          "panthenol"),
    "gt_centella":           ("centella",           "centella asiatica"),
    "gt_bakuchiol":          ("bakuchiol",          "bakuchiol"),
    "gt_azelaic":            ("azelaic acid",       "azelaic acid"),
    "gt_tranexamic":         ("tranexamic acid",    "tranexamic acid"),
    "gt_salicylic":          ("salicylic acid",     "salicylic acid"),
    "gt_squalane":           ("squalane",           "squalane"),
}


class MarketFitPredictor:
    """
    미국 뷰티 시장 적합성 예측기

    Parameters
    ----------
    model_path : str
        lgbm_v99.pkl 경로.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        data = joblib.load(model_path)
        self.model            = data["model"]
        self.feat_cols        = data["feat_cols"]
        self.gt_map           = data["gt_map"]
        self.position_targets = data.get("position_targets", data.get("new_position_targets", {}))
        self.price_arrays     = data.get("price_arrays", {})
        self.thresholds       = data.get("cat_thresholds",
                                  data.get("category_thresholds", {
                                      "skincare": 0.36, "cleansing": 0.44,
                                      "masks": 0.46,   "suncare": 0.55,
                                  }))
        self.gt_trend_stats   = data.get("gt_trend_stats", {})
        self._explainer = shap.TreeExplainer(self.model)

    # ──────────────────────────────────────────────────────────
    # 피처 전처리
    # ──────────────────────────────────────────────────────────
    def _preprocess(self, product: dict) -> np.ndarray:
        price    = float(product.get("공급가(USD)") or 0)
        category = str(product.get("target_category") or "").lower()
        cat_mid  = str(product.get("카테고리(중)") or "").lower().strip()
        spf      = float(product.get("SPF_Index") or 0)

        raw_ing   = product.get("ingredients") or ""
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

        # ── 가격 ──────────────────────────────────────────────
        row["price_low"]  = int(0 < price <= 30)
        row["price_mid"]  = int(30 < price <= 71)
        row["price_high"] = int(price > 71)
        row["log_price"]  = np.log1p(price)
        arr = np.array(self.price_arrays.get(category, []))
        row["price_rank_in_cat"] = float((arr <= price).mean()) if len(arr) > 0 else 0.5

        # ── SPF ───────────────────────────────────────────────
        row["SPF_Index"] = spf
        row["spf_tier"]  = 3 if spf > 50 else 2 if spf > 30 else 1 if spf > 0 else 0
        row["spf_50plus"] = int(spf > 50)

        # ── 카테고리 (대분류 중간 변수) ────────────────────────
        cat_skincare  = int(category == "skincare")
        cat_cleansing = int(category == "cleansing")
        cat_masks     = int(category == "masks")
        cat_suncare   = int(category == "suncare")
        row["cat_masks"] = cat_masks  # feat_cols에 포함된 유일한 cat_* 직접 피처

        # ── 중분류 (mid_*) ─────────────────────────────────────
        for col in self.feat_cols:
            if col.startswith("mid_"):
                row[col] = int(cat_mid == col[4:])

        # ── GT 트렌드 성분 이진 피처 ──────────────────────────
        gt_hits = {}
        for feat, kw in self.gt_map.items():
            gt_hits[feat] = int(kw.lower() in ing_lower)
            row[feat] = gt_hits[feat]

        gt_in_feat = [c for c in self.gt_map if c in self.feat_cols]
        row["us_trend_count"] = sum(gt_hits.get(c, 0) for c in gt_in_feat)
        row["us_trend_ratio"] = row["us_trend_count"] / len(gt_in_feat) if gt_in_feat else 0.0

        top5_kws = ["niacinamide", "ceramide", "retinol", "centella", "bakuchiol"]
        row["top5_trend_count"] = sum(int(kw in ing_lower) for kw in top5_kws)

        # ── 성분 위치 (pos_*) ─────────────────────────────────
        pos = {}
        for feat, pattern in self.position_targets.items():
            pos[feat] = find_pos(pattern.lower())
            row[f"pos_{feat}"] = pos[feat]

        # pos_* × category 교호작용
        pos_cat_pairs = {
            "niacinamide_x_skincare":     ("niacinamide",      cat_skincare),
            "niacinamide_x_cleansing":    ("niacinamide",      cat_cleansing),
            "niacinamide_x_masks":        ("niacinamide",      cat_masks),
            "ceramide_x_skincare":        ("ceramide",         cat_skincare),
            "ceramide_x_masks":           ("ceramide",         cat_masks),
            "hyaluronic_acid_x_skincare": ("hyaluronic_acid",  cat_skincare),
            "retinol_x_skincare":         ("retinol",          cat_skincare),
            "salicylic_acid_x_cleansing": ("salicylic_acid",   cat_cleansing),
            "zinc_oxide_x_suncare":       ("zinc_oxide",       cat_suncare),
            "vitamin_c_x_skincare":       ("vitamin_c",        cat_skincare),
            "homosalate_x_suncare":       ("homosalate",       cat_suncare),
        }
        for suffix, (pos_key, cat_val) in pos_cat_pairs.items():
            row[f"pos_{suffix}"] = max(pos.get(pos_key, -1), 0) * cat_val

        # ── 계면활성제 / 성분 플래그 ──────────────────────────
        row["is_amino_surfactant"] = int(any(x in ing_lower for x in (
            "sodium cocoyl", "potassium cocoyl", "sodium lauroyl",
            "cocoamidopropyl betaine", "cocamidopropyl betaine",
        )))
        row["is_sulfate"] = int(any(x in ing_lower for x in (
            "sodium lauryl sulfate", "sodium laureth sulfate",
            "ammonium lauryl sulfate", "ammonium laureth sulfate",
        )))
        row["is_physical_filter"] = int("zinc oxide" in ing_lower or "titanium dioxide" in ing_lower)
        row["is_chemical_filter"] = int(any(x in ing_lower for x in (
            "avobenzone", "octinoxate", "oxybenzone", "octisalate",
            "homosalate", "octocrylene", "ethylhexyl methoxycinnamate",
        )))
        row["is_hybrid_filter"] = int(row["is_physical_filter"] == 1 and row["is_chemical_filter"] == 1)

        row["combo_adenosine_niacinamide"] = int("adenosine" in ing_lower and "niacinamide" in ing_lower)
        row["combo_vitc_tocopherol"]       = int("ascorbic" in ing_lower and "tocopherol" in ing_lower)

        row["has_retinol"] = int("retinol" in ing_lower)
        row["has_ferment"] = int(any(x in ing_lower for x in (
            "ferment", "bifida", "lactobacillus", "galactomyces",
        )))

        # ── 클렌징 전용 피처 ───────────────────────────────────
        row["is_premium_amino_surf"] = int(row["is_amino_surfactant"] == 1 and row["is_sulfate"] == 0)
        row["surfactant_grade"] = (
            3 if (row["is_amino_surfactant"] == 1 and row["is_sulfate"] == 0) else
            2 if any(x in ing_lower for x in ("glucoside", "betaine")) else
            1 if row["is_sulfate"] == 0 else 0
        )
        row["is_treatment_cleanser"]   = int("salicylic acid" in ing_lower and category == "cleansing")
        row["has_barrier_in_cleanser"] = int(
            category == "cleansing" and ("ceramide" in ing_lower or "panthenol" in ing_lower)
        )
        row["interact_premium_surf_price"] = row["is_premium_amino_surf"] * price

        # ── GT 트렌드 통계 피처 ────────────────────────────────
        gts = self.gt_trend_stats
        # matched: (detect_kw, stats_key) pairs where ingredient is detected AND stats exist
        matched = [
            (det, stat)
            for col, (det, stat) in _GT_KW_MAP.items()
            if det in ing_lower and stat in gts
        ]
        n_m = max(len(matched), 1)
        if matched and gts:
            row["trend_max_recent"]   = max(gts[sk]["recent_12m"] for _, sk in matched)
            row["trend_avg_recent"]   = sum(gts[sk]["recent_12m"] for _, sk in matched) / n_m
            row["trend_rising_count"] = sum(1 for _, sk in matched if gts[sk]["slope_12m"] > 0)
            row["trend_avg_slope"]    = sum(gts[sk]["slope_12m"] for _, sk in matched) / n_m
            tpw_map = {
                "ceramide":           "pos_ceramide",
                "sodium hyaluronate": "pos_hyaluronic_acid",
                "niacinamide":        "pos_niacinamide",
            }
            matched_stats = {det: sk for det, sk in matched}
            tpw = sum(
                max(row.get(pf, -1), 0) * gts[matched_stats[det]]["recent_12m"]
                for det, pf in tpw_map.items()
                if det in matched_stats
            )
            row["trend_pos_weighted"] = tpw
        else:
            row["trend_max_recent"]   = 0.0
            row["trend_avg_recent"]   = 0.0
            row["trend_rising_count"] = 0
            row["trend_avg_slope"]    = 0.0
            row["trend_pos_weighted"] = 0.0

        # ── 교호작용 피처 ──────────────────────────────────────
        row["interact_logprice_skincare"]  = row["log_price"] * cat_skincare
        row["interact_logprice_cleansing"] = row["log_price"] * cat_cleansing
        row["interact_logprice_masks"]     = row["log_price"] * cat_masks
        row["interact_logprice_suncare"]   = row["log_price"] * cat_suncare

        row["interact_trend_skincare"]  = row["us_trend_ratio"] * cat_skincare
        row["interact_trend_cleansing"] = row["us_trend_ratio"] * cat_cleansing

        row["interact_physical_suncare"]     = row["is_physical_filter"] * cat_suncare
        row["interact_spf_suncare"]          = spf * cat_suncare
        row["interact_top5trend_cleansing"]  = row["top5_trend_count"] * cat_cleansing
        row["interact_hyaluronic_cleansing"] = row.get("gt_sodium_hyaluronate", 0) * cat_cleansing

        row["interact_chemical_suncare"]       = row["is_chemical_filter"] * cat_suncare
        row["interact_hybrid_suncare"]         = row["is_hybrid_filter"] * cat_suncare
        row["interact_physical_price_suncare"] = row["is_physical_filter"] * price * cat_suncare

        is_serum = int(any(x in cat_mid for x in ("serum", "ampoule", "essence")))
        is_cream = int("cream" in cat_mid)
        row["interact_serum_retinol"]  = is_serum * row["has_retinol"]
        row["interact_serum_peptide"]  = is_serum * row.get("gt_peptide", 0)
        row["interact_cream_ceramide"] = is_cream * max(pos.get("ceramide", -1), 0)

        return np.array([row.get(c, 0) for c in self.feat_cols], dtype=float)

    # ──────────────────────────────────────────────────────────
    # 예측
    # ──────────────────────────────────────────────────────────
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
                'score': float,        # 0~100점 (percentile 기반)
                'prediction': int,     # 1=시장적합, 0=부적합
                'threshold': float,
                'top_features': list,
            }
        """
        vec   = self._preprocess(product).reshape(1, -1)
        proba = float(self.model.predict_proba(vec)[0, 1])

        shap_vals = self._explainer.shap_values(vec)
        sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        top_idx = np.argsort(np.abs(sv))[::-1][:10]
        top_features = [
            {"feature": self.feat_cols[i], "shap_value": round(float(sv[i]), 4)}
            for i in top_idx
        ]

        category  = str(product.get("target_category") or "").lower()
        threshold = self.thresholds.get(category, 0.5)

        return {
            "score":        round(proba * 100, 1),
            "prediction":   int(proba >= threshold),
            "threshold":    threshold,
            "top_features": top_features,
        }

    def predict_batch(self, products: list) -> list:
        """여러 제품 일괄 예측."""
        return [self.predict(p) for p in products]


if __name__ == "__main__":
    pred = MarketFitPredictor()

    examples = [
        {
            "name": "스킨케어 (belif 스타일)",
            "공급가(USD)": 48.0,
            "target_category": "skincare",
            "카테고리(중)": "cream",
            "SPF_Index": None,
            "ingredients": "Water, Glycerin, Panthenol, Ceramide NP, Centella Asiatica Extract, Niacinamide, Sodium Hyaluronate",
        },
        {
            "name": "선케어",
            "공급가(USD)": 42.0,
            "target_category": "suncare",
            "카테고리(중)": "sun cream",
            "SPF_Index": 50.0,
            "ingredients": "Water, Zinc Oxide, Titanium Dioxide, Glycerin, Niacinamide",
        },
    ]

    for ex in examples:
        name = ex.pop("name")
        result = pred.predict(ex)
        label = "적합" if result["prediction"] == 1 else "부적합"
        print(f"[{name}] {result['score']}점 ({label}, threshold={result['threshold']})")

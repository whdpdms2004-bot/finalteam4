# form 값과 predictor 쪽에서 쓰는 키 매핑
# predictor : 공급가(USD), target_category(skincare, suncare, cleansing, masks),
# 카테고리(중)(직접 확인 필요), SPF_Index, ingredients 

from typing import Dict, Any

CATEGORY_MAP = {
    "스킨케어": "skincare",
    "클렌징": "cleansing",
    "마스크팩": "masks",
    "선케어": "suncare",
}

SUBCATEGORY_MAP = {
    # 스킨케어
    "스킨": "skin/toner",
    "토너": "skin/toner",
    "패드": "skin/toner",
    "에센스": "essence/serum/ampoule",
    "세럼": "essence/serum/ampoule",
    "앰플": "essence/serum/ampoule",
    "스팟케어": "essence/serum/ampoule",
    "로션": "lotion",
    "에멀젼": "lotion",
    "크림": "cream",
    "올인원": "cream",
    "오일": "cream",
    "슬리핑팩": "cream",
    "젤": "cream",
    "밤/멀티밤": "cream",
    "아이크림": "eye care",
    "미스트": "mist/fixer",
    "페이스에센스": "mist/fixer",

    # 선케어
    "선크림": "sun cream",
    "선스프레이": "sun cream",
    "선로션": "sun cream",
    "선스틱": "sun stick",

    # 클렌징
    "클렌징폼": "cleansing foam/gel",
    "클렌징젤": "cleansing foam/gel",
    "클렌징비누": "cleansing foam/gel",
    "클렌징파우더": "cleansing foam/gel",
    "클렌징오일": "cleansing oil",
    "클렌징워터": "cleansing oil",
    "클렌징밤": "cleansing balm",
    "클렌징밀크": "cleansing milk/cream",
    "클렌징크림": "cleansing milk/cream",
    "클렌징로션": "cleansing milk/cream",
    "클렌징티슈": "cleansing tissue/pad",
    "클렌징패드": "cleansing tissue/pad",
    "필링/스크럽": "cleansing foam/gel",
    "스크럽/필링패드": "cleansing foam/gel",
    "립": "lip&eye remover",
    "아이리무버": "lip&eye remover",

    # 마스크팩
    "시트마스크": "sheet mask",
    "워시오프팩": "wash-off pack",
    "필오프팩": "wash-off pack",
    "모델링/고무팩": "wash-off pack",
    "패치": "patch",
    "부분마스크/팩": "patch",
}

def map_front_to_predictor(form: Dict[str, Any]) -> Dict[str, Any]:
    """
    프론트 form JSON -> predictor 입력 dict로 변환
    """
    # 1) 대분류 매핑
    raw_cat = form.get("category", "")
    target_category = CATEGORY_MAP.get(raw_cat, "").lower()

    # 2) 소분류 매핑
    raw_sub = form.get("subCategory", "")
    mid = SUBCATEGORY_MAP.get(raw_sub, "").lower()

    # 3) 가격
    price_raw = form.get("price")
    try:
        price = float(price_raw) if price_raw is not None else 0.0
    except (ValueError, TypeError):
        price = 0.0


    # 4) SPF
    spf_raw = form.get("spf")
    try:
        spf = float(spf_raw) if spf_raw not in (None, "") else 0.0
    except ValueError:
        spf = 0.0

    # 5) 성분
    ingredients = form.get("ingredients") or ""

    return {
        "공급가(USD)": price,
        "target_category": target_category,
        "카테고리(중)": mid,
        "SPF_Index": spf,
        "ingredients": ingredients,
    }
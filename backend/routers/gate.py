from fastapi import APIRouter

router = APIRouter(prefix="/gate", tags=["gate"])

# 규제 성분 리스트
banned_list = [
    "bithionol", "chlorofluorocarbon propellants", "chloroform",
    "halogenated salicylanilides", "hexachlorophene", "mercury compounds",
    "methylene chloride", "prohibited cattle materials", "sunscreens in cosmetics",
    "vinyl chloride", "zirconium-containing complexes"
]

sunscreen_approved_list = [
    "aminobenzoic acid", "avobenzone", "cinoxate", "dioxybenzone",
    "homosalate", "menthyl anthranilate", "octocrylene",
    "octyl methoxycinnamate", "octyl salicylate", "oxybenzone",
    "padimate o", "phenylbenzimidazole sulfonic acid", "sulisobenzone",
    "titanium dioxide", "trolamine salicylate", "zinc oxide"
]

@router.post("/check")
def check_ingredients(ingredients: str, is_sunscreen: bool = False):
    ingredients_lower = ingredients.strip().lower()
    
    # 1단계: 규제 성분 체크
    for banned in banned_list:
        if banned in ingredients_lower:
            return {
                "pass": False,
                "reason": f"규제 성분 포함: {banned}"
            }
    
    # 2단계: 선크림 승인 성분 체크
    if is_sunscreen:
        approved = any(a in ingredients_lower for a in sunscreen_approved_list)
        if not approved:
            return {
                "pass": False,
                "reason": "선크림 승인 성분 미포함"
            }
    
    return {"pass": True, "reason": "통과"}
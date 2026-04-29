# ========================
# 규제 성분 리스트
# ========================
BANNED_INGREDIENTS = [
    "bithionol", "chlorofluorocarbon propellants", "chloroform",
    "halogenated salicylanilides", "hexachlorophene", "mercury compounds",
    "methylene chloride", "prohibited cattle materials", "sunscreens in cosmetics",
    "vinyl chloride", "zirconium-containing complexes"
]

SUNSCREEN_APPROVED = [
    "aminobenzoic acid", "avobenzone", "cinoxate", "dioxybenzone",
    "homosalate", "menthyl anthranilate", "octocrylene",
    "octyl methoxycinnamate", "octyl salicylate", "oxybenzone",
    "padimate o", "phenylbenzimidazole sulfonic acid", "sulisobenzone",
    "titanium dioxide", "trolamine salicylate", "zinc oxide"
]

# ========================
# 규제 성분 게이트
# ========================
def check_regulation(ingredients: str, is_sunscreen: bool = False) -> dict:
    ingredients_lower = ingredients.strip().lower()

    for banned in BANNED_INGREDIENTS:
        if banned in ingredients_lower:
            return {"pass": False, "reason": f"규제 성분 포함: {banned}"}

    if is_sunscreen:
        approved = any(a in ingredients_lower for a in SUNSCREEN_APPROVED)
        if not approved:
            return {"pass": False, "reason": "선크림 승인 성분 미포함"}

    return {"pass": True, "reason": "통과"}
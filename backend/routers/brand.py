from fastapi import APIRouter, UploadFile, File
from routers.gate import check_ingredients

router = APIRouter(prefix="/brand", tags=["brand"])

@router.post("/analyze")
async def analyze_product(
    product_name: str,
    category: str,
    ingredients: str,
    is_sunscreen: bool = False
):
    # 1. 규제 성분 게이트 체크
    gate_result = check_ingredients(ingredients, is_sunscreen)
    
    if not gate_result["pass"]:
        return {
            "pass": False,
            "reason": gate_result["reason"],
            "if_score": None
        }
    
    # 2. IF Score 산출 (임시 - 나중에 ML 모델 연결)
    # sentiment_score: BERT 모델 연결 후 추가
    # trend_score: 구글 트렌드 데이터 연결 후 추가
    # regulation_score: 게이트 통과 시 1.0

    return {
        "pass": True,
        "product_name": product_name,
        "category": category,
        "if_score": None,  # 나중에 ML 모델 연결
        "message": "ML 모델 연결 후 점수 산출 예정"
    }
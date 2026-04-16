from fastapi import APIRouter

router = APIRouter(prefix="/buyer", tags=["buyer"])

@router.get("/products")
def get_products():
    # 나중에 DB 연결 후 상품 목록 + IF Score 반환
    return {
        "message": " 상품 목록 - DB 연결 후 구현 예정",
        "products": []
    }

@router.get("/category")
def get_category_scores():
    # 나중에 CF Score 반환
    return {
        "message": "카테고리별 CF Score - 감성/트렌드 데이터 연결 후 구현 예정",
        "categories": {
            "skincare": None,
            "makeup": None,
            "suncare": None,
            "maskpack": None
        }
    }
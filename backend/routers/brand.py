from fastapi import APIRouter
from schemas import ProductCreate, FitScoreResponse
from services import save_product_and_predict

router = APIRouter(prefix="/brand", tags=["brand"])

@router.post("/analyze", response_model=FitScoreResponse)
def analyze_product(data: ProductCreate):
    return save_product_and_predict(data)

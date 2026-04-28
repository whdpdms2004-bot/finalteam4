from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from schemas import ProductCreate, FitScoreResponse
from services import save_product_and_predict

router = APIRouter(prefix="/brand", tags=["brand"])

@router.post("/analyze", response_model=FitScoreResponse)
def analyze_product(data: ProductCreate, db: Session = Depends(get_db)):
    return save_product_and_predict(db, data)

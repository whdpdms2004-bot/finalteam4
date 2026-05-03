from pydantic import BaseModel
from typing import List, Optional


class IngredientIn(BaseModel):
    ing_name: str
    ing_kor: Optional[str] = None


class ProductCreate(BaseModel):
    brand_id:           int
    category_detail_id: int
    product_name:       str
    brand_name:         str
    price:              float
    spf_index:          Optional[float] = None
    category:           str             # 한글 대분류 (예: "스킨케어")
    sub_category:       str             # 한글 소분류 (예: "크림")
    ingredients:        List[IngredientIn]


class TopFeature(BaseModel):
    feature:    str
    shap_value: float


class FitScoreResponse(BaseModel):
    product_id:   int
    score:        float
    top_features: List[TopFeature]


class ProductOut(BaseModel):
    product_id:         int
    product_name:       str
    brand_name:         str
    sub_category:       Optional[str]
    score:              Optional[float]
    price:              Optional[float]
    spf_index:          Optional[float]
    created_at:         str
    active_ingredients: List[str]

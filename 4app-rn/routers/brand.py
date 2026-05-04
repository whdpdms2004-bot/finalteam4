from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List
from database import get_db
from schemas import ProductCreate, FitScoreResponse, ProductOut
from services import save_product_and_predict
from models import Product, ProIng

router = APIRouter(prefix="/brand", tags=["brand"])

_DETAIL_TO_MAIN = [
    (range(1, 20),  1),  # 스킨케어
    (range(20, 35), 2),  # 클렌징
    (range(35, 39), 3),  # 선케어  (35~38: 선크림·선스프레이·선로션·선스틱)
    (range(39, 70), 4),  # 마스크팩 (39~44: 시트마스크 등, 50~69: 기타)
]

def _detail_to_main(detail_id: int) -> int:
    for r, mid in _DETAIL_TO_MAIN:
        if detail_id in r:
            return mid
    return 1

@router.post("/analyze", response_model=FitScoreResponse)
def analyze_product(data: ProductCreate, db: Session = Depends(get_db)):
    return save_product_and_predict(db, data)

@router.get("/list/{brand_id}", response_model=List[ProductOut])
def get_brand_products(brand_id: int, db: Session = Depends(get_db)):
    products = (
        db.query(Product)
        .filter(Product.brand_id == brand_id)
        .order_by(Product.created_at.desc())
        .all()
    )
    result = []
    for p in products:
        active_ings = [ing.ing_kor or ing.ing_name for ing in p.ingredients]
        result.append(ProductOut(
            product_id=p.product_id,
            product_name=p.product_name,
            brand_name=p.brand_name,
            sub_category=p.sub_category or "",
            score=p.score,
            price=p.price,
            spf_index=p.spf_index,
            created_at=p.created_at.strftime("%Y.%m.%d") if p.created_at else "",
            active_ingredients=active_ings,
        ))
    return result

def _safe_query(db, sql: str, val: int):
    try:
        result = db.execute(text(sql), {"val": val})
        return result
    except Exception:
        db.rollback()
        return None

@router.get("/market-data")
def get_market_data(category_detail_id: int = 1, db: Session = Depends(get_db)):
    mid = _detail_to_main(category_detail_id)

    # trend_timing: category_sub_id 1~17 — category_detail_id 직접 사용, 없으면 main id fallback
    trend = {}
    for val in (category_detail_id, mid):
        r = _safe_query(db, """
            SELECT review_count, search_volume_label, trend, yoy_pct, market_size_label
            FROM trend_timing
            WHERE category_sub_id = :val
            LIMIT 1
        """, val)
        if r:
            row = r.mappings().fetchone()
            if row:
                trend = dict(row)
                break

    # reference: 유사 상품 평균 평점 · 평균 리뷰수
    ref = {}
    for val in (category_detail_id, mid):
        r = _safe_query(db, """
            SELECT avg_review_score, product_count
            FROM "reference"
            WHERE category_sub_id = :val
            LIMIT 1
        """, val)
        if r:
            row = r.mappings().fetchone()
            if row:
                ref = dict(row)
                break

    # price_cluster: 가격 분포
    price = {}
    for val in (category_detail_id, mid):
        r = _safe_query(db, """
            SELECT min_price, avg_price, max_price, product_count
            FROM price_cluster
            WHERE category_sub_id = :val
            LIMIT 1
        """, val)
        if r:
            row = r.mappings().fetchone()
            if row:
                price = dict(row)
                break

    # ingredient_trend: 트렌드 키워드 (성장률 높은 순)
    kws = []
    for val in (category_detail_id, mid):
        r = _safe_query(db, """
            SELECT us_keyword, kr_keyword, us_stage, us_yoy_pct
            FROM ingredient_trend
            WHERE category_sub_id = :val
            ORDER BY us_yoy_pct DESC NULLS LAST
            LIMIT 15
        """, val)
        if r:
            rows = r.mappings().fetchall()
            kws = [dict(row) for row in rows]
            if kws:
                break

    review_count  = trend.get("review_count")
    product_count = price.get("product_count")
    yoy_pct       = trend.get("yoy_pct")
    avg_rs          = ref.get("avg_review_score")
    ref_product_cnt = ref.get("product_count")

    return {
        "review_count":         int(review_count) if review_count is not None else None,
        "search_volume_label":  trend.get("search_volume_label"),
        "trend_id":             trend.get("trend"),
        "yoy_pct":              float(yoy_pct) if yoy_pct is not None else None,
        "market_size_label":    trend.get("market_size_label"),
        "avg_review_score":     float(avg_rs) if avg_rs is not None else None,
        "ref_review_count":     int(ref_product_cnt) if ref_product_cnt is not None else None,
        "min_price":            float(price.get("min_price") or 0),
        "avg_price":            float(price.get("avg_price") or 0),
        "max_price":            float(price.get("max_price") or 0),
        "product_count":        int(product_count) if product_count is not None else None,
        "trend_keywords": [
            {
                "us_kw": r.get("us_keyword") or "",
                "kr_kw": r.get("kr_keyword") or "",
                "stage": r.get("us_stage") or "",
                "yoy":   round(float(r.get("us_yoy_pct") or 0), 1),
            }
            for r in kws
        ],
    }


@router.delete("/products/{product_id}")
def delete_product(product_id: int, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.product_id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    db.query(ProIng).filter(ProIng.product_id == product_id).delete()
    db.delete(product)
    db.commit()
    return {"message": "deleted", "product_id": product_id}

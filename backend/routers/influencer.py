import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import get_db

router = APIRouter(tags=["influencer"])

try:
    from openai import OpenAI as _OpenAI
    _ai = _OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    _ai = None

# 제품명 + 성분 목록
_PRODUCT_SQL = text("""
    SELECT
        p.product_id,
        p.product_name,
        p.brand_name,
        ARRAY_AGG(pi.ing_kor ORDER BY pi.seq_no)
            FILTER (WHERE pi.ing_kor IS NOT NULL) AS ing_list
    FROM products p
    LEFT JOIN pro_ing pi ON pi.product_id = p.product_id
    WHERE p.product_id = :product_id
    GROUP BY p.product_id, p.product_name, p.brand_name
""")

# 인플루언서 + 시장 수요 가장 높은 페르소나 1개
_INFLUENCER_CARD_SQL = text("""
    SELECT
        i.content_type,
        i.audience_spec,
        i.category_main_id,
        i.cosine_score,
        ip.persona_type,
        ip.h12_topic,
        ip.market_demand_pct
    FROM influencer i
    JOIN influencer_persona ip ON ip.category_main_id = i.category_main_id
    WHERE i.influencer_id = :influencer_id
    ORDER BY ip.market_demand_pct DESC
    LIMIT 1
""")

# 카테고리별 소비자 불만 상위 2개
_UNMET_SQL = text("""
    SELECT topic_label_kr
    FROM unmet_needs
    WHERE category_main_id = :cat_id
    ORDER BY topic_pct DESC
    LIMIT 2
""")

_PROMPT_TEMPLATE = """당신은 K-뷰티 인플루언서 마케팅 전문가입니다.
아래 매칭 데이터를 바탕으로 인플루언서 협업 추천 메시지를 한국어 3~4문장으로 작성하세요.
수치는 반드시 포함하고, 제품 성분이 소비자 불만을 해결하는 맥락을 중심으로 서술하세요.
중요한 수치와 핵심 키워드는 [[이렇게]] 이중 대괄호로 강조하세요.

[매칭 데이터]
- 제품명          : {product_name}
- 주요 성분       : {ing_list}
- 소비자 불만 TOP2 : {unmet_top2}
- 페르소나 유형   : {persona_type}
- 시장 수요       : {market_demand_pct}%
- 의미적 유사도   : {cosine_score}
- 콘텐츠 방향     : {content_type}
- 주 타겟         : {audience_spec}

추천 메시지:"""


@router.get("/api/products/{product_id}/influencers/{influencer_id}/message")
def get_influencer_ai_message(product_id: int, influencer_id: int, db: Session = Depends(get_db)):
    prod = db.execute(_PRODUCT_SQL, {"product_id": product_id}).mappings().fetchone()
    if not prod:
        raise HTTPException(status_code=404, detail="제품 없음")

    card = db.execute(_INFLUENCER_CARD_SQL, {"influencer_id": influencer_id}).mappings().fetchone()
    if not card:
        raise HTTPException(status_code=404, detail="인플루언서 없음")

    unmet_rows = db.execute(_UNMET_SQL, {"cat_id": card["category_main_id"]}).mappings().fetchall()
    unmet_top2 = ", ".join(r["topic_label_kr"] for r in unmet_rows)

    ing_list = ", ".join(i for i in (prod["ing_list"] or []) if i)

    if not _ai:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    prompt = _PROMPT_TEMPLATE.format(
        product_name=prod["product_name"],
        ing_list=ing_list or "성분 정보 없음",
        unmet_top2=unmet_top2 or "불만 정보 없음",
        persona_type=card["persona_type"] or "",
        market_demand_pct=float(card["market_demand_pct"] or 0),
        cosine_score=float(card["cosine_score"] or 0),
        content_type=card["content_type"] or "",
        audience_spec=card["audience_spec"] or "",
    )

    try:
        resp = _ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "K-뷰티 인플루언서 마케팅 전문가. 한국어로 답하세요."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        ai_message = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"OpenAI error: {e}")

    return {
        "product_id":    product_id,
        "influencer_id": influencer_id,
        "ai_message":    ai_message,
    }

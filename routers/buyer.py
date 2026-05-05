import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import text, or_, and_
from pydantic import BaseModel
from database import get_db
from models import Product, ProIng

router = APIRouter(prefix="/buyer", tags=["buyer"])

# ── OpenAI 클라이언트 (OPENAI_API_KEY 환경변수 필요) ──────────
try:
    from openai import OpenAI as _OpenAI
    _ai = _OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    _ai = None


class AiMessageRequest(BaseModel):
    influencer_name: str
    persona: str
    cosine_score: float
    h01_risk: str
    market_share: float
    content_direction: str
    target_audience: str
    platform: str
    followers: str
    product_name: str
    product_brand: str

# sub_category(소분류, 한글) → category_main_id (1=스킨케어, 2=클렌징, 3=선케어, 4=마스크팩)
SUBCAT_TO_MAIN = {
    # 스킨케어 (1)
    "토너": 1, "에센스": 1, "세럼": 1, "크림": 1, "로션": 1, "미스트": 1,
    "패드": 1, "앰플": 1, "필링젤": 1, "수분크림": 1, "스킨": 1, "아이크림": 1,
    "젤크림": 1, "에멀젼": 1, "부스터": 1, "시카크림": 1, "barrier cream": 1,
    # 클렌징 (2)
    "클렌징폼": 2, "클렌징오일": 2, "클렌징워터": 2, "클렌징밤": 2,
    "클렌징밀크": 2, "클렌징젤": 2, "버블클렌저": 2, "클렌징": 2,
    # 선케어 (3)
    "선크림": 3, "선스틱": 3, "선쿠션": 3, "선스프레이": 3,
    "선젤": 3, "선밀크": 3, "선로션": 3, "선케어": 3,
    # 마스크팩 (4)
    "시트마스크": 4, "마스크팩": 4, "워시오프마스크": 4,
    "슬리핑마스크": 4, "패치": 4,
}

# category_detail_id 범위 → category_main_id (products 등록 시 사용하는 ID 범위)
DETAIL_ID_TO_MAIN = {
    range(1, 20):   1,  # 스킨케어
    range(20, 35):  2,  # 클렌징
    range(35, 39):  3,  # 선케어  (35~38: 선크림·선스프레이·선로션·선스틱)
    range(39, 70):  4,  # 마스크팩 (39~44: 시트마스크 등, 50~69: 기타)
}

def _resolve_category_main_id(product: Product) -> int:
    """sub_category → category_main_id, 실패 시 category_detail_id 범위로 추정"""
    sub = (product.sub_category or "").strip()
    if sub in SUBCAT_TO_MAIN:
        return SUBCAT_TO_MAIN[sub]
    for r, main_id in DETAIL_ID_TO_MAIN.items():
        if product.category_detail_id in r:
            return main_id
    return 1  # default 스킨케어

def _format_influencer(r: dict) -> dict:
    tags_raw = r.get("tags") or ""
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    cosine = float(r.get("cosine_score") or 0)
    return {
        "influencer_id":   r.get("influencer_id"),
        "name":            (r.get("influencer_handle") or "").lstrip("@"),
        "persona":         r.get("persona_type") or "",
        "score":           round(cosine * 100, 1),
        "platform":        r.get("platform") or "Instagram",
        "followers":       r.get("follower_range") or "",
        "location":        r.get("location") or "USA",
        "target_audience": r.get("audience_spec") or "",
        "image":           r.get("profile_image_url") or "",
        "tags":            tags,
        "market_share":    float(r.get("market_demand_pct") or 0),
        "cosine_score":    cosine,
        "h01_risk":        r.get("h12_topic") or "",
        "content_direction": r.get("content_type") or "",
        "category_main_id": r.get("category_main_id"),
    }

# 인플루언서별 시장수요 가장 높은 페르소나 1개씩 매칭
_INFLUENCER_SQL_BY_CATEGORY = text("""
    SELECT *
    FROM (
        SELECT DISTINCT ON (i.influencer_id)
            i.influencer_id,
            i.influencer_handle,
            i.profile_image_url,
            i.follower_range,
            i.location,
            i.audience_spec,
            i.platform,
            i.content_type,
            i.cosine_score,
            i.category_main_id,
            ip.persona_type,
            ip.h12_topic,
            ip.market_demand_pct,
            ip.tags
        FROM influencer i
        JOIN influencer_persona ip ON ip.category_main_id = i.category_main_id
        WHERE i.category_main_id = :cat_id
          AND i.cosine_score >= 0.6
        ORDER BY i.influencer_id, ip.market_demand_pct DESC
    ) sub
    ORDER BY cosine_score DESC
    LIMIT 10
""")

@router.get("/influencers/{product_id}")
def get_influencers_for_product(product_id: int, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.product_id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    cat_id = _resolve_category_main_id(product)

    try:
        rows = db.execute(_INFLUENCER_SQL_BY_CATEGORY, {"cat_id": cat_id}).mappings().fetchall()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Influencer DB unavailable: {e}")

    if not rows:
        raise HTTPException(status_code=404, detail="No influencers found for this product category")

    return [_format_influencer(dict(r)) for r in rows]


@router.get("/products")
def get_products(
    category: Optional[str] = Query(None),
    search:   Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    MAIN_NAME_MAP = {"스킨케어": 1, "클렌징": 2, "선케어": 3, "마스크팩": 4}

    query = (
        db.query(Product)
        .options(selectinload(Product.ingredients))
        .filter(Product.brand_id == 1)          # 비플레인 상품만
        .filter(Product.status != '심사중')      # 심사 중은 바이어에게 비노출
    )

    # ── 카테고리 필터 ──────────────────────────────────────────
    if category and category not in ("전체", ""):
        cat_id = MAIN_NAME_MAP.get(category)
        if cat_id:
            matching_subs = [sub for sub, mid in SUBCAT_TO_MAIN.items() if mid == cat_id]
            detail_filters = [
                and_(Product.category_detail_id >= r.start, Product.category_detail_id < r.stop)
                for r, mid in DETAIL_ID_TO_MAIN.items() if mid == cat_id
            ]
            f = []
            if matching_subs:
                f.append(Product.sub_category.in_(matching_subs))
            f.extend(detail_filters)
            if f:
                query = query.filter(or_(*f))

    # ── 검색 필터 (브랜드·상품명·성분) ────────────────────────
    if search and search.strip():
        term = f"%{search.strip()}%"
        ing_ids = db.query(ProIng.product_id).filter(
            or_(ProIng.ing_name.ilike(term), ProIng.ing_kor.ilike(term))
        )
        query = query.filter(or_(
            Product.brand_name.ilike(term),
            Product.product_name.ilike(term),
            Product.product_id.in_(ing_ids),
        ))

    products = query.order_by(text("score DESC NULLS LAST")).all()

    result = []
    for p in products:
        ings = sorted(p.ingredients, key=lambda i: i.seq_no)[:5]
        result.append({
            "product_id":   p.product_id,
            "brand":        p.brand_name,
            "name":         p.product_name,
            "price":        p.price,
            "score":        round(float(p.score)) if p.score else 0,
            "sub_category": p.sub_category or "",
            "spf":          p.spf_index,
            "ingredients":  [{"name": i.ing_name, "kor": i.ing_kor or i.ing_name} for i in ings],
        })
    return result


@router.get("/category")
def get_category_scores(db: Session = Depends(get_db)):
    return {
        "message": "카테고리별 CF Score - 구현 예정",
        "categories": {"skincare": None, "makeup": None, "suncare": None, "maskpack": None},
    }


@router.get("/influencers-by-category")
def get_influencers_by_category(category: str, db: Session = Depends(get_db)):
    MAIN_NAME_MAP = {"스킨케어": 1, "클렌징": 2, "선케어": 3, "마스크팩": 4}
    cat_id = SUBCAT_TO_MAIN.get(category.strip()) or MAIN_NAME_MAP.get(category.strip(), 1)

    try:
        rows = db.execute(_INFLUENCER_SQL_BY_CATEGORY, {"cat_id": cat_id}).mappings().fetchall()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Influencer DB unavailable: {e}")

    if not rows:
        raise HTTPException(status_code=404, detail="No influencers found for this category")

    return [_format_influencer(dict(r)) for r in rows]


class MarketingSourceRequest(BaseModel):
    influencer_name:   str
    persona:           str
    cosine_score:      float
    market_share:      float
    platform:          str
    followers:         str
    content_direction: str
    h01_risk:          str
    target_audience:   str
    product_name:      str
    product_brand:     str
    category_main_id:  int = 1

@router.post("/marketing-source")
def get_marketing_source(req: MarketingSourceRequest, db: Session = Depends(get_db)):
    # 1. 채널 분석 — 인플루언서 플랫폼 기반
    PLATFORM_DIST = {
        "Instagram": [("Instagram", 62, "#1C1C1E"), ("TikTok", 24, "#8E8E93"), ("YouTube", 14, "#D1D1D6")],
        "TikTok":    [("TikTok", 60, "#1C1C1E"), ("Instagram", 27, "#8E8E93"), ("YouTube", 13, "#D1D1D6")],
        "YouTube":   [("YouTube", 58, "#1C1C1E"), ("Instagram", 26, "#8E8E93"), ("TikTok", 16, "#D1D1D6")],
    }
    channels = [
        {"name": n, "pct": p, "color": c}
        for n, p, c in PLATFORM_DIST.get(req.platform, PLATFORM_DIST["Instagram"])
    ]

    # 2. 콘텐츠 전략 키워드 — unmet_needs 테이블
    try:
        kw_rows = db.execute(text("""
            SELECT topic_label_en, topic_pct, topic_rank
            FROM unmet_needs
            WHERE category_main_id = :cat_id
            ORDER BY topic_rank
            LIMIT 5
        """), {"cat_id": req.category_main_id}).mappings().fetchall()
        keywords = [
            {
                "tag":    f"#{r['topic_label_en'].replace(' ', '').lower()}",
                "volume": f"{float(r['topic_pct']):.1f}%",
                "trend":  f"+{round(float(r['topic_pct']) * 0.38)}%",
            }
            for r in kw_rows
        ]
    except Exception:
        keywords = []

    # 3. 포지셔닝 점수 — cosine_score·market_share 기반 계산
    cs, ms = req.cosine_score, req.market_share
    scores = [
        {"label": "트렌드 일치", "score": min(round(cs * 102), 100), "color": "#FF6B6B"},
        {"label": "타깃 유사도", "score": min(round(cs * 95 + ms * 0.25), 100), "color": "#A855F7"},
        {"label": "시장 도달",   "score": min(round(cs * 84 + 12), 100),  "color": "#3B82F6"},
        {"label": "콘텐츠 적합", "score": min(round(cs * 91 + 4), 100),   "color": "#10B981"},
    ]

    # 4. 예상 도달 수치 — follower_range(K 단위) 기반
    try:
        f = float(req.followers) * 1000
    except Exception:
        f = 50_000
    reach_7d = f * 0.045 * 7
    unique   = reach_7d * 0.55
    def _fmt(n: float) -> str:
        return f"{n / 1_000_000:.1f}M" if n >= 1_000_000 else f"{n / 1_000:.0f}K"
    reach = [
        {"label": "예상 노출 수", "val": _fmt(reach_7d), "sub": "7일 기준"},
        {"label": "예상 리치",   "val": _fmt(unique),   "sub": "유니크 유저"},
    ]

    # 5. AI 캠페인 전략 (OpenAI)
    strategy = None
    if _ai:
        try:
            prompt = f"""K-뷰티 마케팅 전략가로서 아래 데이터를 바탕으로 캠페인 전략을 2단락으로 작성하세요.
인플루언서: @{req.influencer_name} ({req.platform}, {req.followers}K 팔로워)
페르소나: {req.persona} | 코사인 유사도: {cs:.3f} | 시장 수요: {ms}%
콘텐츠 방향: {req.content_direction}
핵심 해결 리스크: {req.h01_risk}
상품: {req.product_brand} '{req.product_name}'
중요 수치와 핵심 키워드는 [[이렇게]] 이중 대괄호로 강조하세요."""
            resp = _ai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "K-뷰티 마케팅 전략 전문가. 간결한 한국어로 답하세요."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.7,
            )
            strategy = resp.choices[0].message.content.strip()
        except Exception:
            pass

    return {"channels": channels, "keywords": keywords, "scores": scores, "reach": reach, "strategy": strategy}


@router.post("/ai-message")
def get_ai_message(req: AiMessageRequest):
    if not _ai:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")

    prompt = f"""당신은 K-뷰티 마케팅 전략 전문가입니다. 아래 인플루언서 매칭 분석 결과를 바탕으로 브랜드 담당자에게 제공할 마케팅 추천 메시지를 한국어로 작성하세요.

[매칭 데이터]
- 인플루언서: @{req.influencer_name} ({req.platform}, 팔로워 {req.followers})
- 의미적 유사도(코사인): {req.cosine_score:.3f} (임계치 0.6 통과)
- 페르소나: {req.persona} (카테고리 시장 수요 {req.market_share}%)
- 핵심 리스크 해결: {req.h01_risk}
- 콘텐츠 방향: {req.content_direction}
- 주 타겟: {req.target_audience}
- 매칭 상품: {req.product_brand} '{req.product_name}'

다음 흐름으로 3~4 단락(각 1~2 문장)을 작성하세요:
1. 코사인 유사도를 언급한 데이터 기반 매칭 근거
2. 핵심 리스크 해결과 페르소나 연결
3. 타겟 오디언스 구매 전환 가능성
4. 구체적인 콘텐츠 방향 제안

중요한 수치, 퍼센트, 핵심 키워드나 문구는 반드시 [[이렇게]] 이중 대괄호로 감싸주세요."""

    try:
        resp = _ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 K-뷰티 마케팅 전략 전문가입니다. 간결하고 전문적인 한국어로 답하세요."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return {"message": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"OpenAI error: {e}")


# ─────────────────────────────────────────────────────────────
# AI 소싱 모드 채팅 엔드포인트
# ─────────────────────────────────────────────────────────────

_AI_PRODUCTS = [
    {"id": 0,  "name": "선뮤즈 모이스처 선크림",         "category": "선케어",   "price": 37, "spf": "50+", "fitScore": 77, "moq": 100, "ingredients": ["나이아신아마이드", "데실글루코사이드", "아데노신"]},
    {"id": 1,  "name": "그린티 선크림 80ml",             "category": "선케어",   "price": 12, "spf": "50",  "fitScore": 72, "moq": 100, "ingredients": ["녹차", "세라마이드"]},
    {"id": 2,  "name": "무기자차 선크림 50ml",            "category": "선케어",   "price": 13, "spf": "50",  "fitScore": 69, "moq": 100, "ingredients": ["징크옥사이드"]},
    {"id": 3,  "name": "논코메도제닉 선크림 60ml",        "category": "선케어",   "price": 11, "spf": "30",  "fitScore": 66, "moq": 100, "ingredients": ["나이아신아마이드"]},
    {"id": 4,  "name": "비건 선크림 SPF50 70ml",         "category": "선케어",   "price": 14, "spf": "50",  "fitScore": 63, "moq": 100, "ingredients": ["비건성분", "알로에"]},
    {"id": 5,  "name": "무향 선크림 60ml",               "category": "선케어",   "price": 10, "spf": "30",  "fitScore": 60, "moq": 100, "ingredients": ["무향", "판테놀"]},
    {"id": 6,  "name": "어드밴스드 스네일 에센스 100ml",  "category": "스킨케어", "price": 14, "spf": None,  "fitScore": 70, "moq": 100, "ingredients": ["나이아신아마이드", "스네일"]},
    {"id": 7,  "name": "세라마이드 토너 150ml",           "category": "스킨케어", "price": 13, "spf": None,  "fitScore": 67, "moq": 100, "ingredients": ["세라마이드", "히알루론산"]},
    {"id": 8,  "name": "비타민C 브라이트닝 세럼 30ml",   "category": "스킨케어", "price": 15, "spf": None,  "fitScore": 72, "moq": 100, "ingredients": ["비타민C", "나이아신아마이드"]},
    {"id": 9,  "name": "센텔라 진정 앰플 50ml",           "category": "스킨케어", "price": 12, "spf": None,  "fitScore": 65, "moq": 100, "ingredients": ["센텔라", "판테놀"]},
    {"id": 10, "name": "히알루론산 수분크림 60ml",        "category": "스킨케어", "price": 11, "spf": None,  "fitScore": 63, "moq": 100, "ingredients": ["히알루론산", "알로에"]},
    {"id": 11, "name": "비건 클렌징 폼 150ml",           "category": "클렌징",   "price": 10, "spf": None,  "fitScore": 64, "moq": 100, "ingredients": ["알로에", "캐모마일"]},
    {"id": 12, "name": "저자극 클렌징 오일 200ml",       "category": "클렌징",   "price": 13, "spf": None,  "fitScore": 67, "moq": 100, "ingredients": ["호호바오일", "올리브"]},
    {"id": 13, "name": "민감성 클렌징 워터 300ml",       "category": "클렌징",   "price": 9,  "spf": None,  "fitScore": 61, "moq": 100, "ingredients": ["미셀라워터", "알로에"]},
    {"id": 14, "name": "딥클렌징 버블폼 120ml",          "category": "클렌징",   "price": 11, "spf": None,  "fitScore": 64, "moq": 100, "ingredients": ["살리실산", "티트리"]},
    {"id": 15, "name": "약산성 클렌징 젤 150ml",         "category": "클렌징",   "price": 10, "spf": None,  "fitScore": 62, "moq": 100, "ingredients": ["약산성", "판테놀"]},
    {"id": 16, "name": "시카 마스크팩 25ml",             "category": "마스크",   "price": 8,  "spf": None,  "fitScore": 66, "moq": 100, "ingredients": ["센텔라", "판테놀"]},
    {"id": 17, "name": "히알루론산 수분 마스크 25ml",    "category": "마스크",   "price": 7,  "spf": None,  "fitScore": 63, "moq": 100, "ingredients": ["히알루론산", "알로에"]},
    {"id": 18, "name": "비타민C 브라이트닝 마스크 25ml", "category": "마스크",   "price": 9,  "spf": None,  "fitScore": 65, "moq": 100, "ingredients": ["비타민C", "나이아신아마이드"]},
    {"id": 19, "name": "콜라겐 탄력 마스크 25ml",        "category": "마스크",   "price": 8,  "spf": None,  "fitScore": 62, "moq": 100, "ingredients": ["콜라겐", "펩타이드"]},
    {"id": 20, "name": "진정 수딩 마스크 25ml",          "category": "마스크",   "price": 7,  "spf": None,  "fitScore": 60, "moq": 100, "ingredients": ["알로에", "캐모마일"]},
]

def _chat_detect_cat(q: str):
    if any(k in q for k in ["선케어", "선크림", "spf", "끈적", "백탁", "자차", "여드름", "논코메도"]):
        return "선케어", 3
    if any(k in q for k in ["스킨케어", "에센스", "세럼", "토너"]):
        return "스킨케어", 1
    if "클렌징" in q:
        return "클렌징", 2
    if "마스크" in q:
        return "마스크", 4
    return None, None

def _chat_filter(q: str, show_all: bool) -> list:
    cat_name, _ = _chat_detect_cat(q)
    items = [p for p in _AI_PRODUCTS if not cat_name or p["category"] == cat_name]
    items = sorted(items, key=lambda p: p["fitScore"], reverse=True)
    return items if show_all else items[:3]

def _chat_summary(query: str, n: int, category: Optional[str], complaints: list) -> str:
    if not _ai:
        prefix = f"{category} 카테고리에서 " if category else ""
        return f"분석 완료! {prefix}조건에 맞는 제품 {n}개를 찾았어요."
    try:
        c_str = ", ".join(
            f"{r['topic_label_en']}({float(r['topic_pct']):.1f}%)"
            for r in complaints[:3]
        )
        prompt = (
            f'바이어 요청: "{query}"\n'
            f'매칭 제품: {n}개 | 카테고리: {category or "전체"}\n'
            + (f'주요 불만 데이터: {c_str}\n' if c_str else "") +
            '\n2~3문장으로 K뷰티 강점을 불만 데이터와 연결한 소싱 추천 요약을 한국어로 작성하세요.'
        )
        resp = _ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "BeautyBridge AI 소싱 어시스턴트. 간결한 한국어."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=150,
            temperature=0.65,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        prefix = f"{category} 카테고리에서 " if category else ""
        return f"분석 완료! {prefix}조건에 맞는 제품 {n}개를 찾았어요."


class AiChatRequest(BaseModel):
    query:    str
    show_all: bool = False


@router.post("/ai-chat")
def ai_chat(req: AiChatRequest, db: Session = Depends(get_db)):
    q = req.query.lower()

    # 매칭 완료
    if "매칭 요청 완료" in q:
        return {"summary": "매칭 요청이 완료됐어요! 브랜드사가 확인 후 연락드릴 거예요."}

    # 상세보기
    if "상세보기" in q:
        name = req.query.replace(" 상세보기", "").strip()
        product = next((p for p in _AI_PRODUCTS if p["name"] == name), None)
        return {
            "summary": f"{name} 상세 정보예요!",
            "detail":  product,
            "actions": [{"label": "협상가이드"}],
        }

    cat_name, cat_id = _chat_detect_cat(q)
    products = _chat_filter(q, req.show_all)

    # unmet_needs → 트렌드 & 키워드
    try:
        rows = db.execute(text("""
            SELECT topic_label_en, topic_pct
            FROM unmet_needs
            WHERE (:cat_id IS NULL OR category_main_id = :cat_id)
            ORDER BY topic_rank
            LIMIT 6
        """), {"cat_id": cat_id}).mappings().fetchall()
        sorted_rows = sorted(rows, key=lambda r: float(r["topic_pct"]), reverse=True)
        keywords = [r["topic_label_en"].replace(" ", "").lower() for r in sorted_rows[:4]]
        max_pct  = float(sorted_rows[0]["topic_pct"]) if sorted_rows else 50.0
        trends   = [
            {
                "name":   r["topic_label_en"],
                "growth": f"+{round(float(r['topic_pct']) * 0.38)}%",
                "bar":    round(min(float(r["topic_pct"]) / max_pct, 1.0), 2),
                "hot":    i < 2,
            }
            for i, r in enumerate(sorted_rows[:3])
        ]
        complaints = list(sorted_rows)
    except Exception:
        keywords   = ["무기자차", "비건인증", "SPF50+", "논코메도제닉"]
        trends     = [
            {"name": "비건 선케어", "growth": "+39%", "bar": 0.9,  "hot": True},
            {"name": "무기자차",    "growth": "+28%", "bar": 0.75, "hot": True},
            {"name": "SPF50+",     "growth": "+15%", "bar": 0.58, "hot": False},
        ]
        complaints = []

    suppliers = [
        {"name": p["name"], "moq": f"{p['moq']}개", "lead": "30일",
         "score": p["fitScore"], "cert": p["ingredients"][0]}
        for p in products
    ]

    summary = _chat_summary(q, len(products), cat_name, complaints)

    return {
        "summary":   summary,
        "trends":    trends,
        "keywords":  keywords,
        "suppliers": suppliers,
        "actions":   [{"label": "제품 전체 보기"}, {"label": "매칭 요청하기"}],
    }

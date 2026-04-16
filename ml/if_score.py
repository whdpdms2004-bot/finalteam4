import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from regulation import check_regulation

# ========================
# 유사 상품 매칭 & 감성 점수 추출
# ========================

def get_similar_sentiment_score(
    product_name: str,
    category: str,
    df_amazon_meta: pd.DataFrame,
    df_amazon_sentiment: pd.DataFrame,
    top_n: int = 10
) -> float:
    """
    product_name        : 입력 상품명
    category            : 입력 상품 카테고리
    df_amazon_meta      : amazon 메타데이터 (product_id, product_name, main_category)
    df_amazon_sentiment : amazon 감성 점수 (product_id, sentiment_score)
    top_n               : 유사 상품 상위 N개
    반환값              : 유사 상품 평균 감성 점수 (0~1)
    """
    # 1. 카테고리 필터링
    filtered = df_amazon_meta[df_amazon_meta['main_category'] == category].copy()

    if len(filtered) == 0:
        return 0.5  # 데이터 없으면 중립값

    # 2. TF-IDF + 코사인 유사도
    vectorizer = TfidfVectorizer() # 텍스트를 숫자로 바꿔주는 도구 
    amazon_texts = filtered['product_name'].fillna('').tolist() # 아마존 상품들을 리스트로 가져오기 
    all_texts = amazon_texts + [product_name] # 아마존 상품명 리스트 뒤에 우리 입력 상품명 추가 

    tfidf_matrix = vectorizer.fit_transform(all_texts) #모든 텍스트를 숫자 벡터로 변환 
    input_vec = tfidf_matrix[-1] # 우리 입력 상품 벡터
    amazon_vecs = tfidf_matrix[:-1] # 아마존 상품들 벡터 

    similarities = cosine_similarity(input_vec, amazon_vecs).flatten() # 우리 입력 상품이랑 아마존 상품 하나하나의 유사도를 숫자로 계산

    # 3. 상위 N개 추출
    top_indices = similarities.argsort()[-top_n:][::-1] #유사도가 높은 순서대로 상위 10개 
    top_product_ids = filtered.iloc[top_indices]['product_id'].tolist() # 그 인덱스에 해당하는 아마존 상품 id 가져오기 

    # 4. 유사 상품 감성 점수 평균
    top_sentiments = df_amazon_sentiment[
        df_amazon_sentiment['product_id'].isin(top_product_ids)
    ]['sentiment_score'] # 상위 10개 상품의 감성 점수만 가져오기

    if len(top_sentiments) == 0:
        return 0.5 # 감성 점수가 없으면 중립값 0.5 를 반환 

    return round(top_sentiments.mean(), 4) # 감성 점수 평균을 소수점 4자리로 변환 


# ========================
# IF Score 산출
# ========================

def calculate_if_score(
    product_name: str,
    ingredients: str,
    category: str,
    cf_score: float,
    ingredient_trend_score: float,
    df_amazon_meta: pd.DataFrame,
    df_amazon_sentiment: pd.DataFrame,
    is_sunscreen: bool = False,
    top_n: int = 10,
    w1: float = 0.4,   # 유사상품 감성 점수
    w2: float = 0.3,   # CF Score
    w3: float = 0.3    # 성분 트렌드 점수
) -> dict:
    """
    product_name            : 상품명
    ingredients             : 성분
    category                : 카테고리
    cf_score                : 해당 카테고리 CF Score (0~100)
    ingredient_trend_score  : 성분 트렌드 점수 (0~1)
    df_amazon_meta          : amazon 메타데이터
    df_amazon_sentiment     : amazon 감성 점수
    is_sunscreen            : 선크림 여부
    """
    # 1. 규제 성분 게이트
    gate = check_regulation(ingredients, is_sunscreen)
    regulation_multiplier = 1 if gate["pass"] else 0

    # 2. 유사 상품 감성 점수
    sentiment_score = get_similar_sentiment_score(
        product_name, category, df_amazon_meta, df_amazon_sentiment, top_n
    )

    # 3. CF Score 정규화 (0~100 → 0~1)
    cf_score_normalized = cf_score / 100

    # 4. IF Score 산출
    score = (sentiment_score * w1) + \
            (cf_score_normalized * w2) + \
            (ingredient_trend_score * w3)

    final_score = round(score * 100 * regulation_multiplier, 2)

    return {
        "if_score": final_score,
        "gate_pass": gate["pass"],
        "gate_reason": gate["reason"],
        "sentiment_score": sentiment_score,
        "cf_score": cf_score,
        "ingredient_trend_score": ingredient_trend_score,
    }


# ========================
# 테스트
# ========================
if __name__ == "__main__":
    # 임시 데이터로 테스트
    df_meta = pd.DataFrame({
        'product_id': ['A001', 'A002', 'A003'],
        'product_name': ['niacinamide sunscreen spf50', 'zinc oxide sunscreen lightweight', 'sunscreen spf30 sensitive skin'],
        'main_category': ['suncare', 'suncare', 'suncare']
    })

    df_sentiment = pd.DataFrame({
        'product_id': ['A001', 'A002', 'A003'],
        'sentiment_score': [0.85, 0.78, 0.82]
    })

    result = calculate_if_score(
        product_name="아누아 선크림 zinc oxide niacinamide",
        ingredients="water, zinc oxide, niacinamide, glycerin",
        category="suncare",
        cf_score=78.0,
        ingredient_trend_score=0.72,
        df_amazon_meta=df_meta,
        df_amazon_sentiment=df_sentiment,
        is_sunscreen=True
    )
    print(f"IF Score 결과: {result}")
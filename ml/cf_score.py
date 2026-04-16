import pandas as pd
import numpy as np

# ========================
# CF Score 산출
# ========================

def calculate_cf_score(
    sentiment_score: float,
    trend_score: float,
    w1: float = 0.5,
    w2: float = 0.5
) -> float:
    """
    sentiment_score : 카테고리별 평균 감성 점수 (0~1)
    trend_score     : 카테고리별 트렌드 점수 (0~1)
    반환값          : CF Score (0~100점)
    """
    score = (sentiment_score * w1) + (trend_score * w2)
    return round(score * 100, 2)


def calculate_all_category_cf_scores(
    category_sentiment: dict,
    category_trend: dict,
    w1: float = 0.5,
    w2: float = 0.5
) -> dict:
    """
    카테고리별 CF Score 한번에 산출
    category_sentiment : {'skincare': 0.82, 'makeup': 0.75, ...}
    category_trend     : {'skincare': 0.70, 'makeup': 0.65, ...}
    반환값             : {'skincare': 76.0, 'makeup': 70.0, ...}
    """
    cf_scores = {}
    for category in category_sentiment:
        sentiment = category_sentiment.get(category, 0)
        trend = category_trend.get(category, 0)
        cf_scores[category] = calculate_cf_score(sentiment, trend, w1, w2)
    return cf_scores


# ========================
# 테스트
# ========================
if __name__ == "__main__":
    # 단일 카테고리 테스트
    cf = calculate_cf_score(sentiment_score=0.82, trend_score=0.75)
    print(f"CF Score: {cf}점")

    # 전체 카테고리 테스트 (실제 데이터로 대체 예정)
    category_sentiment = {
        'skincare': 0.82,
        'makeup': 0.75,
        'suncare': 0.78,
        'maskpack': 0.70
    }
    category_trend = {
        'skincare': 0.70,
        'makeup': 0.65,
        'suncare': 0.80,
        'maskpack': 0.60
    }

    all_cf = calculate_all_category_cf_scores(category_sentiment, category_trend)
    print(f"\n카테고리별 CF Score:")
    for cat, score in all_cf.items():
        print(f"  {cat}: {score}점")
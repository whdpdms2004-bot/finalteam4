# ml_v13 모델 설계 문서

## 1. 프로젝트 개요

**목적**: 한국 화장품이 미국 뷰티 시장에 적합한가를 예측하는 이진 분류 모델  
**데이터**: `ml_AmazonSephoraUlta.csv` (Amazon + Sephora + ULTA, 총 5,964개 상품)  
**최종 모델**: LightGBM (ml_v13), Test AUC **0.7900**, 과적합 갭 **-0.0005**

---

## 2. 모델 발전 과정

| 버전 | Y값 | Test AUC | 주요 변경 |
|------|-----|----------|---------|
| ml_v6 | PCA(sentiment + log리뷰수) | 0.7673 | 병합 데이터 최초 적용 |
| ml_v10 | 평점 단독 | 0.6929 | PCA 대신 평점 단독 시도 |
| ml_v11 | log(리뷰수) 단독 | 0.7691 | PCA 제거 후 리뷰수만 사용 |
| ml_v12 | 평점 × log(리뷰수) 복합 | 0.7866 | 복합 지수 Y값 도입 |
| **ml_v13** | 평점 × log(리뷰수) 복합 | **0.7900** | position 피처 확대 + k_beauty 제거 |

---

## 3. Y값 재설계

### 3.1 기존 Y값(PCA)의 문제점

ml_v6까지 사용한 Y값은 `PCA(sentiment_score, log_review_count)`였으나 다음 통계 검정 결과 부적합 판정:

| 검정 | 결과 | 판정 기준 |
|------|------|---------|
| Pearson 상관계수 | **음의 상관** | 두 변수가 반대 방향 → PCA 부적합 |
| Bartlett 구형성 검정 | p = 0.5546 (비유의) | p > 0.05 → 변수들이 독립적 |
| KMO | **0.0001** | 0.6 미만 → PCA 완전 부적합 |

세 조건 모두 PCA 적용 금지 조건에 해당하여 ml_v6의 Y값 설계는 통계적으로 무효.

### 3.2 대안 비교 실험

| Y값 | AUC | 이유 |
|-----|-----|------|
| PCA(sentiment + 리뷰수) | 0.7673 | ❌ KMO=0.0001, 통계적 무효 |
| 평점 단독 | 0.6929 | ❌ 4.0~4.5 구간 편중, 변별력 부족 |
| log(리뷰수) 단독 | 0.7691 | △ 객관적이나 질적 신호 없음 |
| **평점 × log(리뷰수)** | **0.7866** | ✅ 질+양 동시 반영, 통계적 유효 |

### 3.3 최종 Y값 설계

```python
# 정규화(평점) × 정규화(log_review_count) → quantile 30/70 이진화
df_y['y_rating_norm'] = df_y['평점'] / 5.0
df_y['y_review_norm'] = np.log1p(df_y['리뷰수']) / np.log1p(df_y['리뷰수'].max())
df_y['y_composite']   = df_y['y_rating_norm'] * df_y['y_review_norm']

th_lo = df_y['y_composite'].quantile(0.30)
th_hi = df_y['y_composite'].quantile(0.70)
```

**가중치 1:1 근거**: 시장 침투(리뷰수)와 소비자 만족(평점) 중 어느 쪽이 더 중요한지 선험적 근거 없음 → 동등 가중치. 곱셈 구조는 AND 조건 — 두 지표 모두 높아야 성공으로 분류.

**전처리**: 평점 결측 35개 + 5 초과 이상치 21개 자동 제거  
**학습 샘플**: 3,546개 (긍정=1,773, 부정=1,773)

---

## 4. 피처 설계

### 4.1 최종 피처 구성 (ml_v13)

| 그룹 | 피처 | 수 |
|------|------|-----|
| 가격대 | price_low, price_mid, price_high | 3 |
| 용량/SPF | volume_ml, SPF_Index | 2 |
| 카테고리(대) | cat_skincare, cat_cleansing, cat_suncare, cat_masks | 4 |
| 카테고리(중) | mid_cream, mid_moisturizers, mid_treatments 등 (n≥50) | 11 |
| US 트렌드 성분 | gt_* 20개 + us_trend_ratio | 21 |
| Position (기존) | niacinamide, ceramide, hyaluronic_acid, zinc_oxide, centella 등 | 17 |
| Position (신규) | tocopheryl_acetate, peptide, panthenol, caffeine, retinol × 2 | 10 |

### 4.2 k_beauty 피처 제거 근거

ml_v12 importance pruning 결과 k_beauty 9개 중 6개가 importance=0으로 제거됨.  
잔존한 k_centella(32), k_ginseng, k_bakuchiol도 gt_centella_asiatica로 이미 커버.  
→ k_beauty 10개 피처(9개 + k_beauty_ratio) 전량 제거.

### 4.3 신규 position 피처 계산 방법

성분 표기는 한국/미국 모두 농도 내림차순 의무 표기 → 위치가 농도를 반영.

```python
# {ingredient}_position: 정규화 위치 (앞=1, 없음=-1)
df_y[f'{feat}_position'] = np.where(
    idx >= 0, 1 - idx / total_ing_count, -1
)

# {ingredient}_above_1pct: phenoxyethanol(~1% 기준 프록시)보다 앞이면 1
df_y[f'{feat}_above_1pct'] = np.where(
    (idx >= 0) & (pheno_idx >= 0),
    (idx < pheno_idx).astype(int), -1
)
```

추가 대상 선정 기준: GT 커버리지 + ml_v12 importance 상위 성분

| 성분 | 커버리지 | 선정 근거 |
|------|---------|---------|
| tocopheryl_acetate | 14.2% | GT importance 최상위 |
| peptide | 14.6% | importance 상위권 |
| panthenol | 12.0% | 커버리지 안정적 |
| caffeine | 7.3% | 미국 트렌드 성분 |
| retinol | 6.2% | 고관여 성분, 위치가 효능에 직결 |

---

## 5. 모델 학습 파이프라인

### 5.1 전처리

- coverage < 5% GT 성분 제거 (11개)
- importance = 0 피처 제거 (pruning)
- Train/Test 분리: 80/20, stratify=target

### 5.2 하이퍼파라미터 최적화

**Optuna TPE (n_trials=200)** 사용:

```python
params = {
    'n_estimators':      trial.suggest_int('n_estimators', 100, 800),
    'max_depth':         trial.suggest_int('max_depth', 3, 12),
    'num_leaves':        trial.suggest_int('num_leaves', 15, 150),
    'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 60),
    'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
    'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
    'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
}
```

기존 RandomizedSearch(n=100) + GridSearch 2단계 방식 대비 탐색 공간 확대 및 베이지안 최적화 적용.

### 5.3 교차검증

StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

---

## 6. 앙상블 실험 결과

LightGBM + XGBoost 소프트 보팅(0.5/0.5) 시도:

| 모델 | Test AUC |
|------|----------|
| LightGBM | 0.7900 |
| XGBoost | 0.7865 |
| 앙상블 (0.5/0.5) | 0.7902 (+0.0001) |

XGBoost가 LightGBM보다 약한 상태에서의 동등 가중치 앙상블은 강한 모델을 희석시켜 효과 미미.  
→ LightGBM 단독 최종 채택.

---

## 7. 최종 성능

| 지표 | 값 |
|------|-----|
| Train CV AUC | 0.7895 |
| **Test AUC** | **0.7900** |
| 과적합 갭 | **-0.0005** |
| 학습 샘플 수 | 3,546개 |
| 최종 피처 수 | pruning 후 확정 |
| 저장 경로 | `C:\workspace\finalproject\data\model_output\lgbm_v13.pkl` |

과적합 갭 -0.0005는 Train CV ≈ Test로 완전한 일반화를 의미.

---

## 8. 모델 해석 및 서비스 활용 시 유의사항

**모델이 예측하는 것**: "이 상품의 성분/가격/카테고리 프로필이 미국 뷰티 시장에서 높은 소비자 반응(평점×리뷰수)을 보인 상품들과 유사한가"

**생존 편향 주의**: 학습 데이터는 Amazon/Sephora/ULTA에 이미 진입 성공한 상품만 포함. 시장 진입 실패 상품은 미포함.

**서비스 표현 권장안**:
- ❌ "이 상품은 미국 시장에서 성공할 것입니다"
- ✅ "이 상품의 성분 프로필은 미국 시장 인기 상품과 **X%** 유사합니다"

**선케어 카테고리 한계**: 학습 데이터 내 선케어 labeled 샘플 ~55개로 예측 신뢰도 낮음. 서비스 시 별도 안내 필요.

---

## 9. 향후 개선 방향 (0.79 → 0.80)

현재 과적합 갭 -0.0005로 하이퍼파라미터 튜닝으로는 개선 불가능. 새로운 피처 신호 필요.

| 후보 피처 | 기대 |
|---------|------|
| 가격 × 카테고리 교호작용 | 카테고리별 가격 민감도 차별화 |
| 성분 총 개수 (complexity) | 포뮬레이션 고급화 proxy |
| 상위 3개 성분 패턴 | 핵심 성분 조합이 포뮬레이션 철학 반영 |
| phenoxyethanol 위치 | 성분 총 농도 밀도 proxy |

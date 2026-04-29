# ml_v17 모델 설계 문서

## 1. 개요

**베이스라인**: ml_v14 (Test AUC 0.7852, volume_ml proxy 포함)  
**목표**: volume_ml proxy 제거 후 정직한 성능 회복 및 개선  
**데이터**: `ml_AmazonSephoraUlta.csv` (5,964개)  
**최종 AUC**: 0.7462 (volume_ml 완전 제거 기준)

---

## 2. volume_ml 제거 결정

### 2.1 proxy 문제 재확인 (ml_v14 → ml_v15)

ml_v14에서 확인된 proxy 구조가 용량 데이터 보완 후에도 해소되지 않음.

| 조건 | 긍정비율 | 비고 |
|------|---------|------|
| 용량 미기재 (NaN→0) | 33.1% | ml_v1 파일 기준 |
| 용량 기재됨 | 67.7% | 기재 여부가 target과 연동 |
| 피어슨 상관계수 | **0.041** | 실제 용량값 신호 매우 약함 |

ml_AmazonSephoraUlta_v2.csv로 용량 보유율을 82% 이상으로 높여도:
- 긍정/부정 median 모두 50.3ml (median imputation 집중)
- volume_ml importance 여전히 1위 → LightGBM이 "median값 = 미기재 제품" split 학습

**결론**: 용량 보유율과 무관하게 volume_ml은 proxy 역할을 유지 → 서비스 신뢰성 위해 완전 제거.

### 2.2 제거 전후 AUC 변화

| 버전 | 변경 내용 | Test AUC |
|-----|---------|---------|
| ml_v14 | 베이스라인 (volume_ml 포함, 59.6% 실측) | 0.7852 |
| ml_v15 | ingredient_count 추가 (volume_ml 포함, 82.4% 실측) | 0.7433 |
| ml_v16 | volume_ml 제거 | 0.7300 |
| **ml_v17** | 신규 피처 3종 추가 | **0.7462** |

volume_ml 제거로 -0.0133 하락 → 일부 실제 신호도 있었음.  
신규 피처로 +0.0162 회복하여 v16 대비 +0.0162 달성.

---

## 3. ml_v17 신규 피처

### 3.1 가격 연속값 (log_price)

기존 bin 방식(low/mid/high)은 구간 내 가격 차이를 무시.

```python
df_y['log_price'] = np.log1p(df_y['공급가(USD)'])
```

log 변환 이유: 공급가 분포가 우편향 → log 변환으로 정규화.  
pruning 후 **importance 1위(683)** → 가격 연속값이 핵심 신호 확인.

### 3.2 카테고리 내 상대 가격 순위 (price_rank_in_cat)

절대가격보다 같은 카테고리 내 상대적 위치가 더 의미 있을 수 있음.

```python
df_y['price_rank_in_cat'] = df_y.groupby('target_category')['공급가(USD)'].rank(pct=True)
```

예: 스킨케어 $30은 저가이지만, 마스크 $30은 고가.

### 3.3 log_price × category 교호작용

카테고리별로 가격 민감도가 다름을 반영.

```python
df_y['interact_logprice_skincare']  = df_y['log_price'] * df_y['cat_skincare']
df_y['interact_logprice_cleansing'] = df_y['log_price'] * df_y['cat_cleansing']
df_y['interact_logprice_masks']     = df_y['log_price'] * df_y['cat_masks']
df_y['interact_logprice_suncare']   = df_y['log_price'] * df_y['cat_suncare']
```

### 3.4 us_trend_ratio × price bin 교호작용

트렌드 성분 + 가격대 조합 신호.

```python
df_y['interact_trend_price_low']  = df_y['us_trend_ratio'] * df_y['price_low']
df_y['interact_trend_price_mid']  = df_y['us_trend_ratio'] * df_y['price_mid']
df_y['interact_trend_price_high'] = df_y['us_trend_ratio'] * df_y['price_high']
```

### 3.5 masks/suncare 가격 교호작용 (ml_v16 추가분)

ml_v14까지 skincare/cleansing 가격 교호작용만 존재 → masks/suncare 추가.

```python
df_y['interact_masks_price_low']    = df_y['cat_masks']   * df_y['price_low']
df_y['interact_masks_price_mid']    = df_y['cat_masks']   * df_y['price_mid']
df_y['interact_suncare_price_high'] = df_y['cat_suncare'] * df_y['price_high']
df_y['interact_suncare_price_mid']  = df_y['cat_suncare'] * df_y['price_mid']
```

### 3.6 ingredient_count × category 교호작용

카테고리별로 성분 수의 의미가 다름 (스킨케어: 많을수록 프리미엄, 클렌징: 적을수록 심플).

```python
df_y['interact_ic_skincare']  = df_y['ingredient_count'] * df_y['cat_skincare']
df_y['interact_ic_cleansing'] = df_y['ingredient_count'] * df_y['cat_cleansing']
df_y['interact_ic_masks']     = df_y['ingredient_count'] * df_y['cat_masks']
df_y['interact_ic_suncare']   = df_y['ingredient_count'] * df_y['cat_suncare']
```

---

## 4. 최종 성능

### 4.1 전체 성능

| 지표 | 값 |
|------|-----|
| Train CV AUC | 0.7499 |
| **Test AUC** | **0.7462** |
| 과적합 갭 | +0.0037 |
| 최종 피처 수 | 77개 |
| 저장 경로 | `C:\workspace\finalproject\data\model_output\lgbm_v17.pkl` |

### 4.2 카테고리별 Test 성능

| 카테고리 | n (Test) | 긍정비율 | AUC | F1 | 신뢰도 |
|---------|---------|---------|-----|-----|------|
| skincare | 491 | 48.3% | 0.7659 | 0.6898 | ✅ |
| cleansing | 114 | 60.5% | 0.6470 | 0.7114 | ✅ |
| masks | 46 | 39.1% | 0.7679 | 0.6486 | △ |
| suncare | 59 | 52.5% | 0.6982 | 0.7273 | ✅ |
| **전체** | **710** | **50.0%** | **0.7462** | **0.6959** | |

cleansing AUC 0.6470이 전체 성능의 병목.  
원인: amino surfactant 커버리지 ~3%, sulfate 커버리지 ~0% → 클렌징 성분 신호 구조적 부족.

---

## 5. 피처 구성 (ml_v17 최종)

| 그룹 | 피처 | 수 |
|------|------|---|
| 가격대 bin | price_low, price_mid, price_high | 3 |
| 가격 연속 | log_price, price_rank_in_cat | 2 |
| SPF | SPF_Index | 1 |
| 성분 수 | ingredient_count | 1 |
| 카테고리(대) | cat_skincare, cat_cleansing, cat_suncare, cat_masks | 4 |
| 카테고리(중) | mid_* (n≥50) | 11 |
| US 트렌드 성분 | gt_* 9개 (커버리지 5% 이상) + us_trend_ratio | 10 |
| Position (기존) | niacinamide, ceramide, hyaluronic_acid 등 | 17 |
| Position (신규) | tocopheryl_acetate, peptide, panthenol, caffeine, retinol | 5 |
| 카테고리 전용 | is_physical_filter, is_chemical_only 등 | 2 |
| 교호작용 (cat×신호) | interact_cleansing_*, interact_skincare_*, 등 | 13 |
| 교호작용 (ic×cat) | interact_ic_* | 4 |
| 교호작용 (logprice×cat) | interact_logprice_* | 4 |
| 교호작용 (trend×price) | interact_trend_price_* | 3 |
| 교호작용 (pos×cat) | pos_clean_*, pos_skin_*, pos_sun_*, pos_mask_* | ~12 |

pruning 후 최종 피처 수: **77개**

---

## 6. CATEGORY_THRESHOLDS

```python
CATEGORY_THRESHOLDS = {
    'skincare':  0.50,
    'cleansing': 0.50,
    'masks':     0.40,   # 긍정비율 39% → threshold 낮춤
    'suncare':   0.50,
}
```

---

## 7. 버전별 성능 흐름

| 버전 | Test AUC | 주요 변경 |
|-----|---------|---------|
| ml_v13 | 0.7900 | position 피처 확대 |
| ml_v14 | 0.7852 | 카테고리 교호작용 + volume proxy 해소 시도 |
| ml_v15 | 0.7433 | ingredient_count 추가 / 용량 82% 실측 |
| ml_v16 | 0.7300 | volume_ml 완전 제거 |
| **ml_v17** | **0.7462** | log_price + price_rank_in_cat + 교호작용 확대 |

v13/v14의 0.78~0.79는 volume_ml proxy 효과 포함 수치.  
v17의 **0.7462는 proxy 없는 정직한 성능**으로 성분·가격 기반 신호만 반영.

---

## 8. 서비스 활용 시 유의사항

**volume_ml 제거**: 서비스 inference에 용량 입력 불필요. 모델이 용량 피처를 사용하지 않음.

**cleansing 예측 신뢰도**: AUC 0.647로 다른 카테고리 대비 낮음. 클렌징 특성상 성분 신호가 약함.

**masks threshold**: 긍정비율 39%로 불균형 → threshold 0.40 적용. 서비스에서 CATEGORY_THRESHOLDS 딕셔너리 사용 필수.

**생존 편향**: Amazon/Sephora/ULTA 진입 성공 상품만 학습. 시장 진입 실패 상품 미포함.

---

## 9. 향후 개선 방향

| 후보 | 기대 효과 |
|-----|---------|
| cleansing 데이터 확대 | cleansing 구조적 한계(학습 샘플 부족) 해소 |
| suncare SPF 세분화 | SPF 구간별 미국 트렌드 반영 |
| 카테고리 내 성분 다양성 | 동일 카테고리 내 포뮬레이션 차별화 신호 |
| 리뷰 텍스트 감성 피처 | 현재 사용 안 하는 정성적 신호 활용 |

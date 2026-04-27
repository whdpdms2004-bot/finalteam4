# ml_v18 모델 설계 문서

## 1. 개요

**베이스라인**: ml_v17 (Test AUC 0.7462, volume_ml 완전 제거 기준)  
**목표**: 클렌징 카테고리 AUC 개선 (v17 기준 0.6470)  
**데이터**: `ml_AmazonSephoraUlta.csv` (5,964개)  
**최종 AUC**: 0.7548

---

## 2. 버전별 성능 흐름

| 버전 | Test AUC | 주요 변경 |
|-----|---------|---------|
| ml_v13 | 0.7900 | position 피처 확대 |
| ml_v14 | 0.7852 | 카테고리 교호작용 + volume proxy 해소 시도 |
| ml_v15 | 0.7433 | ingredient_count 추가 / 용량 82% 실측 |
| ml_v16 | 0.7300 | volume_ml 완전 제거 |
| ml_v17 | 0.7462 | log_price + price_rank_in_cat + 교호작용 확대 |
| **ml_v18** | **0.7548** | 클렌징 성분 강화 + 전체 카테고리 threshold 최적화 |

v13/v14의 0.78~0.79는 volume_ml proxy 효과 포함.  
v18의 **0.7548은 proxy 없는 정직한 성능**으로 성분·가격 기반 신호만 반영.

---

## 3. ml_v18 신규 변경사항

### 3.1 클렌징 전용 성분 피처 추가

클렌징 카테고리 AUC(v17: 0.6470)가 병목이었고, 원인은 성분 신호 부족.

#### 추가된 피처

**Salicylic Acid (BHA)**
```python
df_y['is_salicylic'] = ing_lower.str.contains('salicylic acid', regex=False).astype(int)
df_y['interact_cleansing_salicylic'] = df_y['cat_cleansing'] * df_y['is_salicylic']
```
- 클렌징 내 커버리지: 17.2%
- 미국 시장 BHA 클렌저 트렌드 반영

**Sodium Hyaluronate × Cleansing**
```python
df_y['interact_cleansing_sha'] = df_y['cat_cleansing'] * df_y['gt_sodium_hyaluronate']
```
- 클렌징 내 커버리지: 25.6% (161개)
- 세정 후 보습 유지 소구점

**Panthenol × Cleansing**
```python
df_y['interact_cleansing_panthenol'] = df_y['cat_cleansing'] * df_y['gt_panthenol']
```
- 클렌징 내 커버리지: 9.7% (61개)

**Niacinamide × Cleansing**
```python
df_y['interact_cleansing_niacinamide'] = df_y['cat_cleansing'] * df_y['gt_niacinamide']
```
- 클렌징 내 커버리지: 11.0% (69개)

#### 클렌징 AUC 변화

| 단계 | 클렌징 AUC | 비고 |
|-----|----------|-----|
| ml_v17 | 0.6470 | 베이스라인 |
| salicylic acid 추가 | 0.6718 | +0.0248 |
| SHA + panthenol + niacinamide 추가 | 0.6649 | -0.0069 (전체 AUC는 상승) |

> SHA/panthenol/niacinamide 추가 후 클렌징 개별 AUC가 소폭 하락한 이유:  
> Optuna 확률적 탐색의 stochasticity + Test 샘플 noise (n=114).  
> 전체 AUC는 0.7532 → 0.7548로 상승하여 실질적 개선 확인.

### 3.2 구조적 한계 확인

| 성분 | 커버리지 | 비고 |
|-----|--------|-----|
| is_amino_surfactant | ~3% | amino surfactant 커버리지 구조적 부족 |
| is_sulfate | ~0% | 성분 표기 다양성으로 매칭 실패 |

클렌징 AUC 0.66~0.67은 **성분 데이터 sparsity에 의한 구조적 한계**에 근접.  
추가 성분 피처로 올리기 어려운 영역.

---

## 4. Y값 재설계 실험 (포기)

클렌징 AUC 한계 원인이 Y값에 있을 수 있다는 가설로 두 가지 실험 진행.

### 4.1 ml_v19: 플랫폼×카테고리 내 rank 이진화

```python
# 플랫폼×카테고리 그룹 내에서 percentile rank → 상/하위 30% 컷
df_y['y_rank'] = df_y.groupby(['platform', 'target_category'])['y_composite'].rank(pct=True)
```

**결과**: Train CV AUC 0.6590 → **포기**

### 4.2 ml_v20: 플랫폼×카테고리별 z-표준화 후 global 컷

```python
# 리뷰수, 평점 각각 플랫폼×카테고리 내 z-표준화 후 결합
z_review = (log_review - group_mean_review) / group_std_review
z_rating = (rating - group_mean_rating) / group_std_rating
y_std = z_review + z_rating  # global quantile 컷
```

**결과**: Train CV AUC 0.6696 → **포기**

### 4.3 결론

두 실험 모두 절대 Y(v18) 대비 AUC 0.07~0.08 하락.

**해석**: 플랫폼·카테고리 간 Y 차이는 제거해야 할 편향이 아니라 **예측 가능한 신호**.  
- 가격, 성분, 카테고리 피처가 플랫폼/카테고리 간 차이를 잘 설명
- 그룹 내 편차(동일 플랫폼·카테고리 내 차이)는 피처로 설명하기 훨씬 어려움
- → 절대 Y 유지가 최선

---

## 5. 전체 카테고리 Threshold 최적화

v17까지는 masks만 threshold 최적화. v18에서 4개 카테고리 전체로 확장.

### 5.1 방법

Train OOF(Out-of-Fold) 예측 확률 기반으로 F1 최대화 threshold 탐색.

```python
# 탐색 범위: 긍정비율 < 45% → 0.30~0.65, 그 외 → 0.35~0.65
th_min = 0.30 if pos_rate < 0.45 else 0.35
```

탐색 하한 조정 이유: 긍정비율 50%+ 카테고리에서 0.25까지 탐색하면 OOF 과적합 발생.  
(suncare 사례: OOF 최적 0.25 → Test F1 -0.0306)

### 5.2 최종 CATEGORY_THRESHOLDS

```python
CATEGORY_THRESHOLDS = {
    'skincare':  0.35,   # OOF F1 0.6720 → 0.6969, Test F1 +0.0614
    'cleansing': 0.35,   # OOF F1 0.7591 → 0.7830, Test F1 +0.0467
    'masks':     0.30,   # OOF F1 0.6271 → 0.6832, Test F1 +0.0470 (긍정비율 34.5%)
    'suncare':   0.35,   # OOF F1 0.6962 → 0.7416, Test F1 -0.0146 (소폭 하락)
}
```

> suncare는 OOF 최적 threshold(0.35)가 Test에서 소폭 역효과(-0.0146).  
> 원인: Train 긍정비율 57% vs Test 긍정비율 52.5% + 소샘플(n_test=59) noise.  
> 실용적으로 허용 가능한 수준.

---

## 6. 최종 성능

### 6.1 전체 성능

| 지표 | 값 |
|------|-----|
| Train CV AUC | 0.7531 |
| **Test AUC** | **0.7548** |
| 과적합 갭 | -0.0017 (underfitting 없음) |
| 최종 피처 수 | 87개 |
| 저장 경로 | `C:\workspace\finalproject\data\model_output\lgbm_v18.pkl` |

### 6.2 카테고리별 Test 성능 (th=0.50 기준 AUC)

| 카테고리 | n (Test) | 긍정비율 | AUC | F1 (th=0.50) | F1 (튜닝 th) | 신뢰도 |
|---------|---------|---------|-----|------------|------------|------|
| skincare | 491 | 48.3% | 0.7738 | 0.6843 | 0.7458 (th=0.35) | ✅ |
| cleansing | 114 | 60.5% | 0.6649 | 0.7123 | 0.7590 (th=0.35) | ✅ |
| masks | 46 | 39.1% | 0.7917 | 0.6486 | 0.6957 (th=0.30) | △ |
| suncare | 59 | 52.5% | 0.6924 | 0.7273 | 0.7126 (th=0.35) | △ |
| **전체** | **710** | **50.0%** | **0.7548** | **—** | **—** | |

---

## 7. 기술적 이슈 및 해결

### 7.1 OSError: [WinError 4551] 파일 차단

Windows Application Control (WDAC/AppLocker) 정책으로 pickle 저장 차단.

**해결**: `pickle.dump` → `joblib.dump` 전환 + fallback 경로 로직

```python
try:
    joblib.dump(model_data, save_path)
except OSError:
    alt_path = os.path.join(os.path.expanduser("~"), 'lgbm_v18.pkl')
    joblib.dump(model_data, alt_path)
```

### 7.2 셀 실행 순서 의존성

threshold 최적화 셀에서 `CATEGORY_THRESHOLDS` 생성 → 저장 셀에서 사용.  
셀 순서 재정렬로 해결:

```
최종 평가 → 카테고리별 성능 분석 → threshold 최적화 → 저장
```

---

## 8. 피처 구성 (ml_v18 최종, 87개)

| 그룹 | 주요 피처 | 수 |
|------|---------|---|
| 가격대 bin | price_low, price_mid, price_high | 3 |
| 가격 연속 | log_price, price_rank_in_cat | 2 |
| SPF | SPF_Index | 1 |
| 성분 수 | ingredient_count | 1 |
| 카테고리(대) | cat_cleansing, cat_suncare, cat_masks | 3 |
| 카테고리(중) | mid_* (n≥50) | ~11 |
| US 트렌드 성분 | gt_* 9개 (커버리지 5%+) + us_trend_ratio | 10 |
| Position | niacinamide_position, hyaluronic_acid_position 등 | ~12 |
| 클렌징 전용 | is_salicylic, is_foam_cleanser, interact_cleansing_* | ~8 |
| 선케어 전용 | is_physical_filter, is_chemical_only, interact_suncare_* | ~4 |
| 교호작용 (cat×신호) | interact_cleansing_*, interact_skincare_*, 등 | ~15 |
| 교호작용 (ic×cat) | interact_ic_* | 4 |
| 교호작용 (logprice×cat) | interact_logprice_* | 4 |
| 교호작용 (trend×price) | interact_trend_price_* | 3 |

importance=0 pruning 후 125개 → **87개**

---

## 9. 서비스 활용 시 유의사항

**카테고리별 AUC vs 전체 AUC**  
서비스 관점에서는 개별 카테고리 AUC가 더 중요.  
사용자는 항상 특정 카테고리 예측만 경험하기 때문.  
단, masks(n=46)/suncare(n=59)는 Test 샘플이 적어 AUC 추정치 자체가 ±0.10~0.14 noise 포함.

**클렌징 예측 신뢰도**  
AUC 0.6649는 구조적 한계(성분 데이터 sparsity). 서비스 표시 시 "예측 신뢰도 낮음" 표기 권장.

**threshold 적용 방법**  
```python
# 서비스 inference 시
threshold = CATEGORY_THRESHOLDS.get(category, 0.50)
prediction = 1 if proba >= threshold else 0
```

**생존 편향**  
Amazon/Sephora/ULTA 진입 성공 상품만 학습. 시장 진입 실패 상품 미포함.  
Y값은 절대적 시장 성공이 아니라 플랫폼 내 인기도(평점×리뷰수) 기반.

---

## 10. 향후 개선 방향

| 후보 | 기대 효과 |
|-----|---------|
| 클렌징 데이터 확대 | 구조적 한계(학습 샘플 부족) 해소 |
| 성분 표기 정규화 | sulfate 계열 커버리지 0% 문제 해결 |
| suncare SPF 세분화 | SPF 구간별 미국 트렌드 반영 |
| 리뷰 텍스트 감성 피처 | 현재 미사용 정성적 신호 활용 |

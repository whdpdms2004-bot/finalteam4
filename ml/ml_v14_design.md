# ml_v14 모델 설계 문서

## 1. 개요

**베이스라인**: ml_v13 (Test AUC 0.7900, 과적합 갭 -0.0005)  
**목표**: 전체 AUC 0.80 + 카테고리별 균형 예측  
**데이터**: `ml_AmazonSephoraUlta_v2.csv` (5,964개, 용량 결측치 보완 버전)  
**최종 AUC**: 0.7852 (proxy variable 제거 후 정직한 수치)

---

## 2. 카테고리별 성능 분석 도입

### 2.1 배경

ml_v13까지 전체 AUC만 평가 → 카테고리별 편향 미파악

### 2.2 분석 결과 (ml_v13 기준)

| 카테고리 | n (Test) | AUC | 신뢰도 |
|---------|---------|-----|------|
| skincare | 491 | 0.8175 | ✅ |
| cleansing | 114 | 0.6667 | ✅ |
| masks | 46 | 0.7143 | △ |
| suncare | 59 | 0.7679 | ✅ |

cleansing 0.6667로 스킨케어 대비 -0.15 격차 확인 → 카테고리 전용 피처 필요

---

## 3. Masks Threshold 최적화

### 3.1 문제

masks 긍정비율 39.1% (불균형) → default threshold 0.50이 부정 편향 예측

### 3.2 해결

Train OOF 확률로 threshold sweep → 최적값 적용 (데이터 누수 없음)

| Threshold | Train OOF F1 | Test F1 | 변화 |
|----------|-------------|---------|-----|
| 0.50 (기본) | 0.7133 | 0.5946 | — |
| **0.40 (최적)** | **0.7320** | **0.6512** | **+0.0566** |

```python
CATEGORY_THRESHOLDS = {
    'skincare':  0.50,
    'cleansing': 0.50,
    'masks':     0.40,   # 긍정비율 39% → threshold 낮춤
    'suncare':   0.50,
}
```

---

## 4. 피처 실험 이력

### 4.1 제형 피처 시도 → 실패

클렌징 제형(오일/밤/미셀라)을 카테고리(중) 또는 성분 텍스트에서 추출:

```python
df_y['is_oil_cleanser']  = (df_y['카테고리(중)'] == 'cleansing oil').astype(int)
df_y['is_balm_cleanser'] = (df_y['카테고리(중)'] == 'cleansing balm').astype(int)
df_y['is_micellar']      = ing_lower.str.contains(r'poloxamer|isohexadecane|micellar')
```

**결과**: cleansing 0.6680 → 0.6594 (-0.0086)  
**원인**: is_oil_cleanser(n=42), is_balm_cleanser(n=28) — 너무 희소해서 노이즈  
→ 제거

### 4.2 교호작용 피처 (카테고리 × 신호) → 성공

```python
df_y['interact_cleansing_amino']     = df_y['cat_cleansing'] * df_y['is_amino_surfactant']
df_y['interact_cleansing_trend']     = df_y['cat_cleansing'] * df_y['us_trend_ratio']
df_y['interact_cleansing_price_low'] = df_y['cat_cleansing'] * df_y['price_low']
df_y['interact_cleansing_price_mid'] = df_y['cat_cleansing'] * df_y['price_mid']
df_y['interact_skincare_trend']      = df_y['cat_skincare']  * df_y['us_trend_ratio']
df_y['interact_skincare_price_high'] = df_y['cat_skincare']  * df_y['price_high']
df_y['interact_suncare_physical']    = df_y['cat_suncare']   * df_y['is_physical_filter']
df_y['interact_suncare_spf_tier']    = df_y['cat_suncare']   * df_y['spf_tier']
df_y['interact_masks_trend']         = df_y['cat_masks']     * df_y['us_trend_ratio']
```

**효과**: cleansing 0.6667 → 0.6765 (+0.0098), 전체 0.7900 → 0.7911 (+0.0011)

### 4.3 Position × Category 교호작용 추가

```python
pos_cat_pairs = [
    ('niacinamide_position',           'cat_cleansing', 'pos_clean_niacinamide'),
    ('amino_acid_surfactant_position', 'cat_cleansing', 'pos_clean_amino'),
    ('us_trend_ingredient_position',   'cat_cleansing', 'pos_clean_trend'),
    ('niacinamide_position',           'cat_skincare',  'pos_skin_niacinamide'),
    ('ceramide_position',              'cat_skincare',  'pos_skin_ceramide'),
    ('hyaluronic_acid_position',       'cat_skincare',  'pos_skin_ha'),
    ('tocopheryl_acetate_position',    'cat_skincare',  'pos_skin_tocopheryl'),
    ('retinol_position',               'cat_skincare',  'pos_skin_retinol'),
    ('zinc_oxide_position',            'cat_suncare',   'pos_sun_zinc'),
    ('us_trend_ingredient_position',   'cat_suncare',   'pos_sun_trend'),
    ('hyaluronic_acid_position',       'cat_masks',     'pos_mask_ha'),
    ('niacinamide_position',           'cat_masks',     'pos_mask_niacinamide'),
]
```

---

## 5. volume_ml Proxy Variable 문제 발견 및 해결

### 5.1 문제 발견

`volume_ml` 피처 중요도가 압도적 1위 → 의심

**진단 결과**:

| 구분 | 긍정비율 | 샘플수 |
|-----|---------|------|
| 미기재(NaN→0) | 33.1% | 1,817 |
| 기재됨 | 67.7% | 1,729 |

- Chi-square p-value: 0.0000 → 유의 (기재여부가 target과 연동)
- **결정적 증거**: 기재된 경우만 보면 target별 분포 거의 동일 (median 50.0 vs 50.3)
- **결론**: 모델이 실제 용량을 학습한 게 아니라 "용량 기재됨 = 잘 팔리는 상품" proxy 학습

### 5.2 해결 과정

1. **v2 파일**: 용량 결측치를 수기 보완 (전체 67.7% 기재)  
2. **문제**: 보완된 값이 단위 없는 순수숫자 (`"50.27495"`) → 기존 regex 미인식  
3. **extract_ml 수정**: ml/oz 패턴 외 순수숫자(`re.fullmatch(r'[\d.]+')`)도 인식, `"1 ct"` 등 비용량 값 제외  
4. **Median Imputation**: 남은 NaN → 중앙값으로 대체

```python
def extract_ml(s):
    if pd.isna(s): return np.nan
    s_str = str(s).strip()
    m = re.search(r'([\d.]+)\s*ml', s_str, re.IGNORECASE)
    if m: return float(m.group(1))
    m = re.search(r'([\d.]+)\s*oz', s_str, re.IGNORECASE)
    if m: return float(m.group(1)) * 29.5735
    if re.fullmatch(r'[\d.]+', s_str):
        return float(s_str)
    return np.nan

df_y['volume_ml'] = df_y['용량'].apply(extract_ml)
vol_median = df_y['volume_ml'].median()
df_y['volume_ml'] = df_y['volume_ml'].fillna(vol_median)
```

### 5.3 Median Imputation의 한계

median imputation 후에도 `volume_ml` 중요도 1위 유지.  
원인: NaN인 제품들이 전부 동일한 중앙값에 집중 → LightGBM이 "volume_ml == 중앙값" split으로 여전히 proxy 학습 가능.

**서비스 관점**: 한국 브랜드가 항상 실제 용량을 입력하면 inference 시 proxy 효과 없음. 실제 용량값의 신호는 미미(median 50.0 vs 50.3)하므로 volume_ml의 기여는 제한적.

---

## 6. 최종 성능 비교

| 버전 | 전체 AUC | skincare | cleansing | masks | suncare |
|-----|---------|---------|---------|------|--------|
| ml_v13 | 0.7900 | 0.8175 | 0.6667 | 0.7143 | 0.7679 |
| ml_v14 (제형 피처) | 0.7920 | 0.8186 | 0.6680 | 0.7242 | 0.7419 |
| ml_v14 (교호작용) | 0.7911 | 0.8165 | **0.6765** | 0.7083 | 0.7742 |
| **ml_v14 최종** | **0.7852** | 0.8106 | 0.6696 | 0.7004 | 0.7604 |

ml_v14 최종은 volume proxy 제거 버전. 전체 AUC는 소폭 하락했으나 더 정직한 수치.

---

## 7. 피처 구성 (ml_v14 최종)

| 그룹 | 피처 | 수 |
|------|------|---|
| 가격대 | price_low, price_mid, price_high | 3 |
| 용량/SPF | volume_ml (median imputed), SPF_Index | 2 |
| 카테고리(대) | cat_skincare, cat_cleansing, cat_suncare, cat_masks | 4 |
| 카테고리(중) | mid_* (n≥50) | 11 |
| US 트렌드 성분 | gt_* 9개 (커버리지 5% 이상) + us_trend_ratio | 10 |
| Position (기존) | niacinamide, ceramide, hyaluronic_acid 등 | 17 |
| Position (신규) | tocopheryl_acetate, peptide, panthenol, caffeine, retinol | 10 |
| 카테고리 전용 | is_amino_surfactant, is_physical_filter, is_chemical_only, spf_tier | 4 |
| 교호작용 (cat×신호) | interact_cleansing_*, interact_skincare_*, 등 | 9 |
| 교호작용 (pos×cat) | pos_clean_*, pos_skin_*, pos_sun_*, pos_mask_* | 12 |

pruning 후 최종 피처 수: **69개**

---

## 8. 저장 정보

| 항목 | 값 |
|------|---|
| 저장 경로 | `C:\workspace\finalproject\data\model_output\lgbm_v14.pkl` |
| 데이터 파일 | `ml_AmazonSephoraUlta_v2.csv` |
| Train CV AUC | 0.7892 |
| Test AUC | 0.7852 |
| 과적합 갭 | +0.0040 |
| CATEGORY_THRESHOLDS | skincare 0.50 / cleansing 0.50 / masks **0.40** / suncare 0.50 |

---

## 9. 서비스 활용 시 유의사항

**volume_ml 입력**: 서비스에서 ml 단위 숫자 입력 필드로 받을 것. 미입력 시 학습 중앙값으로 대체.

**cleansing 예측 신뢰도**: AUC 0.67로 다른 카테고리 대비 낮음. 클렌징 특성상 성분 신호가 약함 (is_sulfate 커버리지 0%, is_amino_surfactant 2.8%).

**masks threshold**: 긍정비율 39%로 불균형 → threshold 0.40 적용. 서비스에서 CATEGORY_THRESHOLDS 딕셔너리 사용 필수.

**생존 편향**: ml_v13 설계 문서 동일. Amazon/Sephora/ULTA 진입 성공 상품만 학습.

---

## 10. 향후 개선 방향

현재 bottleneck은 proxy 제거 후 0.785 수준의 성분/가격 기반 예측력.

| 후보 | 기대 효과 |
|-----|---------|
| 클렌징 데이터 확대 | cleansing 구조적 한계(학습 샘플 부족) 해소 |
| 성분 총 개수 (complexity) | 포뮬레이션 고급화 proxy |
| 가격 × 카테고리 교호작용 확대 | 카테고리별 가격 민감도 반영 |
| volume proxy 완전 해소 | 용량 데이터 전수 실측화 |

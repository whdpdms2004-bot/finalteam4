"""
BeautyBridge — Census API Data Refresh Script
==============================================
ACS 5-Year Estimates (2022) 최신 데이터를 자동으로 수집합니다.

사용법:
  1. API 키 발급 (무료): 5c86034d858d3231beb24d967a44cf891d99f08d
  2. 아래 API_KEY 변수에 발급받은 키를 입력하세요.
  3. 실행: python census_api_refresh.py

출력: census_raw_states.csv, census_raw_cities.csv
"""

import requests
import pandas as pd
import time
import sys

# ── 설정 ─────────────────────────────────────────────────
API_KEY = "5c86034d858d3231beb24d967a44cf891d99f08d"   # ← 여기에 발급받은 API 키 입력
YEAR    = "2024"
DATASET = "acs/acs5"
BASE    = f"https://api.census.gov/data/{YEAR}/{DATASET}"

# ACS 변수 정의 (B03002 = 히스패닉/Latino 교차 인종, B02001 = 인종 단독)
VARIABLES = {
    "B03002_001E": "Total_Population",
    "B03002_003E": "White_NonHispanic",        # White alone, not Hispanic
    "B03002_004E": "Black_NonHispanic",         # Black alone, not Hispanic
    "B03002_006E": "Asian_NonHispanic",         # Asian alone, not Hispanic
    "B03002_012E": "Hispanic_Latino",           # Hispanic or Latino (any race)
    "B02001_004E": "AIAN",                      # American Indian & Alaska Native
    "B02001_006E": "NHPI",                      # Native Hawaiian & Pacific Islander
    "B02001_007E": "Other_Race",
    "B02001_008E": "Two_Or_More_Races",
}

VAR_STRING = "NAME," + ",".join(VARIABLES.keys())


def fetch(params: dict) -> pd.DataFrame:
    r = requests.get(BASE, params={"get": VAR_STRING, "key": API_KEY, **params})

    # 추가: 실제 응답 확인
    print(f"  Status: {r.status_code}")
    print(f"  Response: {r.text[:300]}")

    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df.rename(columns=VARIABLES, inplace=True)
    for col in VARIABLES.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """비율, 다양성 지수, ShadeGap 점수 계산"""
    total = df["Total_Population"]
    for col in list(VARIABLES.values())[1:]:
        df[f"{col}_Pct"] = (df[col] / total * 100).round(1)

    def diversity(row):
        t = row["Total_Population"]
        if t == 0:
            return 0
        groups = [row["White_NonHispanic"], row["Black_NonHispanic"],
                  row["Asian_NonHispanic"], row["Hispanic_Latino"]]
        return round(1 - sum((g/t)**2 for g in groups if g > 0), 4)

    df["Diversity_Index"]      = df.apply(diversity, axis=1)
    df["ShadeGap_Opportunity"] = (df["Diversity_Index"] * 100).round(1)
    return df


def dominant_group(df: pd.DataFrame) -> pd.DataFrame:
    groups = ["White_NonHispanic", "Black_NonHispanic", "Asian_NonHispanic", "Hispanic_Latino"]
    df["Dominant_Group"] = df[groups].idxmax(axis=1)
    return df


# ── 1. 주(State) 단위 수집 ───────────────────────────────
print("📡 Fetching state-level data...")
df_states = fetch({"for": "state:*"})
df_states = add_derived(df_states)
df_states = dominant_group(df_states)
df_states.to_csv("census_raw_states.csv", index=False, encoding="utf-8-sig")
print(f"  ✅ {len(df_states)} states saved → census_raw_states.csv")
time.sleep(1)

# ── 2. 대도시(Place) 단위 수집 ──────────────────────────
# 인구 250,000명 이상 도시 타겟 (for=place:* 전체 수집 후 필터)
# 주(state) 코드별로 순회해야 place-level API 호출 가능
TARGET_STATES = {
    "06": "California", "48": "Texas",   "12": "Florida",
    "36": "New York",   "17": "Illinois","04": "Arizona",
    "42": "Pennsylvania","37": "North Carolina", "53": "Washington",
    "08": "Colorado",   "13": "Georgia", "24": "Maryland",
    "26": "Michigan",   "47": "Tennessee","32": "Nevada",
    "21": "Kentucky",   "22": "Louisiana","29": "Missouri",
}

all_cities = []
print("📡 Fetching city-level data (major states)...")
for fips, sname in TARGET_STATES.items():
    try:
        df_p = fetch({"for": f"place:*", "in": f"state:{fips}"})
        df_p["State"] = sname
        df_p = df_p[df_p["Total_Population"] >= 200000]  # 20만명 이상
        all_cities.append(df_p)
        print(f"  {sname}: {len(df_p)} cities (≥200k)")
        time.sleep(0.5)
    except Exception as e:
        print(f"  ⚠️ {sname} 수집 실패: {e}")

if all_cities:
    df_cities = pd.concat(all_cities, ignore_index=True)
    df_cities = add_derived(df_cities)
    df_cities = dominant_group(df_cities)
    df_cities.sort_values("Total_Population", ascending=False, inplace=True)
    df_cities.to_csv("census_raw_cities.csv", index=False, encoding="utf-8-sig")
    print(f"\n  ✅ {len(df_cities)} cities saved → census_raw_cities.csv")

print("\n🎉 Done! CSV 파일을 BeautyBridge_Census_Dataset.xlsx 업데이트에 활용하세요.")
print("   다음 단계: Shade Gap Map 시각화 → Streamlit app.py에 이 CSV를 로드")
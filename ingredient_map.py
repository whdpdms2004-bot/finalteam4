"""
한국어 화장품 성분명 → 영어(INCI) 변환 딕셔너리
- ML 모델이 체크하는 키워드 위주로 우선 포함
- 없는 성분은 원본 그대로 사용
"""

KOR_TO_ENG: dict[str, str] = {
    # ── 물/용매 ──────────────────────────────────────────
    "정제수": "water",
    "물": "water",
    "에탄올": "ethanol",
    "알코올": "alcohol",
    "부틸렌글라이콜": "butylene glycol",
    "프로판다이올": "propanediol",
    "글리세린": "glycerin",
    "글리세롤": "glycerol",
    "펜틸렌글라이콜": "pentylene glycol",
    "헥산다이올": "hexanediol",

    # ── 보습 핵심 ─────────────────────────────────────────
    "히알루론산": "hyaluronic acid",
    "히아루론산": "hyaluronic acid",
    "소듐하이알루로네이트": "sodium hyaluronate",
    "소듐히알루로네이트": "sodium hyaluronate",
    "히알루로네이트": "hyaluronate",
    "세라마이드": "ceramide",
    "세라미드": "ceramide",
    "판테놀": "panthenol",
    "비타민B5": "panthenol",
    "알란토인": "allantoin",
    "베타인": "betaine",
    "트레할로스": "trehalose",
    "소듐PCA": "sodium pca",

    # ── 미백/브라이트닝 ──────────────────────────────────
    "나이아신아마이드": "niacinamide",
    "나이아신아마": "niacinamide",
    "니아신아마이드": "niacinamide",
    "비타민B3": "niacinamide",
    "아스코르브산": "ascorbic acid",
    "비타민C": "ascorbic acid",
    "비타민c": "ascorbic acid",
    "에칠아스코르빌에텔": "ethyl ascorbyl ether",
    "아스코르빌글루코사이드": "ascorbyl glucoside",
    "마그네슘아스코르빌포스페이트": "magnesium ascorbyl phosphate",
    "알파알부틴": "alpha-arbutin",
    "알부틴": "arbutin",
    "코직산": "kojic acid",
    "트라넥사믹애씨드": "tranexamic acid",
    "글루타치온": "glutathione",

    # ── 주름/리프팅 ───────────────────────────────────────
    "레티놀": "retinol",
    "레티닐팔미테이트": "retinyl palmitate",
    "아데노신": "adenosine",
    "펩타이드": "peptide",
    "펩티드": "peptide",
    "아세틸헥사펩타이드": "acetyl hexapeptide",
    "팔미토일펜타펩타이드": "palmitoyl pentapeptide",
    "팔미토일트리펩타이드": "palmitoyl tripeptide",
    "구리펩타이드": "copper peptide",
    "EGF": "epidermal growth factor",
    "토코페롤": "tocopherol",
    "토코페릴아세테이트": "tocopheryl acetate",
    "비타민E": "tocopheryl acetate",

    # ── 트러블/진정 ───────────────────────────────────────
    "살리실산": "salicylic acid",
    "병풀추출물": "centella asiatica extract",
    "병풀": "centella asiatica",
    "센텔라아시아티카": "centella asiatica",
    "센텔라": "centella",
    "마데카소사이드": "madecassoside",
    "아시아티코사이드": "asiaticoside",
    "아시아틱애씨드": "asiatic acid",
    "마데카식애씨드": "madecassic acid",
    "티트리오일": "tea tree oil",
    "티트리": "tea tree",
    "위치하젤추출물": "witch hazel extract",
    "위치하젤": "witch hazel",
    "어성초추출물": "heartleaf houttuynia extract",
    "어성초": "heartleaf houttuynia",
    "쑥추출물": "artemisia extract",
    "쑥": "artemisia",
    "황금추출물": "scutellaria baicalensis extract",
    "녹차추출물": "green tea extract",
    "녹차": "green tea",
    "카모마일추출물": "chamomile extract",
    "알로에베라": "aloe vera",
    "알로에": "aloe",
    "감초추출물": "licorice extract",
    "글라브리딘": "glabridin",

    # ── 선케어 자외선차단 ─────────────────────────────────
    "징크옥사이드": "zinc oxide",
    "산화아연": "zinc oxide",
    "티타늄디옥사이드": "titanium dioxide",
    "이산화티탄": "titanium dioxide",
    "아보벤존": "avobenzone",
    "옥티녹세이트": "octinoxate",
    "옥시벤존": "oxybenzone",
    "옥티살레이트": "octisalate",
    "호모살레이트": "homosalate",
    "옥토크릴렌": "octocrylene",
    "에칠헥실메톡시신나메이트": "ethylhexyl methoxycinnamate",

    # ── 클렌징 계면활성제 ─────────────────────────────────
    "소듐라우릴설페이트": "sodium lauryl sulfate",
    "소듐라우레스설페이트": "sodium laureth sulfate",
    "암모늄라우릴설페이트": "ammonium lauryl sulfate",
    "암모늄라우레스설페이트": "ammonium laureth sulfate",
    "소듐코코일글루타메이트": "sodium cocoyl glutamate",
    "소듐라우로일글루타메이트": "sodium lauroyl glutamate",
    "포타슘코코일글리시네이트": "potassium cocoyl glycinate",
    "코카미도프로필베타인": "cocamidopropyl betaine",
    "코코아미도프로필베타인": "cocamidopropyl betaine",
    "라우릴글루코사이드": "lauryl glucoside",
    "코코글루코사이드": "coco glucoside",
    "데실글루코사이드": "decyl glucoside",
    "카프릴릴글루코사이드": "caprylyl glucoside",
    "디소듐코코암포디아세테이트": "disodium cocoamphodiacetate",
    "소듐코코암포아세테이트": "sodium cocoamphoacetate",

    # ── AHA/BHA/PHA ───────────────────────────────────────
    "락틱애씨드": "lactic acid",
    "젖산": "lactic acid",
    "글라이콜릭애씨드": "glycolic acid",
    "글리콜산": "glycolic acid",
    "만델릭애씨드": "mandelic acid",
    "글루코노락톤": "gluconolactone",
    "락토비오닉애씨드": "lactobionic acid",
    "폴리하이드록시애씨드": "polyhydroxy acid",

    # ── pH 조절 ───────────────────────────────────────────
    "시트릭애씨드": "citric acid",
    "구연산": "citric acid",
    "소듐하이드록사이드": "sodium hydroxide",
    "수산화나트륨": "sodium hydroxide",

    # ── 보존제 ───────────────────────────────────────────
    "페녹시에탄올": "phenoxyethanol",
    "에칠헥실글리세린": "ethylhexylglycerin",
    "1,2-헥산다이올": "1,2-hexanediol",
    "클로르페네신": "chlorphenesin",
    "벤질알코올": "benzyl alcohol",
    "소듐벤조에이트": "sodium benzoate",
    "포타슘소르베이트": "potassium sorbate",

    # ── 오일/에모리언트 ───────────────────────────────────
    "스쿠알란": "squalane",
    "호호바오일": "jojoba oil",
    "호호바씨오일": "jojoba seed oil",
    "아르간오일": "argan oil",
    "로즈힙오일": "rosehip oil",
    "시어버터": "shea butter",
    "코코넛오일": "coconut oil",
    "올리브오일": "olive oil",
    "마카다미아씨오일": "macadamia seed oil",
    "세테아릴알코올": "cetearyl alcohol",
    "세틸알코올": "cetyl alcohol",
    "베헤닐알코올": "behenyl alcohol",

    # ── 증점제/텍스처 ─────────────────────────────────────
    "카보머": "carbomer",
    "잔탄검": "xanthan gum",
    "하이드록시에칠셀룰로오스": "hydroxyethyl cellulose",
    "셀룰로오스검": "cellulose gum",

    # ── GT 트렌드 성분 (모델 핵심 피처) ─────────────────────
    "세라마이드NP": "ceramide np",
    "세라마이드 NP": "ceramide np",
    "바쿠치올": "bakuchiol",
    "아젤라익애씨드": "azelaic acid",
    "아젤라인산": "azelaic acid",
    "엑토인": "ectoin",
    "카페인": "caffeine",
    "엑소좀": "exosome",
    "엑소솜": "exosome",
    "PDRN": "pdrn",
    "폴리데옥시리보뉴클레오타이드": "pdrn",
    "NAD": "nad",
    "니코틴아마이드아데닌디뉴클레오타이드": "nad",
    "카프릴릭/카프릭트리글리세라이드": "caprylic/capric triglyceride",
    "카프릴릭카프릭트리글리세라이드": "caprylic/capric triglyceride",

    # ── 기타 트렌드 성분 ──────────────────────────────────
    "프로바이오틱스": "probiotics",
    "락토바실러스": "lactobacillus",
    "리코칼콘A": "licochalcone a",
    "레스베라트롤": "resveratrol",
    "박테리오신": "bacteriocin",
    "스핑고신": "sphingosine",
    "마이크로바이옴": "microbiome",
    "콜라겐": "collagen",
    "엘라스틴": "elastin",
    "히알루로니다제": "hyaluronidase",
}


def translate_ingredients(ing_names: list[str]) -> str:
    """
    한국어 성분명 리스트를 영어로 변환하여 쉼표 구분 문자열 반환.
    딕셔너리에 없는 성분은 원본 그대로 사용.
    """
    result = []
    for name in ing_names:
        name = name.strip()
        if not name:
            continue
        english = KOR_TO_ENG.get(name, name)
        result.append(english)
    return ", ".join(result)

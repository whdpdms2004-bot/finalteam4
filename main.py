# 엔드포인트에서 services 호출
# fastAPI 앱을 생성 --> brand, buyer 라우터 연결 

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import brand, buyer, influencer
from database import engine
from sqlalchemy import text

app = FastAPI(title="Beauty Fit Score API")

@app.on_event("startup")
def on_startup():
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS score FLOAT"))
        conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS sub_category VARCHAR(100)"))
        conn.commit()

# CORS 설정
# 프론트가 다른 도메인이나 포트에서 백엔드 API를 호출할 수 있도록 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 연결 
# 기능별로 라우터 분리 --> 핵심 비지니스 API는 router 폴더 안에 
app.include_router(brand.router)
app.include_router(buyer.router)
app.include_router(influencer.router)

@app.get("/")
def root():
    return {"message": "Beauty Fit Score API v2", "routers": ["brand", "buyer", "influencer"]}
# 서버가 정상 작동 중인지 확인 
@app.get("/health")
def health_check():
    return {"status": "ok"}

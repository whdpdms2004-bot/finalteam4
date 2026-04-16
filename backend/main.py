from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import brand, buyer, gate

app = FastAPI(title="Beauty Fit Score API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 연결
app.include_router(gate.router)
app.include_router(brand.router)
app.include_router(buyer.router)

@app.get("/")
def root():
    return {"message": "Beauty Fit Score API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
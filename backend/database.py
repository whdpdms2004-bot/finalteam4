# sqlalchemy : db와 python 연결해주는 라이브러리 
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 어느 db에 연결할지 문자열로 작성
DATABASE_URL = "postgresql://postgres.uulxpyapsfnymlqenehj:beautybridgeDB!!@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres?sslmode=require"

# 이 engine은 models.py에서 테이블 생성 
engine = create_engine(DATABASE_URL)
# 실제 쿼리를 날릴 때 쓸 세션 객체를 만들어줌 
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

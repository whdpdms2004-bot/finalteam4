# sqlalchemy : db와 python 연결해주는 라이브러리 
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 어느 db에 연결할지 문자열로 작성
DATABASE_URL = "oracle+oracledb://FINAL4:1111@192.168.0.183:1521/?service_name=ai4db"

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

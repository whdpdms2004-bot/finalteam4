from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Sequence
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from database import Base


class Product(Base):
    __tablename__ = "products"

    product_id        = Column(Integer, Sequence("products_seq"), primary_key=True)  # 시퀀스 이름 확인 후 수정
    brand_id          = Column(Integer, nullable=False)
    category_detail_id = Column(Integer, nullable=False)
    product_name      = Column(String(200), nullable=False)
    brand_name        = Column(String(100), nullable=False)
    price             = Column(Float, nullable=False)
    spf_index         = Column(Float, nullable=True)
    created_at        = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at        = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    ingredients = relationship("ProIng", back_populates="product", order_by="ProIng.seq_no")


class ProIng(Base):
    __tablename__ = "pro_ing"

    ing_id     = Column(Integer, Sequence("pro_ing_seq"), primary_key=True)  # 시퀀스 이름 확인 후 수정
    product_id = Column(Integer, ForeignKey("products.product_id"), nullable=False)
    ing_name   = Column(String(200), nullable=False)
    ing_kor    = Column(String(200), nullable=True)
    seq_no     = Column(Integer, nullable=False)

    product = relationship("Product", back_populates="ingredients")

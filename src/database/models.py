from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from src.database.database import Base

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    timestamp = Column(String, index=True) # ISO 8601 String untuk SQLite compatibility
    current_price = Column(Float)
    anomaly_score = Column(Float)
    p_value = Column(Float)
    regime = Column(String)
    signal = Column(String, nullable=True)
    confidence = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class TradeSetup(Base):
    __tablename__ = "trade_setups"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    timestamp = Column(String)
    signal = Column(String)
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit_1 = Column(Float)
    take_profit_2 = Column(Float)
    regime = Column(String)
    anomaly_score = Column(Float)
    confidence = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
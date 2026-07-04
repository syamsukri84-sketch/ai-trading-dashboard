from sqlalchemy.orm import Session
from src.database import models
from typing import List

def save_prediction(db: Session, prediction_data: dict) -> models.PredictionHistory:
    db_prediction = models.PredictionHistory(**prediction_data)
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def get_prediction_history(db: Session, ticker: str, limit: int = 50) -> List[models.PredictionHistory]:
    return db.query(models.PredictionHistory)\
             .filter(models.PredictionHistory.ticker == ticker)\
             .order_by(models.PredictionHistory.timestamp.desc())\
             .limit(limit).all()

def save_trade_setup(db: Session, setup_data: dict) -> models.TradeSetup:
    db_setup = models.TradeSetup(**setup_data)
    db.add(db_setup)
    db.commit()
    db.refresh(db_setup)
    return db_setup
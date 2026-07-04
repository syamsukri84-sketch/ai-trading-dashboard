import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.database import Base
from src.database import crud, models

# Gunakan SQLite in-memory untuk testing yang cepat dan terisolasi
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    # Buat skema tabel baru sebelum setiap fungsi test dijalankan
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Hapus skema tabel setelah test selesai agar bersih untuk test selanjutnya
        Base.metadata.drop_all(bind=engine)

def test_save_and_get_prediction(db_session):
    test_data = {
        "ticker": "BBCA",
        "timestamp": "2026-06-17 10:00:00",
        "current_price": 9500.0,
        "anomaly_score": 85.5,
        "p_value": 0.02,
        "regime": "VOLATILE",
        "signal": "LONG",
        "confidence": "0.98"
    }
    
    # Uji operasi Create (Simpan)
    saved_prediction = crud.save_prediction(db_session, test_data)
    assert saved_prediction.id is not None
    assert saved_prediction.ticker == "BBCA"
    assert saved_prediction.anomaly_score == 85.5
    
    # Uji operasi Read (Ambil)
    history = crud.get_prediction_history(db_session, "BBCA", limit=10)
    assert len(history) == 1
    assert history[0].ticker == "BBCA"
    assert history[0].current_price == 9500.0

def test_save_trade_setup(db_session):
    test_setup = {
        "ticker": "BBRI",
        "timestamp": "2026-06-17 11:00:00",
        "signal": "SHORT",
        "entry_price": 5000.0,
        "stop_loss": 5200.0,
        "take_profit_1": 4800.0,
        "take_profit_2": 4500.0,
        "regime": "CALM",
        "anomaly_score": 75.0,
        "confidence": "0.04"
    }
    
    # Uji operasi Create untuk Setup Trading
    saved_setup = crud.save_trade_setup(db_session, test_setup)
    assert saved_setup.id is not None
    assert saved_setup.ticker == "BBRI"
    assert saved_setup.signal == "SHORT"
    assert saved_setup.entry_price == 5000.0
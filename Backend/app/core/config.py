from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Mental Health Risk System"
    DEBUG: bool = True
    MODEL_PATH: str = "app/models/model.pkl"
    RISK_LOW_THRESHOLD: float = 0.3
    RISK_HIGH_THRESHOLD: float = 0.7

    class Config:
        env_file = ".env"


settings = Settings()

"""
Configuration management for the py-face application.
Handles loading and validation of environment variables and provides
a centralized configuration interface for the application.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables with validation.
    Provides type hints and default values for all configuration options.
    """
    # Application Settings
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Database Settings
    DB_TYPE: str = "sqlite"
    DB_HOST: Optional[str] = "localhost"
    DB_PORT: Optional[int] = None
    DB_NAME: str = "pyface"
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    
    # Storage Settings
    IMAGE_STORAGE_PATH: Path = Path("data/images")
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024
    
    # Face Recognition Settings
    MIN_FACE_CONFIDENCE: float = 0.6
    FACE_DETECTION_MODEL: str = "hog"
    
    @property
    def database_url(self) -> str:
        """
        Constructs database URL based on configuration.
        Returns:
            str: Database connection URL
        """
        if self.DB_TYPE == "sqlite":
            return f"sqlite:///{self.DB_NAME}.db"
        return (
            f"{self.DB_TYPE}://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Debug print to verify loaded values
print("\nLoaded configuration values:")
print(f"Database Type: {settings.DB_TYPE}")
print(f"Database URL: {settings.database_url}")
print(f"API Host: {settings.API_HOST}:{settings.API_PORT}")
print(f"Image Storage: {settings.IMAGE_STORAGE_PATH}")
print(f"Max Image Size: {settings.MAX_IMAGE_SIZE} bytes\n")

# Ensure image storage directory exists
settings.IMAGE_STORAGE_PATH.mkdir(parents=True, exist_ok=True) 
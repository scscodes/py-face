"""
Main entry point for the py-face application.
Initializes the FastAPI application and core components.
"""

import uvicorn
from fastapi import FastAPI
from config import settings

app = FastAPI(
    title="py-face",
    description="Facial Recognition API",
    version="0.1.0",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
)

@app.get("/")
async def root():
    """Root endpoint returning application status."""
    return {
        "status": "running",
        "environment": settings.APP_ENV,
        "api_version": "v1"
    }

def main():
    """
    Application entry point.
    Starts the uvicorn server with configured settings.
    """
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )

if __name__ == "__main__":
    main() 
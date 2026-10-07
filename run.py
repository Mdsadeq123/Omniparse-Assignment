import sys
import os

# Ensure the root directory is on the import path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import uvicorn
from app.main import app

if __name__ == "__main__":
    # Disable signal handlers and reload so it runs cleanly on Streamlit Cloud
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False
    )
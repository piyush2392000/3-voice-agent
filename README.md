# Voice + chat agent (deploy to Render)

## Required environment variables
GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY, ALPHA_VANTAGE_API_KEY, POSTGRES_URI,
GOOGLE_CLIENT_ID, SESSION_SECRET, ALLOWED_EMAILS (optional)

## Run locally
    pip install -r requirements.txt
    uvicorn server:app --reload --port 8000

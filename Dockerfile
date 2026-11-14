FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Only what the app needs at runtime (see .dockerignore)
COPY .streamlit ./.streamlit
COPY nba_careers ./nba_careers
COPY nba_data_app ./nba_data_app
COPY models ./models
COPY data/app ./data/app
COPY data/refined_raw_data/Player_Type_Modeling/player_3tier_label_with_target_all_years.csv \
     ./data/refined_raw_data/Player_Type_Modeling/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Railway (and most PaaS hosts) inject $PORT; default to 8501 locally.
ENV PORT=8501
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/_stcore/health')"

CMD ["sh", "-c", "streamlit run nba_data_app/app.py --server.port=${PORT} --server.address=0.0.0.0"]

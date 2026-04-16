FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=models/chatbot-fr-flan-t5-small-v2-convfix
ENV HISTORY_MODE=user-only
ENV HISTORY_TURNS=4
ENV MAX_INPUT_LENGTH=512
ENV MAX_NEW_TOKENS=72
ENV TEMPERATURE=0
ENV TOP_P=0.9
ENV REPETITION_PENALTY=1.1
ENV NO_REPEAT_NGRAM=3

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api_server_fr:app", "--host", "0.0.0.0", "--port", "8000"]

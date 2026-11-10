FROM python:3.11-slim

WORKDIR /srv
COPY requirements.txt .
RUN grep -v '^torch' requirements.txt > /tmp/reqs.txt \
    && pip install --no-cache-dir -r /tmp/reqs.txt \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY app ./app
COPY static ./static

ENV ORCH_FALLBACK=0
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

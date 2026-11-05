FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
RUN python -m scripts.generate_data && python -m scripts.train && python -m scripts.visualize && sh scripts/run_tests.sh
EXPOSE 5000
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "src.api"]

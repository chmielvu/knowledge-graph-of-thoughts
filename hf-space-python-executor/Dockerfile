FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r /app/requirements.txt

COPY --chown=user . /app

USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

CMD ["waitress-serve", "--host=0.0.0.0", "--port=7860", "python_executor:app"]

FROM python:3.13

WORKDIR /app

ENV OPENAI_API_KEY = 

COPY . /app

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

CMD ["python", "llm.py"]

FROM python:3.12

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

# For FastAPI
EXPOSE 8000

# Hosted on 0.0.0.0 on container -> localhost on local machine
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

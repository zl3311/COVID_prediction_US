FROM tensorflow/tensorflow:1.15.4-py3
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN ls
CMD ["python", "app.py"]

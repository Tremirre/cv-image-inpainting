FROM tensorflow/tensorflow

WORKDIR /app

RUN pip install flask
RUN pip install flask-cors

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]

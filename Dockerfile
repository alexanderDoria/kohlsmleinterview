FROM python:3.8 
ARG MODEL_LOCATION
ENV MODEL_LOCATION=$MODEL_LOCATION
WORKDIR /app 
COPY . /app 
RUN pip install -r requirements.txt 
EXPOSE 5000 
CMD ["python3","app.py"]
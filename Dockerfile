FROM r8.im/cog-base@sha256:534578e3091a1fac66d51a2faf02930b9c3480ea31cfeb6224340be67a0eba5e

RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "predict.py"]
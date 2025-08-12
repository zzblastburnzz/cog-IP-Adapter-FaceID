FROM r8.im/cog-base@sha256:534578e3091a1fac66d51a2faf02930b9c3480ea31cfeb6224340be67a0eba5e

# Install minimal system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Install Python dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy the rest
COPY . .

CMD ["python", "predict.py"]
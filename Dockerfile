FROM python:3.10-slim

# Cài thư viện hệ thống đầy đủ (thêm g++ và build-essential để compile insightface)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Cài Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy toàn bộ mã nguồn
COPY . .

# Lệnh mặc định nếu cần (có thể thay bằng CMD ["python", "test.py"])
CMD ["python", "predict.py"]

FROM python:3.10-slim

# Cài thêm các thư viện hệ thống cần thiết (có thể bổ sung dần nếu thiếu)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy và cài dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy toàn bộ code còn lại
COPY . .

# Chạy thử (có thể thay = entrypoint riêng)
CMD ["python", "predict.py"]

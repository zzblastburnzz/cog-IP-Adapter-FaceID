FROM r8.im/cog-base@sha256:534578e3091a1fac66d51a2faf02930b9c3480ea31cfeb6224340be67a0eba5e

# Install minimal dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to disable GUI
ENV DISPLAY=:0
ENV QT_QPA_PLATFORM=offscreen
ENV GDK_BACKEND=x11
ENV WPE_BACKEND=none

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "predict.py"]
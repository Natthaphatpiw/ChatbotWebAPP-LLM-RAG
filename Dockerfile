# ใช้ base image ของ Python
FROM python:3.10-slim

# ตั้ง working directory
WORKDIR /app

# คัดลอกไฟล์ทั้งหมดไปยัง container
COPY . .

# ติดตั้ง dependency
RUN pip install --no-cache-dir -r requirements.txt

# เปิดพอร์ต Flask (ปกติ 5000)
EXPOSE 5000

# เริ่มรันแอป Flask
CMD ["python", "app.py"]

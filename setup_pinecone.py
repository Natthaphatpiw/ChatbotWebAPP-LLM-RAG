import os
import pinecone
from dotenv import load_dotenv

# โหลด environment variables
load_dotenv()

def setup_pinecone_index():
    """สร้าง Pinecone index สำหรับโปรเจ็กต์ vetbot"""
    
    # ตั้งค่า Pinecone
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    
    if not PINECONE_API_KEY:
        print("❌ กรุณาตั้งค่า PINECONE_API_KEY ในไฟล์ .env")
        return
    
    # เชื่อมต่อ Pinecone (สำหรับ pinecone-client 3.x)
    pinecone.init(api_key=PINECONE_API_KEY)
    
    index_name = "vetbot"
    dimension = 384  # สำหรับ sentence-transformers/all-MiniLM-L6-v2
    
    # ตรวจสอบว่ามี index อยู่แล้วหรือไม่
    existing_indexes = pinecone.list_indexes()
    
    if index_name in existing_indexes:
        print(f"✅ Index '{index_name}' มีอยู่แล้ว")
        return
    
    # สร้าง index ใหม่
    try:
        print(f"🔄 กำลังสร้าง index '{index_name}'...")
        
        pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine"
        )
        
        print(f"✅ สร้าง index '{index_name}' เรียบร้อยแล้ว!")
        print("📝 หมายเหตุ: ใช้เวลาสักครู่ก่อนที่ index จะพร้อมใช้งาน")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")

if __name__ == "__main__":
    setup_pinecone_index() 
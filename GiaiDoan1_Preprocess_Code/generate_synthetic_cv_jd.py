import os
import csv
import json
import time
import pandas as pd
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    print("\n[LỖI THƯ VIỆN] Vui lòng mở Terminal chạy lệnh sau trước khi tiếp tục:\npip install openai pandas tqdm\n")
    exit(1)

# ==========================================
# CẤU HÌNH API SIÊU TỐC ĐỘ (GROQ)
# ==========================================
# 1. Truy cập https://console.groq.com/keys (Đăng nhập bằng Gmail/Github)
# 2. Bấm "Create API Key"
# 3. Copy key (bắt đầu bằng gsk_...) dán vào bên dưới:

API_KEY = os.environ.get("GROQ_API_KEY", "") # SỬ DỤNG BIẾN MÔI TRƯỜNG ĐỂ BẢO MẬT API KEY KHI ĐẨY LÊN GITHUB

# Dùng thư viện OpenAI nhưng gọi thẳng sang siêu máy chủ Groq
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# Đường dẫn file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JD_FILE = os.path.join(BASE_DIR, "JobsDataset.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "synthetic_gold_dataset.csv")

# Số lượng JD muốn dùng để sinh dữ liệu (VD: 300 JD -> 900 cặp CV-JD)
NUM_JDS_TO_PROCESS = 300 

# ==========================================
# PROMPT KỸ THUẬT CAO (Hard Negative Mining)
# ==========================================
SYSTEM_PROMPT = """You are an Expert IT Recruiter and AI Data Engineer building an ATS dataset.
Given a Job Description (JD), your task is to generate 3 highly realistic candidate CV profiles in the SAME language as the JD (Vietnamese or English).

Generate exactly 3 CV structures:
1. "positive_cv": A perfectly matching CV. Do NOT just copy the JD exactly. Use realistic synonyms, rephrase technical terms, and include plausible job histories.
2. "medium_cv": A partially matching CV (50%-60%). Has some relevant skills but critically lacks 2-3 mandatory core technologies mentioned in the JD.
3. "hard_negative_cv": A CV that looks superficially similar (e.g., both use Python) but fundamentally FAILS the JD (e.g., a Data Analyst applying for a DevOps Engineer role).

Return ONLY a valid JSON object matching exactly this schema, without any markdown formatting around it:
{
    "positive_cv": "string - text content of CV here",
    "medium_cv": "string - text content of CV here",
    "hard_negative_cv": "string - text content of CV here"
}
"""

def generate_synthetic_cvs(jd_text):
    """Gọi API Groq Llama-3.3 để sinh ra 3 phiên bản CV JSON cực nhanh"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Model Khủng nhất hiện nay của Meta, chạy trên LPU nhanh gấp 4 lần GPT-4
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Create 3 CVs for this JD:\n\n{jd_text}"}
            ],
            response_format={"type": "json_object"}, # Ép Groq xuất output chuẩn JSON 100%
            temperature=0.7
        )
        
        result_str = response.choices[0].message.content
        return json.loads(result_str.strip())
    
    except Exception as e:
        print(f"Lỗi khi gọi API: {e}")
        return None

def main():
    print(f"🚀 BẮT ĐẦU QUÁ TRÌNH SINH DỮ LIỆU BẰNG GROQ LLAMA 3.3 KHỔNG LỒ...")
    
    if not os.path.exists(INPUT_JD_FILE):
        print(f"❌ Không tìm thấy {INPUT_JD_FILE}!")
        return

    df_jobs = pd.read_csv(INPUT_JD_FILE)
    jds_to_process = df_jobs['Query'].dropna().sample(n=min(NUM_JDS_TO_PROCESS, len(df_jobs)), random_state=42).tolist()
    
    # Chuẩn bị file output
    file_exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['CV_Text', 'JD_Text', 'Match_Label', 'Score']) 
            
        print(f"Đang sinh {len(jds_to_process) * 3} cặp CV-JD từ {len(jds_to_process)} JDs...")

        # Vòng lặp
        for jd in tqdm(jds_to_process, desc="Generating CVs"):
            # Groq trả lời siêu tốc, setup delay giới hạn cho 30 Requests Per Minute của Free Tier
            time.sleep(3.1) 
            
            result_json = generate_synthetic_cvs(jd)
            if result_json:
                if "positive_cv" in result_json:
                    writer.writerow([result_json["positive_cv"], jd, "Positive", 1.0])
                if "medium_cv" in result_json:
                    writer.writerow([result_json["medium_cv"], jd, "Medium", 0.5])
                if "hard_negative_cv" in result_json:
                    writer.writerow([result_json["hard_negative_cv"], jd, "Hard_Negative", 0.0])
                    
    print("\n✅ HOÀN TẤT SINH DỮ LIỆU BẰNG GROQ!")
    print(f"File được lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

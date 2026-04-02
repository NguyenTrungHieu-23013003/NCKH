import pandas as pd
import random
import os

# Đường dẫn file
DATA_DIR = "/home/hieu/Downloads/CV_MATCHING-main/GiaiDoan1_Preprocess_Code"
JOBS_FILE = os.path.join(DATA_DIR, "JobsDataset.csv")
RESUME_FILE = os.path.join(DATA_DIR, "Resume.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "expert_evaluation_form.csv")

def create_evaluation_dataset():
    print("⏳ Đang đọc dữ liệu gốc...")
    jobs = pd.read_csv(JOBS_FILE)
    resumes = pd.read_csv(RESUME_FILE)

    # Do 2 tập dữ liệu có cách đặt tên ngành khác nhau, ta mapping thủ công các ngành khớp nhau
    # JD_Category (Query) vs CV_Category
    mapping = {
        'Data Scientist': 'INFORMATION-TECHNOLOGY',
        'Machine Learning': 'INFORMATION-TECHNOLOGY',
        'Data Analyst': 'INFORMATION-TECHNOLOGY',
        'Business Analyst': 'BUSINESS-DEVELOPMENT',
        'Data Engineer': 'ENGINEERING',
        'Database Administrator': 'INFORMATION-TECHNOLOGY',
        'IT Consultant': 'CONSULTANT',
        'Business Intelligence Analyst': 'FINANCE',
        'AI Consultant': 'INFORMATION-TECHNOLOGY',
        'Statistics': 'ENGINEERING'
    }
    
    evaluation_pairs = []

    print(f"📊 Bắt đầu lấy mẫu dữ liệu...")

    # 1. Cùng ngành (25 cặp)
    print("   -> Lấy 25 cặp Cùng ngành (Match)...")
    count = 0
    available_jd_cats = [c for c in mapping.keys() if c in jobs['Query'].unique()]
    
    for _ in range(25):
        if not available_jd_cats:
            # Fallback nếu không khớp mapping nào
            j = jobs.sample(1).iloc[0]
            c = resumes.sample(1).iloc[0]
            cat_label = "Random (Unmatched Categories)"
            type_label = "Mismatch (Random)"
        else:
            jd_cat = random.choice(available_jd_cats)
            cv_cat = mapping[jd_cat]
            
            # Kiểm tra xem có CV nào thuộc ngành này không
            cv_subset = resumes[resumes['Category'] == cv_cat]
            if cv_subset.empty: continue
            
            j = jobs[jobs['Query'] == jd_cat].sample(1).iloc[0]
            c = cv_subset.sample(1).iloc[0]
            cat_label = f"Match: {jd_cat} vs {cv_cat}"
            type_label = "Match (Same Category)"
        
        evaluation_pairs.append({
            "JD_ID": j['ID'],
            "CV_ID": c['ID'],
            "Category": cat_label,
            "Type": type_label,
            "JD_Text": j['Description'], # Giữ full text để xử lý TF-IDF/Semantic
            "CV_Text": str(c['Resume_str']),
            "Expert_Score_0_10": "", # Để trống cho chuyên gia điền
        })
        count += 1

    # 2. Khác ngành (25 cặp) - Trộn lẫn để Expert không đoán được
    print("   -> Lấy 25 cặp Khác ngành (Mismatch)...")
    for _ in range(25):
        j = jobs.sample(1).iloc[0]
        c = resumes.sample(1).iloc[0]
        
        # Đảm bảo khác ngành (Check qua mapping)
        jd_cat = j['Query']
        cv_cat = c['Category']
        
        type_label = "Different Category (Mismatch)"
        if jd_cat in mapping and mapping[jd_cat] == cv_cat:
            type_label = "Same Category (Accidental Match)"

        evaluation_pairs.append({
            "JD_ID": j['ID'],
            "CV_ID": c['ID'],
            "Category": f"{jd_cat} vs {cv_cat}",
            "Type": type_label,
            "JD_Text": j['Description'],
            "CV_Text": str(c['Resume_str']),
            "Expert_Score_0_10": "",
        })

    # Lưu kết quả
    df_eval = pd.DataFrame(evaluation_pairs)
    df_eval.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"✅ Đã tạo file: {OUTPUT_FILE}")
    print(f"👉 Tổng số cặp: {len(df_eval)}")
    print("Lưu ý: Header JD_Text và CV_Text đang chứa nội dung gốc để AI analyze.")

if __name__ == "__main__":
    create_evaluation_dataset()

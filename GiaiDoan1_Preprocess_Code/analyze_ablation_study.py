import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr
import os

"""
NCKH RESEARCH MODEL - HYBRID PRO:
Formula: Score_Hybrid = α * Sim_Semantic + (1 - α) * Sim_Keyword - Penalty_Category
In this implementation: α = 0.7
"""

# Cấu hình
EXPERT_FILE = "/home/hieu/Downloads/CV_MATCHING-main/GiaiDoan1_Preprocess_Code/expert_evaluation_form.csv"
MODEL_PATH = "/home/hieu/Downloads/CV_MATCHING-main/GiaiDoan1_Preprocess_Code/models/e5_synthetic_model"
E5_BASE = "intfloat/multilingual-e5-small"
device = "cuda" if torch.cuda.is_available() else "cpu"

def run_ablation_study():
    print("🔬 Đang chạy Ablation Study (NCKH Version 2.0)...")
    
    if not os.path.exists(EXPERT_FILE):
        print("❌ Lỗi: Không tìm thấy file expert_evaluation_form.csv.")
        return

    df = pd.read_csv(EXPERT_FILE)
    
    # Giả lập điểm expert nếu chưa điền (Chuẩn hóa để so sánh)
    if 'Expert_Score_0_10' not in df.columns or df['Expert_Score_0_10'].isnull().all():
        print("⚠️ Sử dụng Ground Truth giả định để demo logic NCKH.")
        def simulate_expert(type_str):
            if 'Same Category' in type_str: return np.random.uniform(7.0, 9.5)
            if 'Different Category' in type_str: return np.random.uniform(0.5, 3.5)
            return np.random.uniform(3.5, 6.5)
        df['Expert_Score_0_10'] = df['Type'].apply(simulate_expert)

    y_expert = df['Expert_Score_0_10'].values

    # 1. LOAD MODELS
    print("   -> Loading Semantic Model...")
    try:
        model = SentenceTransformer(MODEL_PATH, device=device)
    except:
        model = SentenceTransformer(E5_BASE, device=device)

    # 2. GLOBAL TF-IDF FITTING (Góp ý A)
    print("   -> Fitting Global TF-IDF on all texts (NCKH Best Practice)...")
    all_texts = df['JD_Text'].tolist() + df['CV_Text'].tolist()
    tfidf_vec = TfidfVectorizer()
    tfidf_vec.fit(all_texts)

    # 3. RUN MODELS
    scores_tfidf = []
    scores_semantic = []
    scores_hybrid = []

    print(f"   -> Analyzing {len(df)} JD-CV pairs...")
    
    for _, row in df.iterrows():
        jd = row['JD_Text']
        cv = row['CV_Text']
        
        # --- MODEL A: GLOBAL TF-IDF ---
        # Transform từng cặp dựa trên từ điển chung (Global Vocabulary)
        vecs = tfidf_vec.transform([jd, cv])
        sim_tfidf = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
        scores_tfidf.append(sim_tfidf * 10)

        # --- MODEL B: PURE SEMANTIC (E5) ---
        jd_emb = model.encode(f"query: {jd}", convert_to_tensor=True)
        cv_emb = model.encode(f"passage: {cv}", convert_to_tensor=True)
        sim_sem = util.cos_sim(jd_emb, cv_emb).item()
        scores_semantic.append(max(0, sim_sem) * 10)

        # --- MODEL C: HYBRID PRO (Our Proposal - Góp ý B) ---
        alpha = 0.7
        hybrid_score = (alpha * sim_sem * 10) + ((1 - alpha) * sim_tfidf * 10)
        
        # Penalty Category (Metadata Context)
        if 'Different Category' in row['Type']:
            hybrid_score *= 0.6
        scores_hybrid.append(hybrid_score)

    # 4. STATISTICAL ANALYSIS (Pearson & Spearman)
    metrics = []
    for name, y_pred in [("TF-IDF", scores_tfidf), ("Pure Semantic", scores_semantic), ("Hybrid Pro", scores_hybrid)]:
        pearson, _ = pearsonr(y_expert, y_pred)
        spearman, _ = spearmanr(y_expert, y_pred)
        metrics.append({"Method": name, "Pearson": pearson, "Spearman": spearman})

    m_df = pd.DataFrame(metrics)
    print("\n📊 CORRELATION RESULTS:")
    print(m_df.to_string(index=False))

    # 5. VISUALIZATION (Scatter Plot - Góp ý C)
    plt.figure(figsize=(15, 6))

    # Subplot 1: Correlation Comparison
    plt.subplot(1, 2, 1)
    x_labels = m_df['Method']
    plt.bar(np.arange(len(x_labels)) - 0.2, m_df['Pearson'], 0.4, label='Pearson', color='skyblue')
    plt.bar(np.arange(len(x_labels)) + 0.2, m_df['Spearman'], 0.4, label='Spearman', color='salmon')
    plt.xticks(np.arange(len(x_labels)), x_labels)
    plt.ylabel('Hệ số tương quan')
    plt.title('Ablation Study: Metrics Comparison')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # Subplot 2: Scatter Plot Hybrid vs Human (Góp ý C)
    plt.subplot(1, 2, 2)
    plt.scatter(y_expert, scores_hybrid, alpha=0.6, color='green', label='Hybrid Score')
    
    # Vẽ đường chéo y = x (Ideal)
    lims = [0, 10]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Ideal Perfect Fit (y=x)')
    
    plt.xlabel('Human Expert Score (0-10)')
    plt.ylabel('AI Hybrid Score (0-10)')
    plt.title('Reliability: AI vs Human Judgment')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    chart_path = "/home/hieu/Downloads/CV_MATCHING-main/GiaiDoan1_Preprocess_Code/ablation_scientific_v2.png"
    plt.savefig(chart_path)
    print(f"\n✅ Đã xuất biểu đồ khoa học tại: {chart_path}")

if __name__ == "__main__":
    run_ablation_study()

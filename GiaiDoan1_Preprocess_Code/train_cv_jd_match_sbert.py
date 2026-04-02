import os
import random
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(SEED)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_dir, "synthetic_gold_dataset.csv") # DÙNG FILE DỮ LIỆU MỚI SINH!
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"[1/4] Đang nạp dữ liệu từ {data_file}...")
    df = pd.read_csv(data_file)
    
    # ========================================================
    # KỸ THUẬT HARD NEGATIVE MINING (Bí kíp chống Overfitting)
    # ========================================================
    # Điểm yếu của code cũ: Chỉ dạy AI nhìn nhận CV đúng (Positive).
    # Code mới: Dạy AI nhìn thấy CV Đúng và CV Cố Tình Làm Sai (Hard Negative).
    # Gom nhóm theo JD_Text:
    grouped = df.groupby('JD_Text')
    triplets = []
    
    for jd_text, group in grouped:
        pos_row = group[group['Match_Label'] == 'Positive']
        neg_row = group[group['Match_Label'] == 'Hard_Negative']
        
        if not pos_row.empty and not neg_row.empty:
            pos_cv = pos_row.iloc[0]['CV_Text']
            neg_cv = neg_row.iloc[0]['CV_Text']
            
            # ĐA NGÔN NGỮ E5 BASE BẮT BUỘC PHẢI CÓ TIỀN TỐ NÀY ĐỂ HOẠT ĐỘNG:
            # Query (Tìm kiếm) -> là JD
            # Passage (Văn bản) -> là CV
            jd_formatted = f"query: {jd_text}"
            pos_formatted = f"passage: {pos_cv}"
            neg_formatted = f"passage: {neg_cv}"
            
            # Cấu trúc 3 chân: [Anchor, Positive, Negative]
            triplets.append(InputExample(texts=[jd_formatted, pos_formatted, neg_formatted]))

    print(f"✅ Đã tạo thành công {len(triplets)} cụm Triplets (Anchor, Positive, Hard Negative)!")
    
    # Phân chia Train / Validation
    n = len(triplets)
    random.shuffle(triplets)
    split_idx = int(0.85 * n) # 85% train
    train_examples, val_examples = triplets[:split_idx], triplets[split_idx:]
    
    # Đẩy vào DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2) # Đã hạ từ 4 xuống 2 để giảm tốn RAM
    
    print("[2/4] Thiết lập mô-đun Đánh giá hệ thống IR (MRR, NDCG)...")
    queries, corpus, relevant_docs = {}, {}, {}
    
    for i, example in enumerate(val_examples):
        q_id, c_id = f"jd_{i}", f"cv_{i}"
        queries[q_id] = example.texts[0] # JD
        corpus[c_id] = example.texts[1] # Chỉ lấy Positive CV nhét vào kho Evaluation
        relevant_docs[q_id] = {c_id}
        
    evaluator = InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs,
        name="CV-JD-E5-Synthetic-Eval", mrr_at_k=[1, 5, 10], ndcg_at_k=[5, 10], accuracy_at_k=[1, 5, 10], 
        show_progress_bar=True
    )

    # NÂNG CẤP KIẾN TRÚC MÔ HÌNH (SOTA)
    print("\n[3/4] Khởi tạo Mô hình Lõi: [intfloat/multilingual-e5-small]")
    base_model_name = 'intfloat/multilingual-e5-small'
    model = SentenceTransformer(base_model_name)
    
    # --- BENCHMARK: ĐÁNH GIÁ MÔ HÌNH CƠ SỞ (BASE) TRƯỚC KHI TRAIN ---
    print("\n📊 Đang Benchmark mô hình GỐC (Base Model)...")
    base_results = evaluator(model)
    print(f"Base MRR@10: {base_results.get('CV-JD-E5-Synthetic-Eval_cos_sim_mrr@10', 0):.4f}")
    
    # Loss function for Triplet Training
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    num_epochs = 5 # Giảm xuống 5 epoch nếu data lớn để tránh overfitting
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) 
    
    print("\n[4/4] BẮT ĐẦU FINE-TUNING VỚI HARD NEGATIVE MINING...")
    out_path = os.path.join(model_dir, "e5_synthetic_model")
    os.makedirs(out_path, exist_ok=True)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=50,
        warmup_steps=warmup_steps,
        output_path=out_path,
        save_best_model=True,
        show_progress_bar=True
    )

    # --- FINAL BENCHMARK: ĐÁNH GIÁ MÔ HÌNH ĐÃ FINE-TUNED ---
    print("\n🏆 Đang Benchmark mô hình ĐÃ NÂNG CẤP (Fine-tuned Model)...")
    best_model = SentenceTransformer(out_path)
    ft_results = evaluator(best_model)
    
    # DEBUG: In ra các chìa khóa để kiểm tra nếu bị 0.0000
    print("\n[DEBUG] Các chỉ số đo lường tìm thấy:", ft_results.keys())

    # Hàm hỗ trợ lấy điểm linh hoạt
    def get_score(res, metric):
        # Tìm key nào chứa chuỗi metric (VD: 'mrr@10')
        for k, v in res.items():
            # Chấp nhận cả 'cosine' hoặc 'cos_sim'
            if metric in k.lower() and ('cos_sim' in k.lower() or 'cosine' in k.lower()):
                try:
                    return float(v)
                except:
                    return 0.0
        return 0.0

    print("\n" + "="*60)
    print("🚀 KẾT QUẢ SO SÁNH (NCKH PROOF)")
    print("="*60)
    print(f"Metric\t\t\tBase Model\tFine-tuned")
    
    mrr_base = get_score(base_results, 'mrr@10')
    mrr_ft = get_score(ft_results, 'mrr@10')
    ndcg_base = get_score(base_results, 'ndcg@10')
    ndcg_ft = get_score(ft_results, 'ndcg@10')
    acc_base = get_score(base_results, 'accuracy@1')
    acc_ft = get_score(ft_results, 'accuracy@1')

    print(f"MRR@10\t\t\t{mrr_base:.4f}\t\t{mrr_ft:.4f}")
    print(f"NDCG@10\t\t\t{ndcg_base:.4f}\t\t{ndcg_ft:.4f}")
    print(f"Accuracy@1\t\t{acc_base:.4f}\t\t{acc_ft:.4f}")
    print("="*60)
    
    if mrr_ft > mrr_base:
        print(f"✅ ĐÃ CHỨNG MINH: Fine-tuning đã giúp cải thiện {((mrr_ft-mrr_base)/max(0.001, mrr_base)*100):.1f}% hiệu năng.")
    else:
        print("⚠️ Cảnh báo: Hiệu năng chưa cải thiện rõ rệt. Cần thêm dữ liệu!")
        
    print(f"Lưu tại: {out_path}")
    print("="*60)

if __name__ == "__main__":
    main()

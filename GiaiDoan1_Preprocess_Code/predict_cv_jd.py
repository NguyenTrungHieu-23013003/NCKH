import os
import torch
from sentence_transformers import SentenceTransformer, util

# Configuration
MODEL_PATH = "/home/hieu/Downloads/CV_MATCHING-main/GiaiDoan1_Preprocess_Code/models/sbert_gold_model"

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}. Using pre-trained baseline.")
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return SentenceTransformer(MODEL_PATH)

def match(jd_text, cv_texts):
    model = load_model()
    # Compute embeddings
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    cv_embs = model.encode(cv_texts, convert_to_tensor=True)
    
    # Compute cosine similarity
    scores = util.cos_sim(jd_emb, cv_embs)[0]
    
    # Sort
    results = []
    for i, score in enumerate(scores):
        results.append({
            'index': i,
            'score': score.item(),
            'text_preview': cv_texts[i][:150] + "..."
        })
    
    # Return sorted results
    return sorted(results, key=lambda x: x['score'], reverse=True)

if __name__ == "__main__":
    # Example usage
    sample_jd = "We are looking for a Senior Data Scientist with experience in Python and Transformers."
    sample_cvs = [
        "Experienced Data Scientist skilled in NLP, BERT, and PyTorch.",
        "Sales manager with 10 years experience in real estate and retail.",
        "HR professional specializing in recruitment and payroll management."
    ]
    
    print(f"🔍 Matching JD: {sample_jd}")
    matches = match(sample_jd, sample_cvs)
    
    for m in matches:
        print(f"Score: {m['score']:.4f} | Preview: {m['text_preview']}")

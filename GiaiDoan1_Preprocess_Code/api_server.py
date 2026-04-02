import re
import io
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from typing import List, Set, Dict, Optional

# =========================================================
# 1. CẤU HÌNH & LOAD MODEL (OPTIMIZED)
# =========================================================
app = FastAPI(title="AI CV Matcher Pro", version="10.0")

# THÊM CORS ĐỂ FRONTEND GỌI ĐƯỢC API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép tất cả (localhost:3000)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "/home/hieu/Downloads/CV_MATCHING-main/GiaiDoan1_Preprocess_Code/models/e5_synthetic_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = SentenceTransformer(MODEL_PATH, device=device)
    print(f"✅ Loaded fine-tuned model on {device}")
except:
    model = SentenceTransformer("intfloat/multilingual-e5-small", device=device)
    print(f"✅ Fallback to E5-Small on {device}")

# =========================================================
# 2. DICTIONARY & LOGIC (GIỮ LẠI BẢN CŨ & TỐI ƯU)
# =========================================================
TECH_ALIASES = {
    "python": ["python", "py"], "java": ["java"], "javascript": ["javascript", "js", "jscript"],
    "typescript": ["typescript", "ts"], "react": ["react", "reactjs", "react.js"],
    "nodejs": ["nodejs", "node.js", "node"], "sql": ["sql", "mysql", "postgresql", "postgres"],
    "docker": ["docker", "containerization"], "aws": ["aws", "amazon web services"],
    "machine learning": ["machine learning", "ml", "ai"], "c#": ["c#", "csharp"],
    ".net": [".net", "dotnet", "asp.net"], "php": ["php", "laravel"]
}

CRITICAL_SKILLS = {"python", "java", "javascript", "react", "sql", "aws", "docker"}

# =========================================================
# 3. UTILS & PREPROCESSING
# =========================================================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text) # Xóa xuống dòng, khoảng trắng thừa
    # Giữ lại các ký tự đặc biệt của ngành IT
    text = re.sub(r"[^a-z0-9\+\#\.\s]", " ", text)
    return text.strip()

def extract_keywords(text: str) -> Set[str]:
    text_lower = text.lower()
    # PREVENT FALSE POSITIVES: Ignore keywords inside negative contexts
    text_lower = re.sub(r"(chưa có|không có|không biết|chưa từng|điểm yếu|weakness)[^.]*", "", text_lower)
    
    found = set()
    cleaned = clean_text(text_lower)
    for canonical, aliases in TECH_ALIASES.items():
        for alias in aliases:
            # Regex boundary để tránh match nhầm 'java' trong 'javascript'
            pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
            if re.search(pattern, cleaned):
                found.add(canonical)
                break
    return found

# =========================================================
# 4. LONG TEXT HANDLING (CHIA ĐOẠN ĐỂ AI KHÔNG BỎ SÓT)
# =========================================================
def get_embedding(text: str, prefix: str = "passage: "):
    """Chia văn bản dài thành các đoạn nhỏ và lấy trung bình vector"""
    max_length = 400 # Token limit an toàn cho E5
    words = text.split()
    chunks = [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    
    if not chunks: return None
    
    # Thêm prefix cho mỗi chunk
    prefixed_chunks = [f"{prefix}{chunk}" for chunk in chunks]
    chunk_embeddings = model.encode(prefixed_chunks, convert_to_tensor=True, normalize_embeddings=True)
    
    # Trả về vector trung bình của các đoạn
    return torch.mean(chunk_embeddings, dim=0, keepdim=True)

# =========================================================
# 5. CORE MATCHING LOGIC
# =========================================================
class MatchRequest(BaseModel):
    jd: str
    cv: str

class BatchCV(BaseModel):
    id: str # ID hoặc tên file
    text: str

class BatchMatchRequest(BaseModel):
    jd: str
    cvs: List[BatchCV]

@app.post("/api/batch-match")
async def batch_match_cv_jd(data: BatchMatchRequest):
    """Xử lý HÀNG LOẠT CV bằng Batch Encoding (Tối ưu NCKH)"""
    try:
        jd_raw = data.jd
        if not jd_raw or not data.cvs:
            raise HTTPException(status_code=400, detail="Thiếu dữ liệu")

        # 1. Tiền xử lý & Trích xuất Keyword 1 lần cho JD
        jd_clean = clean_text(jd_raw)
        jd_keys = extract_keywords(jd_raw)
        jd_emb = get_embedding(jd_clean, prefix="query: ") # (1, 384)

        # 2. Chuẩn bị Batch dữ liệu CV
        cv_texts = []
        cv_metadatas = []
        for cv_item in data.cvs:
            cv_clean = clean_text(cv_item.text)
            cv_texts.append(f"passage: {cv_clean[:2000]}") # Tránh quá dài
            cv_metadatas.append({
                "id": cv_item.id,
                "keys": extract_keywords(cv_item.text)
            })

        # 3. BATCH ENCODING (Đây là phần tối ưu chính)
        # Thay vì loop, ta đẩy cả List vào model
        print(f"⚡ Batch Processing {len(cv_texts)} CVs...")
        all_cv_embs = model.encode(cv_texts, convert_to_tensor=True, normalize_embeddings=True) # (N, 384)

        # 4. Tính toán Similarity hàng loạt bằng Ma trận
        # Cosine Similarity của vector đã normalize = Dot Product
        # jd_emb: (1, 384), all_cv_embs: (N, 384) -> Kết quả: (1, N)
        cos_sims = util.cos_sim(jd_emb, all_cv_embs)[0] 

        results = []
        for i, sim_score in enumerate(cos_sims):
            raw_sim = sim_score.item()
            cv_meta = cv_metadatas[i]
            
            # Tính điểm Keyword
            matched = jd_keys.intersection(cv_meta['keys'])
            missing = jd_keys.difference(cv_meta['keys'])
            
            # Chuẩn hóa Semantic Calibration (0.75 - 0.95 -> 0-100 cho model Fine-tuned)
            norm_semantic = max(0.0, min(1.0, (raw_sim - 0.75) / 0.20))
            kw_score = len(matched) / len(jd_keys) if jd_keys else norm_semantic
            
            final_score = (norm_semantic * 0.70 + kw_score * 0.30) * 100
            
            critical_missing = CRITICAL_SKILLS.intersection(missing)
            final_score -= (len(critical_missing) * 8)
            if len(jd_keys) > 0 and len(matched) == 0:
                final_score *= 0.5
            
            final_score = max(5.0, min(99.5, round(final_score, 1)))

            results.append({
                "id": cv_meta['id'],
                "score": final_score,
                "status": "EXCELLENT" if final_score >= 75 else "POTENTIAL" if final_score >= 60 else "CONSIDER" if final_score >= 40 else "REJECTED",
                "matched_count": len(matched),
                "semantic_sim": round(raw_sim, 4)
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return {"success": True, "count": len(results), "leaderboard": results}

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

@app.post("/api/match")
async def match_cv_jd(data: MatchRequest):
    try:
        jd_raw, cv_raw = data.jd, data.cv
        if not jd_raw or not cv_raw:
            raise HTTPException(status_code=400, detail="Dữ liệu không được để trống")

        # 1. Trích xuất Keywords (Thực tế)
        jd_keys = extract_keywords(jd_raw)
        cv_keys = extract_keywords(cv_raw)
        matched = jd_keys.intersection(cv_keys)
        missing = jd_keys.difference(cv_keys)

        # 2. Tính Semantic Similarity (AI)
        jd_emb = get_embedding(clean_text(jd_raw), prefix="query: ")
        cv_emb = get_embedding(clean_text(cv_raw), prefix="passage: ")
        
        raw_sim = util.cos_sim(jd_emb, cv_emb).item()

        # 3. Chuẩn hóa điểm Semantic Calibration (0.75 - 0.95 -> 0 - 100)
        norm_semantic = (raw_sim - 0.75) / 0.20
        norm_semantic = max(0.0, min(1.0, norm_semantic))

        # 4. Tính điểm Keyword
        kw_score = len(matched) / len(jd_keys) if jd_keys else norm_semantic

        # 5. Công thức Hybrid (AI 70% + Keyword 30%)
        final_score = (norm_semantic * 0.70 + kw_score * 0.30) * 100

        # 6. Penalty & Logic bổ trợ
        critical_missing = CRITICAL_SKILLS.intersection(missing)
        final_score -= (len(critical_missing) * 8) # Trừ 8 điểm/skill quan trọng

        if len(jd_keys) > 0 and len(matched) == 0:
            final_score *= 0.5 # Phạt nặng nếu không trùng từ khóa nào

        final_score = max(5.0, min(99.5, round(final_score, 1)))

        # 7. Phân loại ứng viên
        status = "REJECTED"
        if final_score >= 75: status = "EXCELLENT"
        elif final_score >= 60: status = "POTENTIAL"
        elif final_score >= 40: status = "CONSIDER"

        # 8. Xây dựng bản nhận xét chi tiết (Detailed Critique)
        critique = []

        # Dùng điểm đã chuẩn hóa (norm_semantic) thay vì raw_sim
        # để tránh mâu thuẫn: "tương đồng 83% nhưng điểm chỉ có 31%"
        sim_percent = norm_semantic * 100

        if final_score >= 85:
            critique.append(f"🌟 **Ứng viên xuất sắc.** Điểm tổng hợp đạt **{final_score:.1f}%** (Độ khớp AI ngữ nghĩa: {sim_percent:.1f}%).")

            if not missing:
                critique.append("Hồ sơ hoàn hảo, đáp ứng đầy đủ mọi yêu cầu kỹ thuật.")
            else:
                critique.append(f"Khớp {len(matched)}/{len(jd_keys)} kỹ năng: {', '.join(sorted(matched))}.")
                critique.append(f"Lưu ý nhỏ: Cần kiểm tra thêm về {', '.join(sorted(missing))}.")

        elif final_score >= 60:
            critique.append(f"✅ **Ứng viên tiềm năng.** Điểm tổng hợp đạt **{final_score:.1f}%** (Độ khớp AI ngữ nghĩa: {sim_percent:.1f}%).")

            if matched:
                critique.append(f"Điểm mạnh: Sở hữu các kỹ năng quan trọng như {', '.join(sorted(matched))}.")

            if critical_missing:
                critique.append(f"⚠️ **Rào cản:** Thiếu một số công nghệ lõi: {', '.join(sorted(critical_missing))}.")
            elif missing:
                critique.append(f"Cần bồi dưỡng thêm về: {', '.join(sorted(missing))}.")

        elif final_score >= 40:
            critique.append(f"🧐 **Ứng viên cần cân nhắc thêm.** Điểm tổng hợp đạt **{final_score:.1f}%** (Độ khớp AI ngữ nghĩa: {sim_percent:.1f}%).")

            if matched:
                critique.append(f"Có nền tảng về {', '.join(sorted(matched))}, nhưng chưa đủ bao quát JD.")

            if critical_missing:
                critique.append(f"🚩 **Cảnh báo:** Thiếu hụt nghiêm trọng các kỹ năng cốt lõi: {', '.join(sorted(critical_missing))}.")

        else:
            critique.append(f"❌ **Chưa phù hợp.** Trọng tâm hồ sơ hoàn toàn lệch so với yêu cầu vị trí.")

            if critical_missing:
                critique.append(f"Hồ sơ thiếu các kỹ năng bắt buộc: {', '.join(sorted(critical_missing))}.")

            critique.append("Khuyến dùng: Tìm kiếm các ứng viên có nền tảng công nghệ sát thực tế hơn hoặc xem xét ứng viên này cho các vị trí hỗ trợ/thực tập.")

        full_explanation = " ".join(critique)


        return {
            "score": final_score,
            "status": status,
            "analysis": {
                "semantic_sim": round(raw_sim, 4),
                "matched_skills": sorted(list(matched)),
                "missing_skills": sorted(list(missing)),
                "critical_missing": sorted(list(critical_missing))
            },
            "explanation": full_explanation
        }

    except Exception as e:
        return {"error": str(e)}


# =========================================================
# 6. PDF / DOCX FILE PARSING ENDPOINT
# =========================================================
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Trích xuất text từ PDF – xử lý layout 2 cột thông minh"""
    try:
        import pdfplumber
        text_pages = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                width = page.width
                # Thử phát hiện layout 2 cột
                left_bbox  = (0,          0, width * 0.52, page.height)
                right_bbox = (width * 0.48, 0, width,      page.height)

                left_text  = page.within_bbox(left_bbox).extract_text()  or ""
                right_text = page.within_bbox(right_bbox).extract_text() or ""

                # Nếu cả 2 cột đều có nội dung → layout 2 cột
                if left_text.strip() and right_text.strip():
                    combined = left_text.strip() + "\n" + right_text.strip()
                else:
                    # Layout 1 cột bình thường
                    combined = page.extract_text() or ""

                text_pages.append(combined)

        return "\n".join(text_pages).strip()
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Thư viện 'pdfplumber' chưa được cài. Chạy: pip install pdfplumber"
        )


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Trích xuất text từ file DOCX"""
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # Bao gồm cả text trong bảng (Skills table thường nằm trong table)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        paragraphs.append(cell.text.strip())
        return "\n".join(paragraphs).strip()
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Thư viện 'python-docx' chưa được cài. Chạy: pip install python-docx"
        )


@app.post("/api/parse-file")
async def parse_file(file: UploadFile = File(...)):
    """
    Nhận file PDF / DOCX / TXT và trả về text đã trích xuất.
    Frontend dùng text này để điền vào textarea CV hoặc JD.
    """
    try:
        filename  = file.filename or ""
        ext       = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(status_code=400, detail="File rỗng, không thể đọc.")

        if ext == "pdf":
            text = extract_text_from_pdf(file_bytes)
        elif ext in ("docx", "doc"):
            text = extract_text_from_docx(file_bytes)
        elif ext == "txt":
            text = file_bytes.decode("utf-8", errors="replace")
        else:
            raise HTTPException(
                status_code=415,
                detail=f"Định dạng '.{ext}' không được hỗ trợ. Chỉ chấp nhận: PDF, DOCX, TXT."
            )

        if not text.strip():
            raise HTTPException(
                status_code=422,
                detail="Không trích xuất được text từ file. File có thể là ảnh (scanned PDF) hoặc bị bảo vệ."
            )

        # Đếm số từ để validate
        word_count = len(text.split())

        return {
            "success": True,
            "filename": filename,
            "text": text,
            "word_count": word_count,
            "char_count": len(text)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
    
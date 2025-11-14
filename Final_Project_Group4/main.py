# main.py
import os
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
from TorchCRF import CRF

# --- Cấu hình ứng dụng FastAPI ---
app = FastAPI(title="ABSA Model API")
#app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Lớp Pydantic cho dữ liệu đầu vào ---
class TextInput(BaseModel):
    text: str

# --- Định nghĩa Model (ĐÃ SỬA LỖI TORCH.GATHER) ---

class BertAbsaModel(nn.Module):
    def __init__(self, bert_model_name, num_tags, num_sentiments):
        super(BertAbsaModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        
        # Head 1: Aspect Term Extraction (ATE)
        self.ate_projection = nn.Linear(self.bert.config.hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        
        # Head 2: Aspect Sentiment Classification (ASC)
        # Input: [CLS] (768) + [START] (768) + [END] (768) = 2304
        self.asc_projection = nn.Linear(self.bert.config.hidden_size * 3, num_sentiments)

    def forward(self, input_ids, attention_mask, span_indices=None, span_mask=None, ate_labels=None):
        # --- BERT Output ---
        # outputs[0] (sequence_output) shape: [batch_size, seq_len, hidden_size]
        # Ví dụ: [1, 128, 768]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # Lấy [CLS] token hidden state (luôn ở vị trí 0)
        # cls_output shape: [batch_size, hidden_size]
        # Ví dụ: [1, 768]
        cls_output = sequence_output[:, 0, :] 

        # --- ATE (CRF) ---
        # ate_logits shape: [batch_size, seq_len, num_tags]
        ate_logits = self.ate_projection(sequence_output)
        
        ate_loss = None
        ate_preds = None
        if ate_labels is not None:
            # Khi training
            ate_loss = -self.crf(ate_logits, ate_labels, mask=attention_mask.byte(), reduction='mean')
        else:
            # Khi inference (dự đoán)
            # ate_preds shape: [batch_size, seq_len]
            ate_preds = self.crf.decode(ate_logits, mask=attention_mask.byte())

        # --- ASC (Span-based) ---
        sent_logits = None
        sent_loss = None 
        
        # span_indices shape: [batch_size, max_spans, 2]
        # Ví dụ: [1, 10, 2]
        if span_indices is not None and span_mask is not None:
            
            # Lấy các hằng số
            batch_size, max_spans, _ = span_indices.shape # 1, 10, 2
            hidden_size = self.bert.config.hidden_size # 768
            
            # --- BẮT ĐẦU LOGIC GATHER MỚI (ĐÃ SỬA) ---
            
            # 1. Lấy chỉ số (index) BẮT ĐẦU và KẾT THÚC của các span
            # Shape: [batch_size, max_spans] -> [1, 10]
            span_starts_idx = span_indices[:, :, 0]
            span_ends_idx = span_indices[:, :, 1]
            
            # 2. Chuẩn bị index cho hàm gather
            # Cần shape: [batch_size, max_spans, 1]
            span_starts_idx = span_starts_idx.unsqueeze(-1)
            span_ends_idx = span_ends_idx.unsqueeze(-1)
            
            # 3. Mở rộng (expand) index để khớp với hidden_size
            # Cần shape: [batch_size, max_spans, hidden_size] -> [1, 10, 768]
            span_starts_idx_expanded = span_starts_idx.expand(-1, -1, hidden_size)
            span_ends_idx_expanded = span_ends_idx.expand(-1, -1, hidden_size)

            # 4. Dùng torch.gather
            # sequence_output shape: [1, 128, 768]
            # index shape: [1, 10, 768]
            # dim=1 (chiều seq_len)
            # -> start_states shape: [1, 10, 768]
            start_states = sequence_output.gather(dim=1, index=span_starts_idx_expanded)
            end_states = sequence_output.gather(dim=1, index=span_ends_idx_expanded)
            
            # 5. Chuẩn bị [CLS] output
            # cls_output shape: [1, 768]
            # -> Mở rộng thành: [1, 10, 768]
            cls_output_expanded = cls_output.unsqueeze(1).expand(-1, max_spans, -1)
            
            # --- KẾT THÚC LOGIC GATHER MỚI ---

            # 6. Ghép 3 tensor lại
            # Ghép ở dim=2 (chiều hidden_size)
            # [1, 10, 768] + [1, 10, 768] + [1, 10, 768]
            # -> span_repr shape: [1, 10, 2304]
            span_repr = torch.cat([cls_output_expanded, start_states, end_states], dim=2)
            
            # 7. Đưa qua lớp projection
            # asc_projection nhận đầu vào [..., 2304]
            # -> sent_logits shape: [1, 10, 3] (num_sentiments)
            sent_logits = self.asc_projection(span_repr)
            
            # 8. Áp dụng mask (che đi các span không dùng)
            # span_mask shape: [1, 10] -> unsqueeze(-1) -> [1, 10, 1]
            sent_logits = sent_logits * span_mask.unsqueeze(-1).float()

        return ate_logits, ate_preds, sent_logits, ate_loss, sent_loss

# --- Load Model và Tokenizer (Chạy 1 lần khi app khởi động) ---
@torch.inference_mode()
def load_model_components():
    """Hàm này load tất cả các thành phần cần thiết cho model."""
    BERT_MODEL_NAME = 'bert-base-uncased'
    MODEL_PATH = 'best_absa_bert_span_v2.pth'
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy file model: {MODEL_PATH}. "
                              f"Hãy đảm bảo bạn đã đặt file .pth vào cùng thư mục với main.py")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Các mapping (lấy từ file .py của bạn)
    tag2id = {'O': 0, 'B-ASP': 1, 'I-ASP': 2}
    sentiment2id = {'positive': 0, 'neutral': 1, 'negative': 2}
    id2tag = {v: k for k, v in tag2id.items()}
    id2sent = {v: k for k, v in sentiment2id.items()}
    
    # Khởi tạo model và tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertAbsaModel(
        bert_model_name=BERT_MODEL_NAME,
        num_tags=len(tag2id),
        num_sentiments=len(sentiment2id)
    )
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError as e:
        print("Lỗi khi load state_dict. Có thể do tên class model không khớp.")
        print("Đang thử load 'model' state_dict từ checkpoint...")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            raise e

    model.to(device)
    model.eval()
    
    print("Model loaded successfully.")
    return model, tokenizer, device, id2tag, id2sent

# --- Khởi tạo các biến global ---
try:
    model, tokenizer, device, id2tag, id2sent = load_model_components()
    MAX_LEN = 128 # Đặt max length cho tokenizer
    MAX_ASPECTS = 10 # Giới hạn số aspect/câu
except FileNotFoundError as e:
    print(e)
    model = None # Sẽ báo lỗi nếu model không được load

# --- Hàm xử lý dự đoán (Trích xuất logic từ file .py của bạn) ---
@torch.inference_mode()
def predict_aspects(text: str):
    """
    Hàm này nhận 1 câu text và trả về 1 dict chứa các aspect,
    sentiment của chúng, và sentiment tổng thể.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model không được load. Kiểm tra file model .pth")

    try:
        # 1. Tokenize
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # 2. ATE (Trích xuất Aspect)
        _, ate_preds, _, _, _ = model(input_ids, attention_mask)
        ate_preds = ate_preds[0] # Lấy kết quả của batch 0
        
        # 3. Chuyển ATE tags (B-ASP, I-ASP) thành spans
        spans = []
        start_idx = -1
        for i, tag_id in enumerate(ate_preds):
            tag = id2tag.get(tag_id)
            if tag == 'B-ASP':
                if start_idx != -1: # Đóng span trước đó
                    spans.append((start_idx, i - 1))
                start_idx = i
            elif tag == 'I-ASP':
                if start_idx == -1: # I-ASP mà không có B-ASP trước, bỏ qua
                    continue
            elif tag == 'O':
                if start_idx != -1: # Đóng span
                    spans.append((start_idx, i - 1))
                    start_idx = -1
        if start_idx != -1: # Đóng span cuối cùng
            spans.append((start_idx, len(ate_preds) - 1))
        
        # Giới hạn số spans
        spans = spans[:MAX_ASPECTS]
        n = len(spans)

        if n == 0:
            return {"aspects": [], "overall": "neutral"} # Không có aspect
        
        # 4. Chuẩn bị đầu vào cho ASC
        span_indices = torch.zeros((1, MAX_ASPECTS, 2), dtype=torch.long, device=device)
        span_mask = torch.zeros((1, MAX_ASPECTS), dtype=torch.bool, device=device)
        
        for i, (s, e) in enumerate(spans):
            span_indices[0, i, 0] = s
            span_indices[0, i, 1] = e
        span_mask[0, :n] = True

        # 5. ASC (Phân loại Sentiment)
        _, _, sent_logits, _, _ = model(input_ids, attention_mask, span_indices, span_mask)
        
        if sent_logits is None or sent_logits.shape[1] == 0:
            return {"aspects": [], "overall": "neutral"}
        
        # Lấy kết quả sentiment
        preds = torch.argmax(sent_logits, dim=2)[0][:n] # Lấy n kết quả đầu
        
        # 6. Tổng hợp kết quả (Logic từ file main.py (back) của bạn)
        aspects = []
        sentiments = []
        for i, (s, e) in enumerate(spans):
            term = tokenizer.convert_tokens_to_string(tokens[s:e+1]).strip()
            sent = id2sent[preds[i].item()]
            
            # Xử lý các token đặc biệt (ví dụ: '##')
            term = term.replace(' ##', '').replace('##', '')
            
            aspects.append({"term": term, "sentiment": sent})
            sentiments.append(sent)

        # 7. Tính Overall Sentiment (Logic từ file main.py (back) của bạn)
        pos = sentiments.count('positive')
        neg = sentiments.count('negative')
        overall = "positive" if pos > neg else "negative" if neg > pos else "neutral"

        return {"aspects": aspects, "overall": overall}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Inference error: {str(e)}"}


# --- Định nghĩa API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Phục vụ file index.html"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def handle_prediction(text_input: TextInput):
    """Nhận text và trả về kết quả dự đoán (JSON)"""
    if not text_input.text.strip():
        raise HTTPException(status_code=400, detail="Text không được để trống")
    
    result = predict_aspects(text_input.text)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

# --- Chạy ứng dụng ---
if __name__ == "__main__":
    print("Khởi chạy server Uvicorn tại http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
# ♟️ Fischer-Style Chess AI — Behavioral Cloning Move Predictor

> **Trạng thái: đang phát triển (work in progress)** — pipeline dữ liệu, mô hình, huấn luyện và đánh giá đã hoạt động end-to-end; đang trong giai đoạn cải thiện chất lượng dự đoán.

Hệ thống AI dự đoán nước đi cờ vua theo **phong cách chơi của Bobby Fischer**, huấn luyện bằng **Behavioral Cloning** trên chính các ván đấu lịch sử của ông (827 ván), lọc chất lượng dữ liệu bằng Stockfish, và đánh giá sức mạnh bằng cách cho model đấu trực tiếp với Stockfish.

**Stack:** Python, PyTorch (CNN Residual Policy Network), FastAPI · React 18 + `react-chessboard`

---

## ✨ Đã triển khai

| Hạng mục | Trạng thái |
|---|---|
| **Data pipeline** | Parse 827 ván PGN của Fischer → lọc nước đi bằng Stockfish (loại "blunder" theo ngưỡng centipawn) → cache ra `.jsonl` (~33.000 mẫu huấn luyện) để không phải chạy lại engine mỗi epoch |
| **Board/Move encoding** | FEN → tensor 18 kênh (12 kênh quân cờ, 4 kênh quyền nhập thành, 1 kênh en-passant, 1 kênh lượt đi); action space rời rạc 4.672 nước đi hợp lệ có thể có |
| **Mô hình (FischerPolicyNet)** | Residual CNN (8 residual block, 128 channels) → policy head dự đoán logits trên 4.672 hành động |
| **Huấn luyện (Behavioral Cloning)** | Train/val split theo cấp độ ván đấu (tránh rò rỉ dữ liệu — data leakage), learning-rate scheduler, mixed-precision, checkpoint theo best-val — đã chạy 30 epoch, log đầy đủ loss/accuracy mỗi epoch |
| **Suy luận có che nước đi bất hợp lệ** | API luôn chọn nước có xác suất cao nhất **trong số các nước đi hợp lệ** (legal-move masking bằng `python-chess`), không bao giờ trả về nước sai luật |
| **Đánh giá đối kháng với Stockfish** | Script tự động cho model đấu nhiều ván với Stockfish (mức Elo, thời gian suy nghĩ cấu hình được), xuất kết quả PGN + CSV để phân tích |
| **REST API (FastAPI)** | `POST /api/play/fischer` nhận FEN, trả về nước đi UCI từ policy đã huấn luyện; `GET /api/health` cho health check |
| **Giao diện chơi cờ (React)** | Bàn cờ đầy đủ: chọn màu quân, xoay bàn theo màu người chơi, kéo-thả/click để đi, highlight nước hợp lệ/nước vừa đi/chiếu tướng, sổ ghi nước đi (scoresheet), giao diện kính mờ (glassmorphism) tự thiết kế |
| **Unit test** | Kiểm thử encoding (shape, dtype tensor), action-space mapping, và tính đúng đắn của việc chia tập train/val theo ván (không leak dữ liệu giữa 2 tập) |

## 🚧 Đang trong quá trình cải thiện

- Model hiện **overfit rõ rệt trên tập train** (~97% top-1 accuracy) trong khi **độ chính xác trên tập validation dừng ở khoảng ~26%** sau 30 epoch — đúng như dự kiến với lượng dữ liệu còn hạn chế (827 ván) so với độ phức tạp của cờ vua.
- Kết quả đối đầu Stockfish hiện dùng để **làm cơ sở đo lường tiến bộ qua các lần huấn luyện lại**, chưa phải mục tiêu cuối; các hướng cải thiện đang cân nhắc: tăng dữ liệu huấn luyện, regularization mạnh hơn, data augmentation (xoay/lật bàn cờ), hoặc bổ sung self-play/RL sau giai đoạn Behavioral Cloning.

---

## 🏗️ Kiến trúc hệ thống

```
chess_system/
├── backend/
│   ├── src/
│   │   ├── data_processing/   # PGN parsing, lọc bằng Stockfish, FEN↔tensor encoding
│   │   ├── models/             # FischerPolicyNet (Residual CNN)
│   │   ├── training/           # Vòng lặp huấn luyện BC, script đánh giá vs. Stockfish
│   │   ├── services/           # ai_engine.py — load checkpoint, suy luận có che nước bất hợp lệ
│   │   ├── routes/             # FastAPI endpoints
│   │   └── config/             # Cấu hình tập trung (pydantic-settings)
│   ├── data/                   # raw/ (PGN gốc) + cache/ (dữ liệu đã xử lý)
│   ├── logs/                   # Metrics huấn luyện + kết quả đấu Stockfish
│   └── test/                   # Unit test cho data pipeline
└── frontend/
    └── src/                    # React app: bàn cờ, context quản lý ván đấu, gọi API
```

```
PGN (827 ván Fischer)
   │  lọc bằng Stockfish (loại blunder)
   ▼
Training examples (.jsonl, ~33k mẫu)
   │  FEN → tensor 18 kênh
   ▼
FischerPolicyNet (Residual CNN, PyTorch)
   │  Behavioral Cloning — CrossEntropyLoss trên 4.672 action
   ▼
Checkpoint (.pth)
   │
   ▼
FastAPI /api/play/fischer  ──legal-move masking──▶  React chessboard
```

---

## 🚀 Cài đặt & chạy dự án

### Yêu cầu
- Python ≥ 3.10, Node.js ≥ 18
- [Stockfish](https://stockfishchess.org/download/) (binary riêng, dùng để lọc dữ liệu và đánh giá đối kháng)

### Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```
Frontend gọi API tại `VITE_API_BASE_URL` (mặc định `http://localhost:8000`).

### Chạy lại pipeline dữ liệu & huấn luyện (tùy chọn)

```bash
# 1. Build cache dữ liệu huấn luyện từ PGN (chạy 1 lần, tốn thời gian vì gọi Stockfish cho từng nước)
python -m src.data_processing.build_cache

# 2. Huấn luyện Behavioral Cloning
python -m src.training.train_bc

# 3. Đánh giá đối đầu Stockfish
python -m src.training.evaluate_vs_stockfish
```

---

## 👤 Vai trò cá nhân trong dự án

Tự thiết kế và triển khai toàn bộ pipeline: xử lý dữ liệu PGN + lọc chất lượng bằng Stockfish, thiết kế encoding bàn cờ/nước đi cho mạng neural, xây dựng kiến trúc Residual CNN policy network bằng PyTorch, viết vòng lặp huấn luyện Behavioral Cloning (kèm chia tập tránh rò rỉ dữ liệu, learning-rate scheduling, mixed precision), script đánh giá đối kháng với Stockfish, API suy luận bằng FastAPI có che nước đi bất hợp lệ, và giao diện chơi cờ bằng React.


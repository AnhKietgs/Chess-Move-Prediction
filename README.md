# Fischer-Style Chess AI

Hệ thống cờ vua mô phỏng một phần lựa chọn nước đi của Bobby Fischer bằng
Behavioral Cloning. Dự án gồm pipeline PGN, mô hình PyTorch, API FastAPI và
giao diện React để chơi, phân tích và trình bày bằng chứng về phong cách.

> Trạng thái: sẵn sàng chạy và demo local. Mô hình là policy bắt chước phong
> cách, không phải chess engine có tìm kiếm sâu; sức mạnh thi đấu raw policy
> không tương đương một engine như Stockfish.

## Điểm chính

- PGN nguồn hiện có 2.015 ván. Cache hiện tại chứa 48.479 mẫu từ 1.238
  game ID sau khi khử main line trùng, lọc điều kiện game và loại nước đi
  blunder bằng Stockfish.
- FEN được mã hóa thành tensor 18 kênh; nước đi dùng action space 4.672 lớp.
- FischerPolicyNet là Residual CNN 64 channels, 4 residual blocks, 32 policy
  channels và action-plane policy head 73 planes.
- Action-plane head giữ đúng ánh xạ from-square x 73 move planes, sau đó trả
  raw logits có shape batch x 4672.
- Legal-move masking được dùng nhất quán khi train, đánh giá và inference:
  model chỉ xếp hạng giữa các nước đi hợp lệ của vị trí hiện tại.
- Loss là legal-move cross-entropy với label smoothing; train dùng AdamW,
  ReduceLROnPlateau theo Validation Top-1, mixed precision, resume checkpoint
  và early stopping.
- Train augmentation lật ngang được thực hiện lazy trong DataLoader. Các thế
  còn quyền nhập thành không bị lật để tránh cặp board/label không hợp lệ.
- Tách dữ liệu theo game; chế độ strict loại mọi FEN val/test đã xuất hiện ở
  split trước, dùng FEN chuẩn hóa 4 trường đầu.
- API có raw policy mode và Stockfish safety-net tùy chọn. Safety-net kiểm tra
  Top-3 legal moves, từ chối blunder theo ngưỡng centipawn và fallback về
  Stockfish nếu cần.

## Kết quả đánh giá hiện tại

Checkpoint mặc định:

    backend/checkpoints/action_plane/best_fischer_bc.pth

Đánh giá strict FEN-disjoint trên 2.716 mẫu test:

| Chỉ số | Kết quả |
| --- | ---: |
| Test loss | 3.0148 |
| Test Top-1 | 20.77% |
| Test Top-3 | 38.81% |

Kết quả này tốt hơn checkpoint dense cũ trên cùng strict test
(Top-1 19.18%, Top-3 37.15%), nhưng chưa chứng minh mô hình tái tạo hoàn
chỉnh phong cách hay năng lực chiến thuật của Fischer.

Raw policy không dùng safety-net đạt 2.00% score khi đấu 100 ván Stockfish
Elo 1320 (0 thắng, 4 hòa, 96 thua). Đây là baseline sức mạnh policy thuần.
Kết quả có safety-net phải được báo cáo riêng vì Stockfish tham gia chọn nước
đi để bảo vệ demo khỏi blunder.

## Kiến trúc

    chess_system/
    ├── backend/
    │   ├── src/
    │   │   ├── config/          # Pydantic settings và biến môi trường
    │   │   ├── data_processing/ # PGN, Stockfish filter, encoder, dataset
    │   │   ├── models/          # FischerPolicyNet và legal-mask loss
    │   │   ├── training/        # train/evaluate BC và Stockfish arena
    │   │   ├── services/        # inference và safety-net
    │   │   ├── routes/          # play + analytics FastAPI APIs
    │   │   └── middleware/      # CORS
    │   ├── data/
    │   │   ├── raw/             # PGN Fischer gốc
    │   │   └── cache/           # JSONL state/action đã xử lý
    │   ├── checkpoints/         # model checkpoints, không commit Git
    │   ├── logs/                # metrics CSV, PGN/CSV arena
    │   ├── tests/               # 40 automated tests
    │   └── Dockerfile
    └── frontend/
        ├── src/
        │   ├── components/      # board, dashboard, scoresheet, material
        │   ├── hooks/           # game loop và playback bàn phím
        │   ├── pages/
        │   └── services/        # FastAPI client
        └── public/

Luồng dữ liệu:

    PGN Fischer
      -> parse + game deduplication + Stockfish blunder filter
      -> JSONL records: fen, move_uci, game_id
      -> game-level / strict FEN-disjoint split
      -> lazy tensor + legal mask + optional mirror augmentation
      -> FischerPolicyNet action-plane policy
      -> checkpoint
      -> FastAPI legal-mask inference
      -> React chessboard và analytics dashboard

## Yêu cầu

- Python 3.11 trở lên
- Node.js 18 trở lên
- Stockfish cho local backend, với đường dẫn cấu hình qua STOCKFISH_PATH
- Môi trường ảo Python và npm dependencies

## Chạy local

Từ thư mục repository:

    cd chess_system/backend
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Tạo hoặc cập nhật backend/.env cho máy local:

    ENVIRONMENT=local
    CORS_ORIGINS=http://localhost:1010,http://127.0.0.1:1010
    MODEL_CHECKPOINT_PATH=./checkpoints/action_plane/best_fischer_bc.pth
    STOCKFISH_PATH=D:\duong-dan\den\stockfish.exe
    TRAINING_STRICT_FEN_DISJOINT=true

Chạy backend:

    python -m uvicorn main:app --reload --port 8000

Ở terminal khác:

    cd chess_system/frontend
    npm install
    npm run dev

Vite chạy tại http://localhost:1010. Frontend mặc định gọi
http://localhost:8000. Khi backend ở host khác, tạo frontend/.env:

    VITE_API_BASE_URL=https://your-backend.example.com

## API

| Endpoint | Mô tả |
| --- | --- |
| GET /api/health | Health check |
| POST /api/play/fischer | Nhận FEN, trả UCI move hợp lệ |
| GET /api/analytics/opening_stats | So sánh khai cuộc hoặc phòng thủ Fischer và AI |
| GET /api/analytics/heldout_examples | Ví dụ strict hold-out không chọn theo match rate |
| POST /api/analytics/evaluate_pgn | Upload PGN để tính Top-1 và Top-3 match rate |

Ví dụ request:

    POST /api/play/fischer
    {
      "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "use_safety_net": true
    }

use_safety_net bằng false yêu cầu raw model chọn Top-1 hợp lệ. Giá trị true
là chế độ mặc định cho demo an toàn, không dùng để tuyên bố sức mạnh raw model.

## Giao diện

- Chọn chơi Trắng hoặc Đen, kéo-thả/click quân cờ, phong cấp, highlight nước
  hợp lệ, nước gần nhất và chiếu tướng.
- Fischer is thinking có avatar và thời gian hiển thị tối thiểu 3.5 giây.
- Hiển thị quân bị bắt và material advantage theo giá trị Tốt 1,
  Mã/Tượng 3, Xe 5, Hậu 9.
- Scoresheet hỗ trợ phím mũi tên trái/phải để tua từng half-move. Bàn cờ,
  material balance, heatmap và nước đang chọn cùng quay về vị trí lịch sử.
- Có nút resign, new game và switch bật/tắt Stockfish safety-net.
- Dashboard gồm ba tab: Style verification, Strict hold-out examples và
  Opening/Defensive distribution. Biểu đồ tự đổi sang phân tích phòng thủ khi
  AI cầm quân Đen.
- Có thể upload PGN hold-out bên ngoài để lấy Top-1/Top-3 match rate.

## Pipeline dữ liệu và train

Build cache từ PGN. Mỗi worker tạo và tái sử dụng một Stockfish process:

    cd chess_system/backend
    python -m src.data_processing.build_cache --num-workers 4

Train action-plane policy với cấu hình hiện tại:

    python -m src.training.train_bc

Các hyperparameter nằm trong src/config/settings.py và có thể override bằng
biến môi trường. Ví dụ PowerShell cho một run action-plane riêng:

    $env:TRAINING_CHECKPOINT_DIR="./checkpoints/action_plane"
    $env:TRAINING_METRICS_PATH="./logs/behavioral_cloning_metrics_action_plane.csv"
    $env:TRAINING_LEARNING_RATE="3e-4"
    $env:TRAINING_WEIGHT_DECAY="5e-4"
    $env:TRAINING_STRICT_FEN_DISJOINT="true"
    python -m src.training.train_bc

Checkpoint gồm model, optimizer, scheduler, epoch, best validation Top-1 và
early-stopping state. Đặt TRAINING_RESUME_PATH để tiếp tục một run bị dừng.

Đánh giá checkpoint cấu hình hiện tại:

    python -m src.training.evaluate_bc

Đấu với Stockfish, 100 ván, chia đều màu:

    python -m src.training.evaluate_vs_stockfish --games 100 --elo 1320 --no-safety-net

Bỏ --no-safety-net để đo chất lượng trải nghiệm demo có blunder guard. Elo
thấp nhất phụ thuộc phiên bản Stockfish cài trên máy; build hiện tại hỗ trợ từ
khoảng Elo 1320.

## Kiểm thử

Backend có test cho encoder/action planes, legal mask/loss, mirror
augmentation, strict FEN split, game deduplication, analytics, safety-net và
play lifecycle.

    cd chess_system/backend
    ..\.venv\Scripts\python.exe -m pytest -q

Kết quả kiểm tra cuối: 40 passed.

Build frontend:

    cd chess_system/frontend
    npm run build

## Deploy

backend/Dockerfile dùng Python 3.11 slim và cài Stockfish Linux, phù hợp để
deploy backend bằng Docker trên Railway hoặc nền tảng tương tự.

Lưu ý bắt buộc trước khi deploy:

1. File checkpoint .pth bị Git ignore, nên phải upload/mount checkpoint vào
   service hoặc dùng storage riêng. MODEL_CHECKPOINT_PATH phải trỏ tới file đó.
2. Cấu hình CORS_ORIGINS thành URL frontend production.
3. Build frontend với VITE_API_BASE_URL là URL backend production.
4. Không commit .env chứa đường dẫn local hay secret.

## Giới hạn và hướng phát triển

Mô hình hiện chứng minh được một phần mức độ khớp lựa chọn nước đi Fischer
trên strict held-out test, không phải một tái tạo hoàn hảo của phong cách.
Các hướng cải thiện thực tế gồm mở rộng dữ liệu Fischer/nguồn phong cách
tương tự, tăng đa dạng vị trí, tune hyperparameter trên validation, thêm
value head + search, hoặc PPO/self-play sau khi giữ một baseline BC rõ ràng.

## Tác giả

Tự thiết kế và triển khai pipeline dữ liệu, Stockfish filtering, board/move
encoding, action-plane policy, Behavioral Cloning loop, đánh giá, inference
API, safety-net, analytics dashboard và giao diện React.

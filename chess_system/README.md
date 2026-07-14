# Fischer-Style Chess AI — Full-Stack Scaffold

## Run it

**Backend**
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```
Frontend expects the API at `VITE_API_BASE_URL` (see `frontend/.env`), default `http://localhost:8000`.

## What's implemented in this pass

- `POST /api/play/fischer` — accepts a FEN, returns a **random legal move**
  via `python-chess` (explicit placeholder for the trained BC model).
- Full React board: color-select modal, board orientation flips to match
  the human's color, auto AI opening move when the human plays Black,
  drag-and-drop + click-to-move, legal-move / last-move / check
  highlighting, and a move-history "scoresheet" sidebar.
- Glassmorphism design system ("The Fischer Study") defined as CSS
  variables in `frontend/src/index.css`.

## What's intentionally empty

`backend/src/data_processing` and `backend/src/training` contain only
README placeholders — that's Weeks 1-6 of the project plan (PGN parsing,
FEN→Tensor encoding, the BC model itself). Swapping the placeholder in
`backend/src/services/chess_engine.py::select_move` for a real model call
is the only change needed to wire a trained model into this API without
touching the frontend.

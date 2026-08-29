"""Chay tu thu muc backend: python debug_safety_net.py
In ra checkpoint dang duoc load va delta_cp thuc te cua top-3 nuoc model
de tim nguyen nhan ty le fallback bat thuong."""
import sys
sys.path.insert(0, ".")

import chess
from src.config.settings import settings
from src.services.ai_engine import FischerAI, create_stockfish_engine

print("Checkpoint dang load:", settings.model_checkpoint_path)
print("inference_stockfish_depth:", settings.inference_stockfish_depth)
print("inference_top_k:", settings.inference_top_k)
print("inference_blunder_threshold_cp:", settings.inference_blunder_threshold_cp)

engine = create_stockfish_engine()
fischer_ai = FischerAI(engine=engine)

# Vi tri khai cuoc - de xem model de xuat gi ngay tu dau
board = chess.Board()
from src.services.ai_engine import _model_top_moves, _MATE_SCORE_CP
top_moves = _model_top_moves(board, fischer_ai.model, settings.inference_top_k)
print("\nVi tri khai cuoc - model de xuat:", [m.uci() for m in top_moves])

mover = board.turn
limit__ = chess.engine.Limit(depth=settings.inference_stockfish_depth)
score_before = engine.analyse(board, limit__)["score"].pov(mover).score(mate_score=_MATE_SCORE_CP)
print("score_before:", score_before)
for mv in top_moves:
    board.push(mv)
    score_after = engine.analyse(board, limit__)["score"].pov(mover).score(mate_score=_MATE_SCORE_CP)
    board.pop()
    print(f"  {mv.uci()}: score_after={score_after}, delta_cp={score_after - score_before}")

engine.quit()
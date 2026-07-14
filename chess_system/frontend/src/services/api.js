/**
 * Thin fetch wrapper around the FastAPI backend. This is the ONLY module
 * that should know the backend's URL shape — components/hooks call these
 * functions, never `fetch` directly.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Ask the backend for the AI's move given the current position.
 *
 * @param {string} fen - Current board state in FEN notation.
 * @returns {Promise<{moveUci: string, moveSan: string, fenAfter: string, isCheckmate: boolean, isStalemate: boolean, isCheck: boolean, gameOver: boolean}>}
 */
export async function requestFischerMove(fen) {
  const response = await fetch(`${API_BASE_URL}/api/play/fischer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fen }),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(body.detail || "The AI failed to produce a move.", response.status);
  }

  const data = await response.json();
  return {
    moveUci: data.move_uci,
    moveSan: data.move_san,
    fenAfter: data.fen_after,
    isCheckmate: data.is_checkmate,
    isStalemate: data.is_stalemate,
    isCheck: data.is_check,
    gameOver: data.game_over,
  };
}

export { ApiError };

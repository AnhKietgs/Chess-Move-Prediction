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
 * @param {boolean} [useSafetyNet=true] - Whether Stockfish may reject a
 * likely policy blunder before returning the move.
 * @returns {Promise<{moveUci: string}>}
 */
export async function requestFischerMove(fen, useSafetyNet = true) {
  const response = await fetch(`${API_BASE_URL}/api/play/fischer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fen, use_safety_net: useSafetyNet }),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(body.detail || "The AI failed to produce a move.", response.status);
  }

  const data = await response.json();
  if (typeof data.move !== "string" || data.move.length === 0) {
    throw new ApiError("The AI returned an invalid move response.", 502);
  }

  return {
    moveUci: data.move,
  };
}

/**
 * Fetch Fischer-versus-policy opening distributions for one Fischer color.
 *
 * @param {"white"|"black"} fischerColor - Fischer's side in the game.
 * @returns {Promise<{actual: {move: string, value: number, count: number}[], ai: {move: string, value: number, count: number}[], sample_size: number}>}
 */
export async function getOpeningStats(fischerColor) {
  const color = fischerColor === "black" ? "black" : "white";
  const response = await fetch(
    `${API_BASE_URL}/api/analytics/opening_stats?color=${color}`
  );
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(body.detail || "Could not load opening analytics.", response.status);
  }
  return response.json();
}

/**
 * Evaluate policy agreement against an uploaded Fischer PGN.
 *
 * @param {File} pgnFile - PGN file selected by the user.
 * @returns {Promise<{top1_match_rate: number, top3_match_rate: number, positions_evaluated: number, games_evaluated: number}>}
 */
export async function evaluateFischerPgn(pgnFile) {
  const formData = new FormData();
  formData.append("pgn_file", pgnFile);
  const response = await fetch(`${API_BASE_URL}/api/analytics/evaluate_pgn`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(body.detail || "Could not evaluate the PGN.", response.status);
  }
  return response.json();
}

export { ApiError };

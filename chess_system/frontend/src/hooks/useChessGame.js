import { useCallback, useEffect, useRef, useState } from "react";
import { Chess } from "chess.js";
import { requestFischerMove, ApiError } from "../services/api.js";

/**
 * @typedef {Object} MoveHistoryEntry
 * @property {number} moveNumber
 * @property {string} san
 * @property {"w"|"b"} color
 */

/**
 * Owns the live chess.js game instance and orchestrates the play loop:
 * player drags a piece -> if legal, apply it -> if it's now the AI's turn,
 * lock the board, call the backend, apply the returned move, unlock.
 *
 * @param {"w"|"b"|null} playerColor
 */
export function useChessGame(playerColor) {
  const gameRef = useRef(new Chess());

  const [fen, setFen] = useState(gameRef.current.fen());
  const [history, setHistory] = useState(/** @type {MoveHistoryEntry[]} */ ([]));
  const [lastMove, setLastMove] = useState(/** @type {{from: string, to: string}|null} */ (null));
  const [isAiThinking, setIsAiThinking] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const syncFromGame = useCallback(() => {
    const game = gameRef.current;
    setFen(game.fen());

    const verboseHistory = game.history({ verbose: true });
    setHistory(
      verboseHistory.map((move, index) => ({
        moveNumber: Math.floor(index / 2) + 1,
        san: move.san,
        color: move.color,
      }))
    );

    if (game.isCheckmate()) {
      setStatusMessage(`Checkmate — ${game.turn() === "w" ? "Black" : "White"} wins.`);
    } else if (game.isStalemate()) {
      setStatusMessage("Stalemate — draw.");
    } else if (game.isDraw()) {
      setStatusMessage("Draw.");
    } else if (game.inCheck()) {
      setStatusMessage("Check.");
    } else {
      setStatusMessage("");
    }
  }, []);

  const requestAiMove = useCallback(async () => {
    const game = gameRef.current;
    if (game.isGameOver()) return;

    setIsAiThinking(true);
    setErrorMessage("");
    try {
      const result = await requestFischerMove(game.fen());
      const applied = game.move(result.moveUci, { sloppy: true });
      if (applied) {
        setLastMove({ from: applied.from, to: applied.to });
      }
      syncFromGame();
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Could not reach the AI backend.";
      setErrorMessage(message);
    } finally {
      setIsAiThinking(false);
    }
  }, [syncFromGame]);

  /**
   * Attempt to play the human's move. Returns true if it was legal and applied.
   *
   * @param {string} from
   * @param {string} to
   * @param {string} [promotion="q"]
   */
  const makePlayerMove = useCallback(
    (from, to, promotion = "q") => {
      const game = gameRef.current;
      if (game.isGameOver() || isAiThinking) return false;
      if (playerColor && game.turn() !== playerColor) return false;

      const move = game.move({ from, to, promotion });
      if (!move) return false;

      setLastMove({ from: move.from, to: move.to });
      syncFromGame();
      return true;
    },
    [playerColor, isAiThinking, syncFromGame]
  );

  const resetGame = useCallback(() => {
    gameRef.current = new Chess();
    setLastMove(null);
    setErrorMessage("");
    syncFromGame();
  }, [syncFromGame]);

  // If the human chose Black, the AI (White) must play the opening move
  // automatically as soon as the game starts.
  useEffect(() => {
    const game = gameRef.current;
    if (playerColor === "b" && game.history().length === 0) {
      requestAiMove();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playerColor]);

  // After every player move, if it's now the AI's turn, trigger it.
  useEffect(() => {
    const game = gameRef.current;
    if (!playerColor || game.isGameOver()) return;
    if (game.turn() !== playerColor && !isAiThinking) {
      requestAiMove();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fen]);

  return {
    game: gameRef.current,
    fen,
    history,
    lastMove,
    isAiThinking,
    statusMessage,
    errorMessage,
    makePlayerMove,
    resetGame,
    isPlayerTurn: playerColor ? gameRef.current.turn() === playerColor && !isAiThinking : false,
  };
}

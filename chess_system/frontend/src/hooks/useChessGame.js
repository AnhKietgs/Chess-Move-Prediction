import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Chess } from "chess.js";
import { requestFischerMove, ApiError } from "../services/api.js";

const MINIMUM_AI_THINKING_MS = 1500;

/**
 * Wait for a bounded UI delay without blocking the browser.
 *
 * @param {number} milliseconds Delay duration in milliseconds.
 * @returns {Promise<void>} Promise resolved after the requested delay.
 */
function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

/**
 * @typedef {Object} MoveHistoryEntry
 * @property {number} moveNumber
 * @property {string} san
 * @property {"w"|"b"} color
 * @property {string} from
 * @property {string} to
 * @property {string|undefined} captured
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
  const [hasResigned, setHasResigned] = useState(false);
  const [useSafetyNet, setUseSafetyNet] = useState(true);
  const [viewedPly, setViewedPly] = useState(/** @type {number|null} */ (null));
  const hasResignedRef = useRef(false);

  const displayedPly = viewedPly ?? history.length;
  const isReviewingHistory = displayedPly < history.length;
  const visibleHistory = useMemo(
    () => history.slice(0, displayedPly),
    [displayedPly, history]
  );
  const displayGame = useMemo(() => {
    const replayGame = new Chess();
    visibleHistory.forEach((move) => replayGame.move(move.san));
    return replayGame;
  }, [visibleHistory]);
  const displayLastMove = displayedPly > 0
    ? {
      from: history[displayedPly - 1].from,
      to: history[displayedPly - 1].to,
    }
    : null;

  const syncFromGame = useCallback(() => {
    const game = gameRef.current;
    setFen(game.fen());

    const verboseHistory = game.history({ verbose: true });
    setHistory(
      verboseHistory.map((move, index) => ({
        moveNumber: Math.floor(index / 2) + 1,
        san: move.san,
        color: move.color,
        from: move.from,
        to: move.to,
        captured: move.captured,
      }))
    );

    if (hasResignedRef.current) {
      return;
    }
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
    if (game.isGameOver() || hasResignedRef.current) return;

    const thinkingStartedAt = Date.now();
    setIsAiThinking(true);
    setErrorMessage("");
    try {
      const result = await requestFischerMove(game.fen(), useSafetyNet);
      const remainingThinkingTime = MINIMUM_AI_THINKING_MS - (Date.now() - thinkingStartedAt);
      if (remainingThinkingTime > 0) {
        await wait(remainingThinkingTime);
      }

      if (hasResignedRef.current || gameRef.current !== game) return;
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
  }, [syncFromGame, useSafetyNet]);

  /**
   * Attempt to play the human's move. Returns true if it was legal and applied.
   *
   * chess.js 1.x *throws* on an illegal move object instead of returning
   * null (unlike older versions) — this must stay wrapped in try/catch, or
   * an illegal click-target (e.g. clicking a second own piece to switch
   * selection) throws mid-handler and the caller's state update never runs.
   *
   * @param {string} from
   * @param {string} to
   * @param {string} [promotion="q"]
   */
  const makePlayerMove = useCallback(
    (from, to, promotion = "q") => {
      const game = gameRef.current;
      if (game.isGameOver() || isAiThinking || hasResignedRef.current) return false;
      if (playerColor && game.turn() !== playerColor) return false;

      let move;
      try {
        move = game.move({ from, to, promotion });
      } catch {
        return false; // illegal move — chess.js throws rather than returning null
      }
      if (!move) return false;

      setLastMove({ from: move.from, to: move.to });
      syncFromGame();
      return true;
    },
    [playerColor, isAiThinking, syncFromGame]
  );

  const resetGame = useCallback(() => {
    gameRef.current = new Chess();
    hasResignedRef.current = false;
    setHasResigned(false);
    setViewedPly(null);
    setLastMove(null);
    setErrorMessage("");
    syncFromGame();
  }, [syncFromGame]);

  const resignGame = useCallback(() => {
    if (!playerColor || gameRef.current.isGameOver() || hasResignedRef.current) return;
    hasResignedRef.current = true;
    setHasResigned(true);
    setIsAiThinking(false);
    setErrorMessage("");
    setStatusMessage("You resigned — Fischer wins.");
  }, [playerColor]);

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
    if (!playerColor || game.isGameOver() || hasResigned) return;
    if (game.turn() !== playerColor && !isAiThinking) {
      requestAiMove();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fen]);

  // Arrow keys navigate the score sheet one ply at a time. A null
  // viewedPly represents the live position so incoming AI moves stay live.
  useEffect(() => {
    const handleKeyDown = (event) => {
      const target = event.target instanceof HTMLElement ? event.target : null;
      if (target?.closest("input, textarea, select, [contenteditable='true']")) {
        return;
      }
      if (!["ArrowLeft", "ArrowRight"].includes(event.key) || history.length === 0) {
        return;
      }

      event.preventDefault();
      setViewedPly((currentPly) => {
        const basePly = currentPly ?? history.length;
        if (event.key === "ArrowLeft") {
          return Math.max(0, basePly - 1);
        }

        const nextPly = Math.min(history.length, basePly + 1);
        return nextPly === history.length ? null : nextPly;
      });
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [history.length]);

  return {
    game: gameRef.current,
    fen,
    history,
    lastMove,
    displayGame,
    displayFen: displayGame.fen(),
    displayLastMove,
    visibleHistory,
    displayedPly,
    isReviewingHistory,
    isAiThinking,
    statusMessage,
    errorMessage,
    makePlayerMove,
    resetGame,
    resignGame,
    hasResigned,
    useSafetyNet,
    setUseSafetyNet,
    isPlayerTurn: playerColor ? gameRef.current.turn() === playerColor && !isAiThinking && !hasResigned : false,
  };
}

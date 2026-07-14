/**
 * Pure helper functions around a chess.js `Chess` instance. Kept free of
 * React so they're trivially testable and reusable outside components.
 */

/**
 * Build a react-chessboard `customSquareStyles` map highlighting every
 * legal destination square for a piece on `square`.
 *
 * @param {import("chess.js").Chess} game
 * @param {string} square - Origin square, e.g. "e2".
 * @returns {Record<string, React.CSSProperties>}
 */
export function getLegalMoveSquareStyles(game, square) {
  const moves = game.moves({ square, verbose: true });
  const styles = {};

  moves.forEach((move) => {
    const isCapture = Boolean(move.captured);
    styles[move.to] = {
      background: isCapture
        ? "radial-gradient(circle, transparent 55%, rgba(179,84,63,0.55) 56%)"
        : "radial-gradient(circle, rgba(201,162,39,0.55) 22%, transparent 23%)",
      borderRadius: "50%",
    };
  });

  return styles;
}

/**
 * Style for the origin/destination squares of the most recently played move.
 *
 * @param {{from: string, to: string} | null} lastMove
 * @returns {Record<string, React.CSSProperties>}
 */
export function getLastMoveSquareStyles(lastMove) {
  if (!lastMove) return {};
  const highlight = { backgroundColor: "rgba(201, 162, 39, 0.35)" };
  return {
    [lastMove.from]: highlight,
    [lastMove.to]: highlight,
  };
}

/**
 * Style for the king's square when in check.
 *
 * @param {import("chess.js").Chess} game
 * @returns {Record<string, React.CSSProperties>}
 */
export function getCheckSquareStyles(game) {
  if (!game.inCheck()) return {};

  const turnColor = game.turn(); // 'w' | 'b' — the side currently in check
  const board = game.board();

  for (let rank = 0; rank < 8; rank += 1) {
    for (let file = 0; file < 8; file += 1) {
      const piece = board[rank][file];
      if (piece && piece.type === "k" && piece.color === turnColor) {
        const files = "abcdefgh";
        const square = `${files[file]}${8 - rank}`;
        return {
          [square]: {
            background:
              "radial-gradient(circle, rgba(179,84,63,0.65) 0%, rgba(179,84,63,0.15) 70%)",
          },
        };
      }
    }
  }
  return {};
}

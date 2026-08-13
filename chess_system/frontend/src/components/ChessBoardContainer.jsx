import React, { useState } from "react";
import { Chessboard } from "react-chessboard";
import { TouchBackend } from "react-dnd-touch-backend";
import PromotionDialog from "./PromotionDialog.jsx";
import {
  getLegalMoveSquareStyles,
  getLastMoveSquareStyles,
  getCheckSquareStyles,
  isPromotionMove,
} from "../utils/chessHelpers.js";

// react-chessboard defaults to react-dnd's HTML5Backend, which drives drag
// via the browser's *native* HTML5 Drag and Drop API. That native API is
// what draws the small dashed-rectangle cursor icon next to the pointer —
// it's an OS/browser-level overlay, not something CSS can touch. Swapping
// in TouchBackend makes react-dnd drive drag entirely from mouse/touch/
// pointer events instead, so the native drag API (and its cursor icon)
// never gets invoked in the first place.
const dndBackendOptions = { enableMouseEvents: true };

/**
 * The board itself, framed in glass, with click-to-highlight legal moves,
 * drag-and-drop, last-move and check highlighting, and orientation locked
 * to the human's chosen color.
 *
 * @param {{
 *   game: import("chess.js").Chess,
 *   fen: string,
 *   playerColor: "w"|"b",
 *   lastMove: {from: string, to: string}|null,
 *   isLocked: boolean,
 *   onMove: (from: string, to: string, promotion?: "q"|"r"|"b"|"n") => boolean,
 * }} props
 */
export default function ChessBoardContainer({ game, fen, playerColor, lastMove, isLocked, onMove }) {
  const [selectedSquare, setSelectedSquare] = useState(/** @type {string|null} */ (null));
  const [pendingPromotion, setPendingPromotion] = useState(
    /** @type {{from: string, to: string, color: "w"|"b"}|null} */ (null)
  );

  const attemptMove = (from, to) => {
    if (isPromotionMove(game, from, to)) {
      // Defer the actual chess.js move until the player picks a piece in
      // PromotionDialog — `game` isn't mutated yet, so returning without
      // calling onMove here just leaves the board as-is.
      const movingPiece = game.get(from);
      setPendingPromotion({ from, to, color: movingPiece?.color ?? playerColor });
      return true; // treat as "handled" so react-chessboard doesn't snap the piece back
    }
    return onMove(from, to);
  };

  const handleSquareClick = (square) => {
    if (isLocked || pendingPromotion) return;

    if (selectedSquare) {
      const handled = attemptMove(selectedSquare, square);
      setSelectedSquare(handled ? null : squareHasOwnPiece(game, square, playerColor) ? square : null);
      return;
    }

    if (squareHasOwnPiece(game, square, playerColor)) {
      setSelectedSquare(square);
    }
  };

  const handlePieceDrop = (sourceSquare, targetSquare) => {
    if (isLocked || pendingPromotion) return false;
    const handled = attemptMove(sourceSquare, targetSquare);
    setSelectedSquare(null);
    return handled;
  };

  const handlePromotionSelect = (promotionPiece) => {
    if (!pendingPromotion) return;
    onMove(pendingPromotion.from, pendingPromotion.to, promotionPiece);
    setPendingPromotion(null);
  };

  const handlePromotionCancel = () => {
    setPendingPromotion(null);
  };

  const squareStyles = {
    ...getLastMoveSquareStyles(lastMove),
    ...getCheckSquareStyles(game),
    ...(selectedSquare
      ? { [selectedSquare]: { backgroundColor: "rgba(201, 162, 39, 0.45)" } }
      : {}),
    ...(selectedSquare ? getLegalMoveSquareStyles(game, selectedSquare) : {}),
  };

  return (
    // NOTE: intentionally NOT using <GlassPanel> as the direct ancestor here.
    // react-chessboard renders the dragged piece in a `position: fixed`
    // layer positioned via viewport coordinates. CSS spec: any ancestor
    // with `backdrop-filter`/`filter`/`transform` becomes the containing
    // block for `position: fixed` descendants instead of the viewport —
    // which sent the dragged piece flying to the wrong spot on screen.
    // Fix: the blurred glass surface is a sibling `::behind` layer, not a
    // parent of the board, so nothing above the board in the DOM has a
    // filter and the drag layer's fixed positioning stays viewport-relative.
    <div style={{ position: "relative", borderRadius: "var(--radius-lg)" }}>
      <div
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          background: "var(--color-glass)",
          backdropFilter: "blur(18px)",
          WebkitBackdropFilter: "blur(18px)",
          border: "1px solid var(--color-hairline)",
          borderRadius: "var(--radius-lg)",
          boxShadow: "var(--shadow-panel)",
          zIndex: 0,
        }}
      />
      <div style={{ position: "relative", zIndex: 1, padding: "1.5rem" }}>
        {isLocked && !pendingPromotion && (
          <div
            style={{
              position: "absolute",
              inset: "1.5rem",
              zIndex: 5,
              cursor: "not-allowed",
            }}
            aria-hidden
          />
        )}
        {pendingPromotion && (
          <PromotionDialog
            color={pendingPromotion.color}
            onSelect={handlePromotionSelect}
            onCancel={handlePromotionCancel}
          />
        )}
        <div style={{ width: "min(72vw, 560px)" }}>
          <Chessboard
            id="fischer-board"
            position={fen}
            boardOrientation={playerColor === "b" ? "black" : "white"}
            onPieceDrop={handlePieceDrop}
            onSquareClick={handleSquareClick}
            arePiecesDraggable={!isLocked && !pendingPromotion}
            onPromotionCheck={() => false}
            customDndBackend={TouchBackend}
            customDndBackendOptions={dndBackendOptions}
            customSquareStyles={squareStyles}
            customBoardStyle={{
              borderRadius: "10px",
              boxShadow: "0 12px 30px rgba(0,0,0,0.5)",
            }}
            customDarkSquareStyle={{ backgroundColor: "var(--color-walnut)" }}
            customLightSquareStyle={{ backgroundColor: "var(--color-ivory)" }}
          />
        </div>
      </div>
    </div>
  );
}

function squareHasOwnPiece(game, square, playerColor) {
  const piece = game.get(square);
  return Boolean(piece && piece.color === playerColor);
}

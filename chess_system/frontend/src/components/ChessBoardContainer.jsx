import React, { useState } from "react";
import { Chessboard } from "react-chessboard";
import GlassPanel from "./GlassPanel.jsx";
import {
  getLegalMoveSquareStyles,
  getLastMoveSquareStyles,
  getCheckSquareStyles,
} from "../utils/chessHelpers.js";

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
 *   onMove: (from: string, to: string) => boolean,
 * }} props
 */
export default function ChessBoardContainer({ game, fen, playerColor, lastMove, isLocked, onMove }) {
  const [selectedSquare, setSelectedSquare] = useState(/** @type {string|null} */ (null));

  const handleSquareClick = (square) => {
    if (isLocked) return;

    if (selectedSquare) {
      const moved = onMove(selectedSquare, square);
      setSelectedSquare(moved ? null : squareHasOwnPiece(game, square, playerColor) ? square : null);
      return;
    }

    if (squareHasOwnPiece(game, square, playerColor)) {
      setSelectedSquare(square);
    }
  };

  const handlePieceDrop = (sourceSquare, targetSquare) => {
    if (isLocked) return false;
    const moved = onMove(sourceSquare, targetSquare);
    setSelectedSquare(null);
    return moved;
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
    // 1. Root wrapper: Plain div with relative positioning. NO glass effects here.
    <div style={{ padding: "1.5rem", position: "relative" }}>
      
      {/* 2. The Glass background: Isolated and sent to the back (-z) */}
      <div style={{ position: "absolute", inset: 0, zIndex: -1 }}>
        {/* Pass an empty GlassPanel that stretches to fill the absolute parent */}
        <GlassPanel style={{ width: "100%", height: "100%", padding: 0 }} />
      </div>

      {/* 3. Board Content: Safely decoupled from backdrop-filter */}
      {isLocked && (
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
      <div style={{ width: "min(72vw, 560px)" }}>
        <Chessboard
          id="fischer-board"
          position={fen}
          boardOrientation={playerColor === "b" ? "black" : "white"}
          onPieceDrop={handlePieceDrop}
          onSquareClick={handleSquareClick}
          arePiecesDraggable={!isLocked}
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
  );
}

function squareHasOwnPiece(game, square, playerColor) {
  const piece = game.get(square);
  return Boolean(piece && piece.color === playerColor);
}

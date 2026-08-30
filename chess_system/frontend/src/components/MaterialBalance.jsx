import React, { useMemo } from "react";
import GlassPanel from "./GlassPanel.jsx";

const PIECE_VALUES = { p: 1, n: 3, b: 3, r: 5, q: 9 };
const PIECE_SYMBOLS = {
  w: { p: "♙", n: "♘", b: "♗", r: "♖", q: "♕" },
  b: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛" },
};

/**
 * Show one side's captured material above or below the board.
 *
 * @param {{history: {color: "w"|"b", captured?: string}[], playerColor: "w"|"b", position: "top"|"bottom"}} props
 * @returns {React.JSX.Element} Captured-piece and material-balance display.
 */
export default function MaterialBalance({ history, playerColor, position }) {
  const material = useMemo(() => calculateMaterial(history, playerColor), [history, playerColor]);
  const isTop = position === "top";
  const pieces = isTop ? material.aiCaptures : material.playerCaptures;
  const capturedColor = isTop ? playerColor : oppositeColor(playerColor);
  const label = isTop ? "Fischer captured" : "You captured";
  const advantage = isTop ? Math.max(-material.balance, 0) : Math.max(material.balance, 0);

  return (
    <GlassPanel style={{ width: "calc(min(82vw, 740px) + 48px)", padding: "0.68rem 1.2rem", display: "flex", alignItems: "center", justifyContent: "space-between", gap: "1rem" }}>
      <CapturedRow label={label} pieces={pieces} color={capturedColor} />
      {advantage > 0 && <strong style={{ color: "var(--color-success)", fontFamily: "var(--font-mono)", fontSize: "0.82rem" }}>+{advantage}</strong>}
    </GlassPanel>
  );
}

function CapturedRow({ label, pieces, color }) {
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <span style={captionStyle}>{label}</span>
      <div className="captured-piece-strip">
        {pieces.length ? pieces.map((piece, index) => (
          <span
            className={`captured-piece-icon captured-piece-icon--${color}`}
            key={`${piece}-${index}`}
            aria-label={`Captured ${piece}`}
          >
            {PIECE_SYMBOLS[color][piece]}
          </span>
        )) : null}
      </div>
    </div>
  );
}

function calculateMaterial(history, playerColor) {
  const playerCaptures = [];
  const aiCaptures = [];
  let playerValue = 0;
  let aiValue = 0;

  history.forEach((move) => {
    if (!move.captured || !PIECE_VALUES[move.captured]) return;
    if (move.color === playerColor) {
      playerCaptures.push(move.captured);
      playerValue += PIECE_VALUES[move.captured];
    } else {
      aiCaptures.push(move.captured);
      aiValue += PIECE_VALUES[move.captured];
    }
  });

  return { playerCaptures, aiCaptures, balance: playerValue - aiValue };
}

function oppositeColor(color) {
  return color === "w" ? "b" : "w";
}

const captionStyle = { display: "block", fontFamily: "var(--font-mono)", fontSize: "0.6rem", letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--color-text-muted)", marginBottom: "0.18rem" };

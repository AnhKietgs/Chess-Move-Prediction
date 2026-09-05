import React, { useEffect, useRef } from "react";
import GlassPanel from "./GlassPanel.jsx";

/**
 * Right-hand sidebar rendering standard algebraic notation as a
 * brass-ruled ledger — the signature element of the design: it reads
 * like a tournament scoresheet rather than a chat log.
 *
 * @param {{history: {moveNumber: number, san: string, color: "w"|"b"}[], displayedPly: number}} props
 */
export default function MoveHistory({ history, displayedPly }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [history.length]);

  // Group into rows of [white move, black move] keyed by move number.
  const rows = [];
  history.forEach((entry) => {
    const rowIndex = entry.moveNumber - 1;
    if (!rows[rowIndex]) rows[rowIndex] = { moveNumber: entry.moveNumber, w: "", b: "" };
    rows[rowIndex][entry.color] = entry.san;
  });

  return (
    <GlassPanel
      style={{ width: 320, flex: 1, minHeight: 0, display: "flex", flexDirection: "column", padding: "1.35rem" }}
    >
      <h2 style={{ fontSize: "1.12rem", color: "var(--color-brass-bright)", marginBottom: "0.25rem" }}>
        Scoresheet
      </h2>
      <p
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.68rem",
          color: "var(--color-text-muted)",
          letterSpacing: "0.06em",
          margin: "0 0 1rem",
        }}
      >
        {history.length === 0 ? "No moves yet" : `${history.length} ply recorded`}
      </p>

      <div
        ref={scrollRef}
        style={{
          flex: 1,
          overflowY: "auto",
          borderTop: "1px solid var(--color-hairline)",
        }}
      >
        {rows.map((row) => (
          <div
            key={row.moveNumber}
            style={{
              display: "grid",
              gridTemplateColumns: "2rem 1fr 1fr",
              gap: "0.5rem",
              padding: "0.4rem 0",
              borderBottom: "1px solid rgba(201,162,39,0.08)",
              fontFamily: "var(--font-mono)",
              fontSize: "0.92rem",
            }}
          >
            <span style={{ color: "var(--color-text-muted)" }}>{row.moveNumber}.</span>
            <span style={moveCellStyle(displayedPly === row.moveNumber * 2 - 1, "var(--color-ivory)")} title={castleTitle(row.w)}>{formatSan(row.w)}</span>
            <span style={moveCellStyle(displayedPly === row.moveNumber * 2, "var(--color-slate)")} title={castleTitle(row.b)}>{formatSan(row.b)}</span>
          </div>
        ))}
      </div>
    </GlassPanel>
  );
}

function formatSan(san) {
  return san.replace(/^0-0-0/, "O-O-O").replace(/^0-0/, "O-O");
}

function castleTitle(san) {
  if (/^(O|0)-(O|0)-(O|0)/.test(san)) return "Queenside castling";
  if (/^(O|0)-(O|0)/.test(san)) return "Kingside castling";
  return undefined;
}

function moveCellStyle(isActive, color) {
  return {
    background: isActive ? "rgba(201,162,39,0.25)" : "transparent",
    borderRadius: "3px",
    color,
    padding: "0.08rem 0.15rem",
  };
}
